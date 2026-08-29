from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import httpx
import numpy as np
import pandas as pd

from app.storage import Storage


TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


@dataclass(frozen=True)
class CSI300HedgeConfig:
    lookback: int = 120
    threshold: float = 1.0
    hedge_ratio: float = 0.50
    change_cost_bps: float = 2.0

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("hedge lookback must be at least two sessions")
        if self.threshold <= 0:
            raise ValueError("hedge threshold must be positive")
        if not 0 <= self.hedge_ratio <= 1:
            raise ValueError("hedge ratio must be between zero and one")
        if self.change_cost_bps < 0:
            raise ValueError("hedge change cost cannot be negative")


class CSI300IndexProvider:
    """Load CSI300 OHLC through the signal date with a local fallback cache."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)

    def load(self, as_of: date) -> tuple[pd.DataFrame, Dict]:
        start = as_of - timedelta(days=900)
        params = {
            "param": (
                f"sh000300,day,{start.isoformat()},{as_of.isoformat()},1000,qfq"
            )
        }
        try:
            with httpx.Client(
                timeout=20,
                trust_env=False,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            ) as client:
                response = client.get(TENCENT_KLINE_URL, params=params)
                response.raise_for_status()
                payload = response.json()
            node = payload.get("data", {}).get("sh000300", {})
            rows = node.get("qfqday") or node.get("day") or []
            if len(rows) < 150:
                raise RuntimeError("CSI300 live response has insufficient history")
            frame = pd.DataFrame(rows).iloc[:, :6]
            frame.columns = ["date", "open", "close", "high", "low", "volume"]
            frame = self._clean(frame)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.reset_index().to_csv(
                self.cache_path, index=False, encoding="utf-8-sig", float_format="%.6f"
            )
            source = {
                "mode": "live_tencent_api",
                "index": "CSI300",
                "fetched_through": frame.index[-1].date().isoformat(),
            }
        except Exception as exc:
            if not self.cache_path.exists():
                raise RuntimeError(f"CSI300 data unavailable: {exc}") from exc
            frame = self._clean(pd.read_csv(self.cache_path))
            source = {
                "mode": "cached_fallback",
                "index": "CSI300",
                "reason": str(exc),
                "fetched_through": frame.index[-1].date().isoformat(),
            }
        frame = frame.loc[: pd.Timestamp(as_of)]
        if len(frame) < 2:
            raise RuntimeError("CSI300 history is empty through the stock signal date")
        return frame, source

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        clean = frame.copy()
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
        for column in ("open", "close"):
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        return (
            clean.dropna(subset=["date", "open", "close"])
            .loc[lambda item: item["open"].gt(0) & item["close"].gt(0)]
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .set_index("date")[["open", "close"]]
        )


class CSI300HedgePaperService:
    """Paper ledger for a causal CSI300 futures hedge over a stock account."""

    def __init__(
        self,
        storage: Storage,
        cache_path: Path,
        *,
        account_id: str,
        initial_capital: float = 1_000_000,
        config: CSI300HedgeConfig | None = None,
        provider: CSI300IndexProvider | None = None,
        strategy_label: str | None = None,
    ):
        if initial_capital <= 0:
            raise ValueError("hedged paper capital must be positive")
        self.storage = storage
        self.account_id = str(account_id)
        self.initial_capital = float(initial_capital)
        self.config = config or CSI300HedgeConfig()
        self.provider = provider or CSI300IndexProvider(cache_path)
        self.strategy_label = strategy_label

    def update(self, stock_snapshot: Dict, signal_date: str) -> Dict:
        if not signal_date:
            raise ValueError("hedge paper account requires a signal date")
        state = self.storage.load_strategy_state(self.account_id) or self._initial_state()
        if state.get("last_signal_date") == signal_date and state.get("last_snapshot"):
            return deepcopy(state["last_snapshot"])
        if state.get("last_signal_date") and signal_date < state["last_signal_date"]:
            raise ValueError("hedge signal date is earlier than the paper ledger")

        as_of = date.fromisoformat(signal_date)
        index, source = self.provider.load(as_of)
        latest_date = index.index[-1].date().isoformat()
        if latest_date != signal_date:
            raise RuntimeError(
                f"CSI300 data is stale at {latest_date}; stock signal is {signal_date}"
            )
        if len(index) < self.config.lookback:
            raise RuntimeError("CSI300 history is insufficient for the frozen MA rule")

        close = float(index["close"].iloc[-1])
        moving_average = float(index["close"].iloc[-self.config.lookback :].mean())
        trend_ratio = close / moving_average
        target_ratio = self.config.hedge_ratio if trend_ratio < self.config.threshold else 0.0

        active_ratio = float(state.get("active_hedge_ratio", 0.0))
        pending_ratio = float(state.get("pending_hedge_ratio", active_ratio))
        previous_nav = float(state.get("last_composite_nav", self.initial_capital))
        hedge_equity = float(state.get("hedge_equity", 0.0))
        hedge_pnl = 0.0
        hedge_cost = 0.0
        executed_change = 0.0
        last_open_date = state.get("last_index_open_date")
        last_open_price = state.get("last_index_open_price")

        if last_open_date and last_open_price:
            new_rows = index.loc[index.index > pd.Timestamp(last_open_date)]
            prior_open = float(last_open_price)
            for row_number, (_, row) in enumerate(new_rows.iterrows()):
                current_open = float(row["open"])
                open_return = current_open / prior_open - 1.0
                hedge_pnl -= active_ratio * previous_nav * open_return
                if row_number == 0:
                    executed_change = abs(pending_ratio - active_ratio)
                    hedge_cost = (
                        executed_change
                        * previous_nav
                        * self.config.change_cost_bps
                        / 10_000.0
                    )
                    active_ratio = pending_ratio
                prior_open = current_open

        hedge_equity += hedge_pnl - hedge_cost
        stock_nav = float(stock_snapshot.get("nav", self.initial_capital))
        composite_nav = stock_nav + hedge_equity
        high_water = max(float(state.get("high_water", self.initial_capital)), composite_nav)
        total_cost = float(state.get("hedge_cost_total", 0.0)) + hedge_cost
        action = self._action_label(active_ratio, target_ratio)
        warnings = [
            "CSI300 hedge is a futures proxy; basis, margin, roll and financing are not modeled."
        ]
        snapshot = {
            "account_id": self.account_id,
            "strategy": self.strategy_label or (
                "75/25 Alpha158-Barra LGBM + "
                f"CSI300 dynamic hedge {self.config.hedge_ratio:.0%} paper"
            ),
            "status": "active_paper_proxy",
            "signal_date": signal_date,
            "index_data_date": latest_date,
            "stock_nav": round(stock_nav, 2),
            "hedge_equity": round(hedge_equity, 2),
            "nav": round(composite_nav, 2),
            "daily_pnl": round(composite_nav - previous_nav, 2),
            "daily_return": composite_nav / previous_nav - 1.0 if previous_nav > 0 else 0.0,
            "cumulative_return": composite_nav / self.initial_capital - 1.0,
            "drawdown": composite_nav / high_water - 1.0 if high_water > 0 else 0.0,
            "hedge_pnl_today": round(hedge_pnl, 2),
            "hedge_cost_today": round(hedge_cost, 2),
            "hedge_cost_total": round(total_cost, 2),
            "active_hedge_ratio": active_ratio,
            "target_hedge_ratio": target_ratio,
            "executed_ratio_change": executed_change,
            "target_effective_at": "next_trading_day_open",
            "action_label": action,
            "index_close": close,
            "index_ma": moving_average,
            "index_trend_ratio": trend_ratio,
            "parameters": {
                "index": "CSI300",
                "lookback": self.config.lookback,
                "threshold": self.config.threshold,
                "maximum_hedge_ratio": self.config.hedge_ratio,
                "change_cost_bps": self.config.change_cost_bps,
                "signal_at": "close",
                "fill_at": "next_open",
                "pnl_window": "open_to_next_open",
            },
            "source": source,
            "warnings": warnings,
        }
        new_state = {
            "version": 1,
            "account_id": self.account_id,
            "last_signal_date": signal_date,
            "last_index_open_date": latest_date,
            "last_index_open_price": float(index["open"].iloc[-1]),
            "active_hedge_ratio": active_ratio,
            "pending_hedge_ratio": target_ratio,
            "hedge_equity": hedge_equity,
            "hedge_cost_total": total_cost,
            "last_composite_nav": composite_nav,
            "high_water": high_water,
            "last_snapshot": deepcopy(snapshot),
        }
        self.storage.save_strategy_snapshot(
            self.account_id, signal_date, new_state, snapshot
        )
        return snapshot

    @staticmethod
    def _action_label(active_ratio: float, target_ratio: float) -> str:
        if target_ratio > active_ratio:
            return f"下一交易日开盘将对冲提高至 {target_ratio:.0%}"
        if target_ratio < active_ratio:
            return f"下一交易日开盘将对冲降低至 {target_ratio:.0%}"
        return f"下一交易日开盘维持 {target_ratio:.0%} 对冲"

    def _initial_state(self) -> Dict:
        return {
            "version": 1,
            "account_id": self.account_id,
            "last_signal_date": None,
            "last_index_open_date": None,
            "last_index_open_price": None,
            "active_hedge_ratio": 0.0,
            "pending_hedge_ratio": 0.0,
            "hedge_equity": 0.0,
            "hedge_cost_total": 0.0,
            "last_composite_nav": self.initial_capital,
            "high_water": self.initial_capital,
            "last_snapshot": None,
        }
