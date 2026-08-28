from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


HISTORY_COLUMNS = [
    "date", "code", "name", "open", "high", "low", "close", "volume", "amount",
    "turnover", "pct_change", "tradestatus", "is_st", "pe", "pb",
]


class DemoHistoryProvider:
    name = "demo"

    def load(self, codes: Sequence[str], names: Dict[str, str], start: date, end: date):
        requested = list(dict.fromkeys(codes)) or [f"{600000 + index:06d}" for index in range(30)]
        if len(requested) < 12:
            requested.extend(f"{601000 + index:06d}" for index in range(12 - len(requested)))
        warmup_start = start - timedelta(days=150)
        dates = pd.bdate_range(warmup_start, end)
        rng = np.random.default_rng(20260825)
        market = rng.normal(0.00025, 0.010, len(dates))
        rows = []
        for index, code in enumerate(requested):
            quality = (index % 7 - 3) * 0.00008
            cycle = np.sin(np.arange(len(dates)) / (18 + index % 9)) * 0.0014
            returns = market * (0.65 + (index % 5) * 0.08) + cycle + quality + rng.normal(0, 0.008, len(dates))
            close = (10 + index * 0.9) * np.exp(np.cumsum(returns))
            open_price = close * (1 + rng.normal(0, 0.003, len(dates)))
            high = np.maximum(open_price, close) * (1 + rng.uniform(0.002, 0.025, len(dates)))
            low = np.minimum(open_price, close) * (1 - rng.uniform(0.002, 0.025, len(dates)))
            volume = rng.lognormal(16.0 + (index % 4) * 0.15, 0.35, len(dates))
            amount = volume * close
            pe = np.clip(12 + index * 0.8 + rng.normal(0, 1.2, len(dates)), 4, 90)
            pb = np.clip(1.0 + index * 0.08 + rng.normal(0, 0.08, len(dates)), 0.4, 12)
            for position, trade_date in enumerate(dates):
                rows.append({
                    "date": trade_date,
                    "code": code,
                    "name": names.get(code, f"回测样例{index + 1:02d}"),
                    "open": open_price[position],
                    "high": high[position],
                    "low": low[position],
                    "close": close[position],
                    "volume": volume[position],
                    "amount": amount[position],
                    "turnover": 2.0 + index % 6,
                    "pct_change": returns[position] * 100,
                    "tradestatus": 1,
                    "is_st": 0,
                    "pe": pe[position],
                    "pb": pb[position],
                })
        return pd.DataFrame(rows, columns=HISTORY_COLUMNS), []


class BaoStockHistoryProvider:
    name = "live"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "backtest"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, codes: Sequence[str], names: Dict[str, str], start: date, end: date):
        import baostock as bs

        warmup_start = start - timedelta(days=150)
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
        frames: List[pd.DataFrame] = []
        warnings: List[str] = []
        try:
            for code in dict.fromkeys(codes):
                try:
                    frame = self._load_symbol(bs, code, names.get(code, code), warmup_start, end)
                    if len(frame) >= 65:
                        frames.append(frame)
                    else:
                        warnings.append(f"{code} 历史数据不足，已跳过")
                except Exception as exc:
                    warnings.append(f"{code} 获取失败: {exc}")
        finally:
            bs.logout()
        if len(frames) < 2:
            raise RuntimeError("可用历史股票不足 2 只，无法进行横截面回测")
        return pd.concat(frames, ignore_index=True), warnings

    def _load_symbol(self, bs, code: str, name: str, start: date, end: date):
        cache_path = self.cache_dir / f"{code}_{start:%Y%m%d}_{end:%Y%m%d}_qfq.csv"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, dtype={"code": str})
            frame["date"] = pd.to_datetime(frame["date"])
            return frame

        market_code = self._market_code(code)
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg,tradestatus,isST,peTTM,pbMRQ"
        result = bs.query_history_k_data_plus(
            market_code,
            fields,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag="2",
        )
        if result.error_code != "0":
            raise RuntimeError(result.error_msg)
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=result.fields)
        if raw.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        frame = pd.DataFrame({
            "date": pd.to_datetime(raw["date"]),
            "code": code,
            "name": name,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
            "amount": pd.to_numeric(raw["amount"], errors="coerce"),
            "turnover": pd.to_numeric(raw["turn"], errors="coerce"),
            "pct_change": pd.to_numeric(raw["pctChg"], errors="coerce"),
            "tradestatus": pd.to_numeric(raw["tradestatus"], errors="coerce").fillna(0),
            "is_st": pd.to_numeric(raw["isST"], errors="coerce").fillna(0),
            "pe": pd.to_numeric(raw["peTTM"], errors="coerce"),
            "pb": pd.to_numeric(raw["pbMRQ"], errors="coerce"),
        }).dropna(subset=["date", "open", "close"])
        frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return frame

    @staticmethod
    def _market_code(code: str) -> str:
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        if code.startswith(("0", "3")):
            return f"sz.{code}"
        raise ValueError("当前历史源仅支持沪深 A 股代码")


def parse_codes(value: Sequence[str]) -> List[str]:
    output = []
    for raw in value:
        code = str(raw).strip().lower().replace("sh.", "").replace("sz.", "")
        if len(code) == 6 and code.isdigit() and code.startswith(("0", "3", "6", "9")):
            output.append(code)
    return list(dict.fromkeys(output))

