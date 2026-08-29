from __future__ import annotations

from copy import deepcopy
from typing import Dict

from app.storage import Storage


class IndependentSleeveMonitorService:
    """Aggregate independent paper sleeves without cross-sleeve rebalancing."""

    ACCOUNT_ID = "trend75_lgbm25_independent_monitor_v1"

    def __init__(
        self,
        storage: Storage,
        *,
        initial_capital: float,
        trend_initial_weight: float = 0.75,
        lgbm_initial_weight: float = 0.25,
        account_id: str = ACCOUNT_ID,
    ):
        if initial_capital <= 0:
            raise ValueError("组合模拟盘初始资金必须大于0")
        if abs(trend_initial_weight + lgbm_initial_weight - 1.0) > 1e-9:
            raise ValueError("两个袖套初始权重之和必须为1")
        self.storage = storage
        self.initial_capital = float(initial_capital)
        self.trend_initial_weight = float(trend_initial_weight)
        self.lgbm_initial_weight = float(lgbm_initial_weight)
        self.account_id = str(account_id)

    def update(
        self,
        *,
        signal_date: str,
        trend_stock: Dict,
        trend_hedged: Dict,
        trend_decision: Dict,
        trend_regime: Dict,
        lgbm_stock: Dict,
        lgbm_decision: Dict,
    ) -> Dict:
        if not signal_date:
            raise ValueError("组合模拟盘缺少信号日期")
        state = self.storage.load_strategy_state(self.account_id) or {}
        if state.get("last_signal_date") == signal_date and state.get("last_snapshot"):
            return deepcopy(state["last_snapshot"])
        if state.get("last_signal_date") and signal_date < state["last_signal_date"]:
            raise ValueError("组合模拟盘信号日期早于账本日期")

        trend_nav = float(trend_hedged["nav"])
        lgbm_nav = float(lgbm_stock["nav"])
        nav = trend_nav + lgbm_nav
        previous_nav = float(state.get("last_nav", self.initial_capital))
        high_water = max(float(state.get("high_water", self.initial_capital)), nav)
        trend_actual_weight = trend_nav / nav if nav > 0 else 0.0
        lgbm_actual_weight = lgbm_nav / nav if nav > 0 else 0.0
        stock_cost = float(trend_stock.get("transaction_cost_total", 0.0)) + float(
            lgbm_stock.get("transaction_cost_total", 0.0)
        )
        hedge_cost = float(trend_hedged.get("hedge_cost_total", 0.0))

        snapshot = {
            "account_id": self.account_id,
            "strategy": "75%趋势专家 + 25%正式LGBM独立袖套",
            "status": "active_forward_paper",
            "signal_date": signal_date,
            "initial_capital": self.initial_capital,
            "nav": round(nav, 2),
            "daily_pnl": round(nav - previous_nav, 2),
            "daily_return": nav / previous_nav - 1.0 if previous_nav > 0 else 0.0,
            "cumulative_pnl": round(nav - self.initial_capital, 2),
            "cumulative_return": nav / self.initial_capital - 1.0,
            "high_water": round(high_water, 2),
            "drawdown": nav / high_water - 1.0 if high_water > 0 else 0.0,
            "transaction_cost_total": round(stock_cost + hedge_cost, 2),
            "stock_transaction_cost_total": round(stock_cost, 2),
            "hedge_cost_total": round(hedge_cost, 2),
            "sleeve_rebalance": "none",
            "cash_transfer_between_sleeves": 0.0,
            "sleeves": {
                "trend": {
                    "label": "趋势专家 + CSI300 MA120动态对冲",
                    "initial_weight": self.trend_initial_weight,
                    "initial_capital": self.initial_capital * self.trend_initial_weight,
                    "stock_nav": float(trend_stock["nav"]),
                    "nav": trend_nav,
                    "actual_weight": trend_actual_weight,
                    "cumulative_return": (
                        trend_nav / (self.initial_capital * self.trend_initial_weight) - 1.0
                    ),
                    "cash": float(trend_stock.get("cash", 0.0)),
                    "exposure": float(trend_stock.get("exposure", 0.0)),
                    "positions": deepcopy(trend_stock.get("positions", [])),
                    "trades": deepcopy(trend_stock.get("trades", [])),
                    "decision": deepcopy(trend_decision),
                    "regime": deepcopy(trend_regime),
                    "active_hedge_ratio": float(
                        trend_hedged.get("active_hedge_ratio", 0.0)
                    ),
                    "target_hedge_ratio": float(
                        trend_hedged.get("target_hedge_ratio", 0.0)
                    ),
                    "hedge_action": trend_hedged.get("action_label", "--"),
                    "index_close": trend_hedged.get("index_close"),
                    "index_ma": trend_hedged.get("index_ma"),
                },
                "lgbm": {
                    "label": "原正式75/25 Alpha158-Barra LGBM",
                    "initial_weight": self.lgbm_initial_weight,
                    "initial_capital": self.initial_capital * self.lgbm_initial_weight,
                    "nav": lgbm_nav,
                    "actual_weight": lgbm_actual_weight,
                    "cumulative_return": (
                        lgbm_nav / (self.initial_capital * self.lgbm_initial_weight) - 1.0
                    ),
                    "cash": float(lgbm_stock.get("cash", 0.0)),
                    "exposure": float(lgbm_stock.get("exposure", 0.0)),
                    "positions": deepcopy(lgbm_stock.get("positions", [])),
                    "trades": deepcopy(lgbm_stock.get("trades", [])),
                    "decision": deepcopy(lgbm_decision),
                },
            },
            "execution": {
                "signal_at": "close",
                "fill_at": "next_trading_day_open",
                "sleeve_rebalance": False,
                "historical_backfill": False,
            },
            "warnings": [
                "这是从启用日起记录的前向模拟盘，不回填历史收益。",
                "CSI300对冲是期货代理，未建模基差、保证金、移仓和融资。",
            ],
        }
        new_state = {
            "version": 1,
            "account_id": self.account_id,
            "last_signal_date": signal_date,
            "last_nav": nav,
            "high_water": high_water,
            "last_snapshot": deepcopy(snapshot),
        }
        self.storage.save_strategy_snapshot(
            self.account_id, signal_date, new_state, snapshot
        )
        return snapshot
