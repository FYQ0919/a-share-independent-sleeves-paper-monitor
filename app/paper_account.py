from __future__ import annotations

from copy import deepcopy
import math
from typing import Dict, Iterable

import pandas as pd

from app.storage import Storage


class LgbmPaperAccountService:
    ACCOUNT_ID = "lgbm_ranker_top5_paper_account_v1"

    def __init__(
        self,
        storage: Storage,
        initial_capital: float = 1_000_000,
        cost_bps: float = 12,
        lot_size: int = 100,
        account_id: str = ACCOUNT_ID,
        strategy_label: str = "LGBM LambdaRank Top5 模拟盘",
    ):
        if initial_capital <= 0:
            raise ValueError("模拟盘初始资金必须大于0")
        if cost_bps < 0:
            raise ValueError("模拟盘交易成本不能为负数")
        if lot_size <= 0:
            raise ValueError("模拟盘买入单位必须大于0")
        self.storage = storage
        self.initial_capital = float(initial_capital)
        self.cost_rate = float(cost_bps) / 10_000
        self.cost_bps = float(cost_bps)
        self.lot_size = int(lot_size)
        self.account_id = str(account_id)
        self.strategy_label = str(strategy_label)

    def tracked_symbols(self) -> Dict[str, str]:
        state = self.storage.load_strategy_state(self.account_id) or {}
        return {
            str(item["code"]): str(item.get("name") or item["code"])
            for item in state.get("positions", [])
        }

    def update(self, decision: Dict, history: pd.DataFrame) -> Dict:
        signal_date = str(decision.get("signal_date") or "")
        if not signal_date:
            raise ValueError("模拟盘缺少信号日期")
        state = self.storage.load_strategy_state(self.account_id) or self._initial_state()
        if state.get("last_signal_date") == signal_date and state.get("last_snapshot"):
            return deepcopy(state["last_snapshot"])
        if state.get("last_signal_date") and signal_date < state["last_signal_date"]:
            raise ValueError("模拟盘信号日期早于账本日期，已拒绝覆盖状态")

        positions = {
            str(item["code"]): deepcopy(item) for item in state.get("positions", [])
        }
        cash = float(state.get("cash", self.initial_capital))
        trades = []
        warnings = []
        executed = decision.get("executed_order") or {}
        if executed:
            cash, positions, trades, execution_warnings = self._execute_rebalance(
                cash,
                positions,
                decision.get("current_holdings") or [],
                history,
                str(executed["execution_date"]),
            )
            warnings.extend(execution_warnings)

        close_prices = self._prices_at_or_before(history, signal_date, "close")
        marked_positions = []
        market_value = 0.0
        unrealized_pnl = 0.0
        for code, position in sorted(positions.items()):
            price = float(close_prices.get(code, position.get("last_price", 0.0)))
            shares = int(position["shares"])
            value = shares * price
            market_value += value
            unrealized_pnl += (price - float(position["avg_cost"])) * shares
            position["last_price"] = price
            position["market_value"] = value
            position["unrealized_pnl"] = (price - float(position["avg_cost"])) * shares
            marked_positions.append(position)

        nav = cash + market_value
        previous_nav = float(state.get("last_nav", self.initial_capital))
        high_water = max(float(state.get("high_water", self.initial_capital)), nav)
        for position in marked_positions:
            position["weight"] = position["market_value"] / nav if nav > 0 else 0.0

        snapshot = {
            "account_id": self.account_id,
            "strategy": self.strategy_label,
            "signal_date": signal_date,
            "status": "active_paper",
            "initial_capital": self.initial_capital,
            "cash": round(cash, 2),
            "market_value": round(market_value, 2),
            "nav": round(nav, 2),
            "daily_pnl": round(nav - previous_nav, 2),
            "daily_return": nav / previous_nav - 1 if previous_nav > 0 else 0.0,
            "cumulative_pnl": round(nav - self.initial_capital, 2),
            "cumulative_return": nav / self.initial_capital - 1,
            "high_water": round(high_water, 2),
            "drawdown": nav / high_water - 1 if high_water > 0 else 0.0,
            "exposure": market_value / nav if nav > 0 else 0.0,
            "transaction_cost_total": round(
                float(state.get("transaction_cost_total", 0.0))
                + sum(float(item["fee"]) for item in trades),
                2,
            ),
            "realized_pnl_total": round(
                float(state.get("realized_pnl_total", 0.0))
                + sum(float(item.get("realized_pnl", 0.0)) for item in trades),
                2,
            ),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "positions": [self._public_position(item) for item in marked_positions],
            "trades": trades,
            "execution": {
                "signal_at": "close",
                "fill_at": "next_trading_day_open",
                "cost_bps_one_way": self.cost_bps,
                "buy_lot_size": self.lot_size,
                "fractional_shares": False,
            },
            "warnings": warnings,
        }
        new_state = {
            "version": 1,
            "account_id": self.account_id,
            "last_signal_date": signal_date,
            "cash": cash,
            "positions": marked_positions,
            "last_nav": nav,
            "high_water": high_water,
            "transaction_cost_total": snapshot["transaction_cost_total"],
            "realized_pnl_total": snapshot["realized_pnl_total"],
            "last_snapshot": deepcopy(snapshot),
        }
        self.storage.save_strategy_snapshot(self.account_id, signal_date, new_state, snapshot)
        return snapshot

    def _execute_rebalance(
        self,
        cash: float,
        positions: Dict[str, Dict],
        targets: Iterable[Dict],
        history: pd.DataFrame,
        execution_date: str,
    ) -> tuple[float, Dict[str, Dict], list[Dict], list[str]]:
        target_rows = {str(item["code"]): deepcopy(item) for item in targets}
        open_prices = self._prices_on(history, execution_date, "open")
        warnings = []
        for code in sorted(set(positions) | set(target_rows)):
            if code not in open_prices:
                warnings.append(f"{code} 在 {execution_date} 缺少有效开盘价，未调整该持仓")

        pre_trade_nav = cash + sum(
            int(item["shares"])
            * float(open_prices.get(code, item.get("last_price", 0.0)))
            for code, item in positions.items()
        )
        investable = max(0.0, pre_trade_nav * (1 - self.cost_rate))
        target_shares: Dict[str, int] = {}
        for code in sorted(set(positions) | set(target_rows)):
            current = int(positions.get(code, {}).get("shares", 0))
            price = open_prices.get(code)
            if price is None:
                target_shares[code] = current
                continue
            weight = float(target_rows.get(code, {}).get("target_weight", 0.0))
            target_shares[code] = (
                math.floor(investable * weight / (price * self.lot_size)) * self.lot_size
            )

        trades = []
        for code in sorted(target_shares):
            current = int(positions.get(code, {}).get("shares", 0))
            quantity = current - target_shares[code]
            if quantity <= 0 or code not in open_prices:
                continue
            price = float(open_prices[code])
            notional = quantity * price
            fee = notional * self.cost_rate
            position = positions[code]
            realized = (price - float(position["avg_cost"])) * quantity - fee
            cash += notional - fee
            remaining = current - quantity
            trades.append(self._trade_payload(
                execution_date, "SELL", code, position.get("name", code), quantity,
                price, fee, realized,
            ))
            if remaining:
                position["shares"] = remaining
                position["last_price"] = price
            else:
                positions.pop(code, None)

        for code in sorted(target_shares):
            current = int(positions.get(code, {}).get("shares", 0))
            desired = target_shares[code] - current
            if desired <= 0 or code not in open_prices:
                continue
            price = float(open_prices[code])
            affordable_lots = math.floor(cash / (price * (1 + self.cost_rate) * self.lot_size))
            quantity = min(desired, affordable_lots * self.lot_size)
            if quantity <= 0:
                warnings.append(f"{code} 模拟盘现金不足，未能完成目标买入")
                continue
            notional = quantity * price
            fee = notional * self.cost_rate
            cash -= notional + fee
            old = positions.get(code)
            old_shares = int(old.get("shares", 0)) if old else 0
            old_cost = float(old.get("avg_cost", 0.0)) * old_shares if old else 0.0
            name = str(target_rows.get(code, {}).get("name") or (old or {}).get("name") or code)
            new_shares = old_shares + quantity
            positions[code] = {
                "code": code,
                "name": name,
                "shares": new_shares,
                "avg_cost": (old_cost + notional + fee) / new_shares,
                "last_price": price,
            }
            trades.append(self._trade_payload(
                execution_date, "BUY", code, name, quantity, price, fee, 0.0,
            ))
        return cash, positions, trades, warnings

    @staticmethod
    def _prices_on(history: pd.DataFrame, when: str, column: str) -> Dict[str, float]:
        frame = history.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        selected = frame[frame["date"].eq(pd.Timestamp(when))].copy()
        if "tradestatus" in selected.columns:
            selected = selected[selected["tradestatus"].eq(1)]
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        selected = selected[selected[column].gt(0)].drop_duplicates("code", keep="last")
        return {str(row["code"]): float(row[column]) for _, row in selected.iterrows()}

    @staticmethod
    def _prices_at_or_before(history: pd.DataFrame, when: str, column: str) -> Dict[str, float]:
        frame = history.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        selected = (
            frame[frame["date"].le(pd.Timestamp(when)) & frame[column].gt(0)]
            .sort_values(["code", "date"])
            .drop_duplicates("code", keep="last")
        )
        return {str(row["code"]): float(row[column]) for _, row in selected.iterrows()}

    @staticmethod
    def _trade_payload(execution_date, side, code, name, shares, price, fee, realized):
        return {
            "execution_date": execution_date,
            "side": side,
            "code": code,
            "name": name,
            "shares": int(shares),
            "price": round(float(price), 4),
            "notional": round(float(shares) * float(price), 2),
            "fee": round(float(fee), 2),
            "realized_pnl": round(float(realized), 2),
        }

    @staticmethod
    def _public_position(item: Dict) -> Dict:
        return {
            "code": str(item["code"]),
            "name": str(item.get("name") or item["code"]),
            "shares": int(item["shares"]),
            "avg_cost": round(float(item["avg_cost"]), 4),
            "last_price": round(float(item["last_price"]), 4),
            "market_value": round(float(item["market_value"]), 2),
            "weight": float(item["weight"]),
            "unrealized_pnl": round(float(item["unrealized_pnl"]), 2),
        }

    def _initial_state(self) -> Dict:
        return {
            "version": 1,
            "account_id": self.account_id,
            "last_signal_date": None,
            "cash": self.initial_capital,
            "positions": [],
            "last_nav": self.initial_capital,
            "high_water": self.initial_capital,
            "transaction_cost_total": 0.0,
            "realized_pnl_total": 0.0,
            "last_snapshot": None,
        }
