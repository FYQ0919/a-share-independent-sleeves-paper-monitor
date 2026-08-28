from copy import deepcopy
from datetime import date
from typing import Dict, List, Sequence

from app.models import StrategyPick
from app.storage import Storage


class AdaptivePortfolioService:
    STRATEGY_ID = "contrarian_top5_adaptive"

    def __init__(
        self,
        storage: Storage,
        top_n: int = 5,
        rebalance_days: int = 10,
        min_hold_days: int = 5,
        rank_buffer: int = 20,
        score_gap: float = 15.0,
        entry_rank: int = 3,
        strategy_id: str = STRATEGY_ID,
        strategy_label: str = "逆向增强 Top5",
        adaptive_enabled: bool = True,
    ):
        self.storage = storage
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.min_hold_days = min_hold_days
        self.rank_buffer = rank_buffer
        self.score_gap = score_gap
        self.entry_rank = entry_rank
        self.strategy_id = strategy_id
        self.strategy_label = strategy_label
        self.adaptive_enabled = bool(adaptive_enabled)

    def tracked_symbols(self) -> Dict[str, str]:
        state = self.storage.load_strategy_state(self.strategy_id) or {}
        symbols: Dict[str, str] = {}
        for item in state.get("holdings", []):
            symbols[str(item["code"])] = str(item.get("name") or item["code"])
        pending = state.get("pending_order") or {}
        for item in pending.get("target_holdings", []):
            symbols[str(item["code"])] = str(item.get("name") or item["code"])
        return symbols

    def evaluate(
        self,
        rankings: Sequence[StrategyPick],
        signal_date: date,
        trading_dates: Sequence[date],
    ) -> Dict:
        if not rankings:
            raise ValueError("每日自适应策略没有可用排名")
        signal_text = signal_date.isoformat()
        state = self.storage.load_strategy_state(self.strategy_id) or self._initial_state()
        previous_signal = state.get("last_signal_date")
        if previous_signal == signal_text and state.get("last_decision"):
            return deepcopy(state["last_decision"])
        if previous_signal and signal_text < previous_signal:
            raise ValueError("策略信号日期早于持仓账本日期，已拒绝覆盖状态")

        calendar = sorted({item.isoformat() for item in trading_dates if item <= signal_date})
        elapsed = self._elapsed_sessions(previous_signal, signal_text, calendar)
        previous_sequence = int(state.get("signal_sequence", -1))
        current_sequence = previous_sequence + elapsed
        holdings = deepcopy(state.get("holdings", []))
        executed_order = None
        pending = deepcopy(state.get("pending_order"))
        if pending and pending.get("signal_date", "") < signal_text:
            execution_date = self._next_session(pending["signal_date"], calendar) or signal_text
            execution_offset = self._elapsed_sessions(previous_signal, execution_date, calendar)
            execution_sequence = previous_sequence + execution_offset
            holdings = self._materialize_holdings(
                pending.get("target_holdings", []), holdings, execution_sequence, execution_date
            )
            executed_order = {
                "signal_date": pending["signal_date"],
                "execution_date": execution_date,
                "action_label": pending["action_label"],
                "reason": pending["reason"],
                "buys": pending.get("buys", []),
                "sells": pending.get("sells", []),
            }
            pending = None

        ranking_rows = [self._pick_payload(item) for item in rankings]
        rank_by_code = {item["code"]: item["rank"] for item in ranking_rows}
        score_by_code = {item["code"]: item["score"] for item in ranking_rows}
        ranking_by_code = {item["code"]: item for item in ranking_rows}
        holdings = self._refresh_holding_marks(holdings, ranking_by_code, current_sequence)

        cycle_day = int(state.get("cycle_day", 0)) + elapsed if previous_signal else 0
        adaptive_used = int(state.get("adaptive_used_in_cycle", 0))
        scheduled = not holdings or cycle_day >= self.rebalance_days
        target_holdings = deepcopy(holdings)
        status = "hold"
        action_label = "继续持有"
        reason = "排名与分差均未触发换股"

        if scheduled:
            target_holdings = deepcopy(ranking_rows[: self.top_n])
            cycle_day = 0
            adaptive_used = 0
            status = "scheduled"
            action_label = "主调仓"
            reason = "首次建仓" if not holdings else f"第{self.rebalance_days}个交易日主调仓"
        elif not self.adaptive_enabled:
            reason = f"固定{self.rebalance_days}交易日调仓，周期内不提前换股"
        else:
            target_codes, status, reason = self._adaptive_target(
                ranking_rows,
                holdings,
                rank_by_code,
                score_by_code,
                current_sequence,
                adaptive_used < 1,
            )
            if status != "hold":
                holding_by_code = {item["code"]: item for item in holdings}
                target_holdings = [
                    deepcopy(ranking_by_code[code] if code in ranking_by_code else holding_by_code[code])
                    for code in target_codes
                    if code in ranking_by_code or code in holding_by_code
                ]
                if len(target_codes) != self.top_n or len(target_holdings) != self.top_n:
                    raise ValueError(
                        f"自适应换股目标必须保留 {self.top_n} 只股票，已拒绝生成不完整订单"
                    )
                action_label = "风险退出" if status == "risk" else "提前换股"
                if status == "adaptive":
                    adaptive_used += 1

        current_codes = [item["code"] for item in holdings]
        target_codes = [item["code"] for item in target_holdings]
        buys = [item for item in target_holdings if item["code"] not in current_codes]
        sells = [item for item in holdings if item["code"] not in target_codes]
        constituents_changed = set(current_codes) != set(target_codes)
        trade_required = scheduled or constituents_changed
        if scheduled and holdings and not constituents_changed:
            reason += "，成分不变，仅恢复等权"
        if trade_required:
            pending = {
                "signal_date": signal_text,
                "execution_date": "下一交易日开盘",
                "status": status,
                "action_label": action_label,
                "reason": reason,
                "buys": buys,
                "sells": sells,
                "target_holdings": self._equal_weight(target_holdings),
            }

        decision = {
            "strategy_id": self.strategy_id,
            "strategy": self.strategy_label,
            "policy": "每日自适应" if self.adaptive_enabled else "固定周期调仓",
            "signal_date": signal_text,
            "status": status,
            "action_label": action_label,
            "trade_required": trade_required,
            "execution_date": "下一交易日开盘" if trade_required else "无交易",
            "reason": reason,
            "cycle_day": cycle_day,
            "next_scheduled_in": self.rebalance_days if scheduled else max(0, self.rebalance_days - cycle_day),
            "adaptive_used_in_cycle": adaptive_used,
            "current_holdings": self._equal_weight(holdings),
            "target_holdings": self._equal_weight(target_holdings),
            "buys": self._equal_weight(buys),
            "sells": self._equal_weight(sells),
            "pending_order": deepcopy(pending),
            "executed_order": executed_order,
            "rules": {
                "top_n": self.top_n,
                "rebalance_days": self.rebalance_days,
                "min_hold_days": self.min_hold_days,
                "rank_buffer": self.rank_buffer,
                "score_gap": self.score_gap,
                "entry_rank": self.entry_rank,
                "max_adaptive_per_cycle": 1,
                "adaptive_enabled": self.adaptive_enabled,
            },
        }
        new_state = {
            "version": 1,
            "strategy_id": self.strategy_id,
            "last_signal_date": signal_text,
            "signal_sequence": current_sequence,
            "cycle_day": cycle_day,
            "adaptive_used_in_cycle": adaptive_used,
            "holdings": self._equal_weight(holdings),
            "pending_order": deepcopy(pending),
            "last_decision": deepcopy(decision),
        }
        self.storage.save_strategy_snapshot(self.strategy_id, signal_text, new_state, decision)
        return decision

    def _adaptive_target(
        self,
        rankings: List[Dict],
        holdings: List[Dict],
        rank_by_code: Dict[str, int],
        score_by_code: Dict[str, float],
        current_sequence: int,
        allow_regular: bool,
    ):
        selected = [item["code"] for item in holdings]
        challengers = [item["code"] for item in rankings[: self.entry_rank] if item["code"] not in selected]
        if not challengers:
            return selected, "hold", "当前 Top3 均已持有"
        entry_code = challengers[0]
        forced = [code for code in selected if code not in rank_by_code]
        if forced:
            exit_code = forced[0]
            selected[selected.index(exit_code)] = entry_code
            selected.sort(key=lambda code: rank_by_code.get(code, 10**6))
            return selected, "risk", f"风险退出 {exit_code}，换入 {entry_code}"
        if not allow_regular:
            return selected, "hold", "本周期已使用一次提前换股额度"
        eligible = [
            item for item in holdings
            if current_sequence - int(item.get("entry_sequence", current_sequence)) >= self.min_hold_days
        ]
        if not eligible:
            return selected, "hold", f"持仓尚未达到最少 {self.min_hold_days} 个交易日"
        exit_item = max(eligible, key=lambda item: rank_by_code.get(item["code"], 10**6))
        exit_code = exit_item["code"]
        rank_trigger = rank_by_code.get(exit_code, 10**6) > self.rank_buffer
        advantage = score_by_code[entry_code] - score_by_code.get(exit_code, 0.0)
        if not rank_trigger and advantage < self.score_gap:
            return selected, "hold", f"最佳挑战者分差 {advantage:.1f}，未达到 {self.score_gap:.0f} 分"
        selected[selected.index(exit_code)] = entry_code
        selected.sort(key=lambda code: rank_by_code.get(code, 10**6))
        reason = (
            f"{exit_code} 跌出排名缓冲第 {self.rank_buffer} 名，换入 {entry_code}"
            if rank_trigger
            else f"{entry_code} 领先 {exit_code} {advantage:.1f} 分"
        )
        return selected, "adaptive", reason

    def _materialize_holdings(self, targets, previous, execution_sequence, execution_date):
        previous_by_code = {item["code"]: item for item in previous}
        output = []
        for target in targets:
            item = deepcopy(target)
            old = previous_by_code.get(item["code"])
            item["entry_sequence"] = int(old.get("entry_sequence")) if old else execution_sequence
            item["entry_date"] = old.get("entry_date") if old else execution_date
            output.append(item)
        return self._equal_weight(output)

    @staticmethod
    def _refresh_holding_marks(holdings, ranking_by_code, current_sequence):
        output = []
        for holding in holdings:
            item = deepcopy(holding)
            ranked = ranking_by_code.get(item["code"])
            if ranked:
                item.update({key: value for key, value in ranked.items() if key != "target_weight"})
            item["holding_days"] = max(0, current_sequence - int(item.get("entry_sequence", current_sequence)))
            output.append(item)
        return output

    @staticmethod
    def _equal_weight(items):
        output = deepcopy(list(items))
        weight = 1 / len(output) if output else 0.0
        for item in output:
            item["target_weight"] = round(weight, 4)
        return output

    @staticmethod
    def _pick_payload(item: StrategyPick) -> Dict:
        return {
            "rank": item.rank,
            "code": item.code,
            "name": item.name,
            "score": item.score,
            "price": item.price,
            "target_weight": item.target_weight,
            "factor_scores": deepcopy(item.factor_scores),
            "reason": item.reason,
        }

    @staticmethod
    def _elapsed_sessions(previous_signal, current_signal, calendar):
        if not previous_signal:
            return 1
        sessions = [item for item in calendar if previous_signal < item <= current_signal]
        return max(1, len(sessions))

    @staticmethod
    def _next_session(after_date, calendar):
        return next((item for item in calendar if item > after_date), None)

    def _initial_state(self):
        return {
            "version": 1,
            "strategy_id": self.strategy_id,
            "last_signal_date": None,
            "signal_sequence": -1,
            "cycle_day": 0,
            "adaptive_used_in_cycle": 0,
            "holdings": [],
            "pending_order": None,
            "last_decision": None,
        }
