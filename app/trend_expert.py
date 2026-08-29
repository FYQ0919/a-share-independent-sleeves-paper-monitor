from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestEngine
from app.models import StrategyPick
from app.storage import Storage


EQUIPMENT_CODES = frozenset({"688037", "688120", "688200", "688630"})


class TrendExpertRanker:
    """Frozen semiconductor-equipment trend overlay for forward paper use."""

    STRATEGY_ID = "semiconductor_trend_expert_v1"

    def rank(
        self,
        base_rankings: Sequence[StrategyPick],
        history: pd.DataFrame,
        signal_date: date,
    ) -> tuple[List[StrategyPick], Dict]:
        if len(base_rankings) < 5:
            raise ValueError("趋势专家需要至少5只有效的基础LGBM排名")

        features = BacktestEngine()._features(history)
        cross = features[
            features["date"].eq(pd.Timestamp(signal_date))
        ][["code", "ret_20", "ret_60", "amount_20", "amount_60"]].copy()
        cross["code"] = cross["code"].astype(str)
        cross = cross.drop_duplicates("code", keep="last").set_index("code")

        rows = []
        for item in base_rankings:
            if item.code not in cross.index:
                continue
            factor_scores = item.factor_scores or {}
            rows.append({
                "code": item.code,
                "name": item.name,
                "price": item.price,
                "lgbm_rank": float(factor_scores.get("LGBM排名", 0.0)) / 100.0,
                "baseline_rank": float(factor_scores.get("基础策略排名", 0.0)) / 100.0,
                **cross.loc[item.code].to_dict(),
            })
        frame = pd.DataFrame(rows).dropna(
            subset=["ret_20", "ret_60", "amount_20", "amount_60"]
        )
        if len(frame) < 5:
            raise ValueError("趋势专家信号日的有效因子股票不足5只")

        frame["amount_expansion"] = (
            frame["amount_20"].div(frame["amount_60"].replace(0.0, np.nan)).sub(1.0)
        )
        frame["ret20_rank"] = frame["ret_20"].rank(pct=True)
        frame["ret60_rank"] = frame["ret_60"].rank(pct=True)
        frame["amount_rank"] = frame["amount_expansion"].rank(pct=True)
        frame["individual_trend"] = (
            frame["ret20_rank"].mul(0.45)
            + frame["ret60_rank"].mul(0.35)
            + frame["amount_rank"].mul(0.20)
        )
        frame["equipment_member"] = frame["code"].isin(EQUIPMENT_CODES)

        equipment = frame[frame["equipment_member"]]
        strong_equipment = bool(
            len(equipment) >= 3
            and equipment["ret_20"].gt(0.0).mean() >= 0.75
            and equipment["ret_20"].median() > frame["ret_20"].median()
            and equipment["ret_60"].median() > frame["ret_60"].median()
        )

        frame["trend_expert"] = frame["individual_trend"].mul(0.75)
        if strong_equipment:
            frame.loc[frame["equipment_member"], "trend_expert"] += 0.25
            frame["score"] = (
                frame["lgbm_rank"].mul(0.60)
                + frame["baseline_rank"].mul(0.20)
                + frame["trend_expert"].mul(0.20)
            ).mul(100.0)
        else:
            frame["score"] = (
                frame["lgbm_rank"].mul(0.60)
                + frame["baseline_rank"].mul(0.40)
            ).mul(100.0)

        ranked = frame.sort_values(
            ["score", "amount_20"], ascending=[False, False], kind="mergesort"
        ).reset_index(drop=True)
        target_weight = 1.0 / min(5, len(ranked))
        picks = []
        for row_number, row in ranked.iterrows():
            trend_score = float(row["trend_expert"] * 100.0)
            reason = "趋势增强"
            if strong_equipment and bool(row["equipment_member"]):
                reason = "半导体设备强势 + 趋势增强"
            elif not strong_equipment:
                reason = "普通环境：LGBM + 防御规则"
            picks.append(StrategyPick(
                rank=row_number + 1,
                code=str(row["code"]),
                name=str(row["name"]),
                score=round(float(row["score"]), 2),
                price=round(float(row["price"]), 2),
                target_weight=round(target_weight, 4),
                factor_scores={
                    "LGBM排名": round(float(row["lgbm_rank"] * 100.0), 2),
                    "基础策略排名": round(float(row["baseline_rank"] * 100.0), 2),
                    "趋势专家": round(trend_score, 2),
                },
                reason=reason,
            ))

        metadata = {
            "strategy_id": self.STRATEGY_ID,
            "strategy": "半导体设备趋势专家",
            "signal_date": signal_date.isoformat(),
            "strong_equipment": strong_equipment,
            "equipment_count": int(len(equipment)),
            "equipment_breadth": (
                float(equipment["ret_20"].gt(0.0).mean()) if len(equipment) else 0.0
            ),
            "maximum_replacements": 2 if strong_equipment else 1,
            "score_contract": (
                "60%融合LGBM + 20%防御规则 + 20%趋势专家"
                if strong_equipment
                else "60%融合LGBM + 40%防御规则"
            ),
            "equipment_codes": sorted(EQUIPMENT_CODES),
            "execution": "收盘生成信号；每10个交易日调仓；下一交易日开盘执行",
        }
        return picks, metadata


class TrendPortfolioService:
    """Ten-session paper policy with one or two replacements per rebalance."""

    STRATEGY_ID = "semiconductor_trend_expert_top5_fixed10_v1"

    def __init__(
        self,
        storage: Storage,
        top_n: int = 5,
        rebalance_days: int = 10,
        strategy_id: str = STRATEGY_ID,
    ):
        self.storage = storage
        self.top_n = int(top_n)
        self.rebalance_days = int(rebalance_days)
        self.strategy_id = str(strategy_id)

    def tracked_symbols(self) -> Dict[str, str]:
        state = self.storage.load_strategy_state(self.strategy_id) or {}
        rows = list(state.get("holdings", []))
        rows.extend((state.get("pending_order") or {}).get("target_holdings", []))
        return {
            str(item["code"]): str(item.get("name") or item["code"])
            for item in rows
        }

    def evaluate(
        self,
        rankings: Sequence[StrategyPick],
        signal_date: date,
        trading_dates: Sequence[date],
        *,
        strong_equipment: bool,
    ) -> Dict:
        if len(rankings) < self.top_n:
            raise ValueError("趋势专家排名不足以构建Top5")
        signal_text = signal_date.isoformat()
        state = self.storage.load_strategy_state(self.strategy_id) or self._initial_state()
        if state.get("last_signal_date") == signal_text and state.get("last_decision"):
            return deepcopy(state["last_decision"])
        if state.get("last_signal_date") and signal_text < state["last_signal_date"]:
            raise ValueError("趋势袖套信号日期早于持仓账本日期")

        calendar = sorted({item.isoformat() for item in trading_dates if item <= signal_date})
        previous_signal = state.get("last_signal_date")
        elapsed = self._elapsed_sessions(previous_signal, signal_text, calendar)
        holdings = deepcopy(state.get("holdings", []))
        pending = deepcopy(state.get("pending_order"))
        executed_order = None
        if pending and pending.get("signal_date", "") < signal_text:
            execution_date = self._next_session(pending["signal_date"], calendar) or signal_text
            holdings = deepcopy(pending.get("target_holdings", []))
            for item in holdings:
                item.setdefault("entry_date", execution_date)
            executed_order = {
                "signal_date": pending["signal_date"],
                "execution_date": execution_date,
                "action_label": pending["action_label"],
                "reason": pending["reason"],
                "buys": pending.get("buys", []),
                "sells": pending.get("sells", []),
            }
            pending = None

        ranked_rows = [self._pick_payload(item) for item in rankings]
        rank_by_code = {item["code"]: item for item in ranked_rows}
        refreshed = []
        for item in holdings:
            current = deepcopy(item)
            if current["code"] in rank_by_code:
                entry_date = current.get("entry_date")
                current.update(rank_by_code[current["code"]])
                current["entry_date"] = entry_date
            refreshed.append(current)
        holdings = self._equal_weight(refreshed)

        cycle_day = int(state.get("cycle_day", 0)) + elapsed if previous_signal else 0
        scheduled = not holdings or cycle_day >= self.rebalance_days
        target = deepcopy(holdings)
        action_label = "继续持有"
        reason = f"固定{self.rebalance_days}个交易日调仓，周期内不换股"
        replacements = 0
        if scheduled:
            maximum = 2 if strong_equipment else 1
            if not holdings:
                target = deepcopy(ranked_rows[: self.top_n])
                reason = "趋势袖套首次建仓"
            else:
                target, replacements = self._replacement_target(
                    holdings, ranked_rows, maximum
                )
                regime = "半导体设备强势" if strong_equipment else "普通环境"
                reason = f"第{self.rebalance_days}个交易日主调仓；{regime}，最多替换{maximum}只"
            cycle_day = 0
            action_label = "主调仓"

        current_codes = {item["code"] for item in holdings}
        target_codes = {item["code"] for item in target}
        buys = [item for item in target if item["code"] not in current_codes]
        sells = [item for item in holdings if item["code"] not in target_codes]
        trade_required = scheduled
        if trade_required:
            pending = {
                "signal_date": signal_text,
                "execution_date": "下一交易日开盘",
                "action_label": action_label,
                "reason": reason,
                "buys": self._equal_weight(buys),
                "sells": self._equal_weight(sells),
                "target_holdings": self._equal_weight(target),
            }

        decision = {
            "strategy_id": self.strategy_id,
            "strategy": "半导体设备趋势专家 Top5",
            "policy": "固定10日周期 + 环境替换上限",
            "signal_date": signal_text,
            "status": "scheduled" if scheduled else "hold",
            "action_label": action_label,
            "trade_required": trade_required,
            "execution_date": "下一交易日开盘" if trade_required else "无交易",
            "reason": reason,
            "cycle_day": cycle_day,
            "next_scheduled_in": self.rebalance_days if scheduled else max(
                0, self.rebalance_days - cycle_day
            ),
            "strong_equipment": bool(strong_equipment),
            "maximum_replacements": 2 if strong_equipment else 1,
            "replacements": replacements,
            "current_holdings": self._equal_weight(holdings),
            "target_holdings": self._equal_weight(target),
            "buys": self._equal_weight(buys),
            "sells": self._equal_weight(sells),
            "pending_order": deepcopy(pending),
            "executed_order": executed_order,
            "rules": {
                "top_n": self.top_n,
                "rebalance_days": self.rebalance_days,
                "normal_max_replacements": 1,
                "strong_max_replacements": 2,
            },
        }
        new_state = {
            "version": 1,
            "strategy_id": self.strategy_id,
            "last_signal_date": signal_text,
            "cycle_day": cycle_day,
            "holdings": self._equal_weight(holdings),
            "pending_order": deepcopy(pending),
            "last_decision": deepcopy(decision),
        }
        self.storage.save_strategy_snapshot(
            self.strategy_id, signal_text, new_state, decision
        )
        return decision

    def _replacement_target(self, holdings, rankings, maximum):
        ranking_by_code = {item["code"]: item for item in rankings}
        available = [item for item in holdings if item["code"] in ranking_by_code]
        missing = [item for item in holdings if item["code"] not in ranking_by_code]
        ranked_holdings = sorted(
            available,
            key=lambda item: ranking_by_code[item["code"]]["score"],
        )
        challengers = [
            item for item in rankings if item["code"] not in {row["code"] for row in holdings}
        ]
        count = min(maximum, len(ranked_holdings) + len(missing), len(challengers))
        dropped_codes = [item["code"] for item in missing[:count]]
        dropped_codes.extend(
            item["code"] for item in ranked_holdings[: max(0, count - len(dropped_codes))]
        )
        target = [item for item in holdings if item["code"] not in dropped_codes]
        target.extend(challengers[:count])
        if len(target) < self.top_n:
            target.extend(
                item for item in rankings if item["code"] not in {row["code"] for row in target}
            )
        target = sorted(
            target[: self.top_n],
            key=lambda item: ranking_by_code.get(item["code"], item)["score"],
            reverse=True,
        )
        return self._equal_weight(target), count

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
    def _equal_weight(rows):
        output = deepcopy(list(rows))
        weight = 1.0 / len(output) if output else 0.0
        for item in output:
            item["target_weight"] = round(weight, 4)
        return output

    @staticmethod
    def _elapsed_sessions(previous, current, calendar):
        if not previous:
            return 0
        return sum(previous < item <= current for item in calendar)

    @staticmethod
    def _next_session(signal_date, calendar):
        return next((item for item in calendar if item > signal_date), None)

    def _initial_state(self):
        return {
            "version": 1,
            "strategy_id": self.strategy_id,
            "last_signal_date": None,
            "cycle_day": 0,
            "holdings": [],
            "pending_order": None,
        }
