from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from app.backtest_data import BaoStockHistoryProvider
from app.backtest_engine import BacktestEngine
from app.models import Candidate, StrategyPick
from app.adaptive_portfolio import AdaptivePortfolioService
from app.lgbm_strategy import load_frozen_strategy_model
from app.paper_account import LgbmPaperAccountService


class StrategySignalService:
    def __init__(
        self,
        cache_dir,
        top_n: int = 5,
        rebalance_days: int = 10,
        portfolio: AdaptivePortfolioService = None,
        strategy_backend: str = "contrarian",
        lgbm_model_dir=None,
        lgbm_history_days: int = 550,
        paper_account: LgbmPaperAccountService = None,
    ):
        self.provider = BaoStockHistoryProvider(cache_dir)
        self.engine = BacktestEngine()
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.portfolio = portfolio
        self.strategy_backend = strategy_backend.lower()
        if self.strategy_backend not in {"contrarian", "lgbm_shadow", "lgbm_active"}:
            raise ValueError(
                "STRATEGY_MODEL_BACKEND 仅支持 contrarian、lgbm_shadow 或 lgbm_active"
            )
        self.lgbm_history_days = max(240, int(lgbm_history_days))
        self.lgbm_model_dir = lgbm_model_dir
        self.lgbm_model = None
        self.paper_account = paper_account

    def build(self, candidates: Sequence[Candidate], as_of: date) -> Tuple[List[StrategyPick], Dict, List[str]]:
        tracked = self.portfolio.tracked_symbols() if self.portfolio else {}
        if self.paper_account:
            tracked.update(self.paper_account.tracked_symbols())
        codes = list(dict.fromkeys([item.code for item in candidates] + list(tracked)))
        if len(codes) < 2:
            return [], {}, ["候选股票不足，未生成逆向增强 Top5"]
        names = {item.code: item.name for item in candidates}
        names.update({code: name for code, name in tracked.items() if code not in names})
        history_days = self.lgbm_history_days if self.strategy_backend.startswith("lgbm_") else 180
        history, warnings = self.provider.load(codes, names, as_of - timedelta(days=history_days), as_of)
        rankings, metadata, trading_dates = self.rank_from_history(history, as_of)
        picks = rankings[: self.top_n]
        if self.portfolio and self.strategy_backend != "lgbm_shadow":
            metadata["decision"] = self.portfolio.evaluate(
                rankings, date.fromisoformat(metadata["signal_date"]), trading_dates
            )
            if self.strategy_backend == "lgbm_active" and self.paper_account:
                metadata["paper_account"] = self.paper_account.update(
                    metadata["decision"], history
                )
        elif self.strategy_backend == "lgbm_shadow":
            metadata["model_status"] = "shadow"
            metadata["decision"] = {
                "strategy_id": metadata["strategy_id"],
                "strategy": metadata["strategy"],
                "policy": "影子模式",
                "signal_date": metadata["signal_date"],
                "status": "shadow",
                "action_label": "影子观察",
                "trade_required": False,
                "execution_date": "无交易",
                "reason": "LGBM仅记录排名，不写入持仓账本或生成订单",
                "cycle_day": 0,
                "next_scheduled_in": self.rebalance_days,
                "adaptive_used_in_cycle": 0,
                "current_holdings": [],
                "target_holdings": [],
                "buys": [],
                "sells": [],
                "rules": {"rebalance_days": self.rebalance_days},
            }
        return picks, metadata, warnings

    def select_from_history(self, history: pd.DataFrame, as_of: date):
        rankings, metadata, _ = self.rank_from_history(history, as_of)
        return rankings[: self.top_n], metadata

    def rank_from_history(self, history: pd.DataFrame, as_of: date):
        if self.strategy_backend.startswith("lgbm_"):
            return self._rank_lgbm(history, as_of)
        frame = self.engine._features(history)
        available_dates = pd.DatetimeIndex(frame.loc[frame["date"] <= pd.Timestamp(as_of), "date"].dropna().unique())
        if available_dates.empty:
            raise ValueError("没有可用的策略信号日期")
        signal_date = available_dates.max()
        cross = frame[frame["date"] == signal_date].copy()
        required = [
            "ret_5", "ret_60", "amount_20", "amount_60", "absolute_return_60",
            "volatility_20", "drawdown_60", "pe", "pb",
        ]
        cross = cross[
            (cross["tradestatus"] == 1)
            & (cross["is_st"] == 0)
            & (cross["close"] > 1)
            & (cross["amount_20"] >= 10_000_000)
        ].dropna(subset=required)
        if cross.empty:
            raise ValueError("当前候选没有满足历史因子条件的股票")
        strategy = self.engine.STRATEGIES["contrarian"]
        ranked = self.engine._score(cross, strategy["weights"]).sort_values(
            ["score", "amount_20"], ascending=[False, False]
        )
        target_weight = 1 / min(self.top_n, len(ranked))
        factor_labels = {
            "medium_reversal": "中期反转",
            "inefficiency": "趋势低效",
            "liquidity_cooling": "流动性降温",
            "low_volatility": "低波动",
            "risk": "风险质量",
            "value": "估值",
        }
        rankings = []
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            scores = {
                factor_labels[name]: round(float(row[f"{name}_score"]), 4)
                for name in strategy["weights"]
            }
            strongest = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:2]
            rankings.append(StrategyPick(
                rank=rank,
                code=str(row["code"]),
                name=str(row["name"]),
                score=round(float(row["score"]), 2),
                price=round(float(row["close"]), 2),
                target_weight=round(target_weight, 4),
                factor_scores=scores,
                reason="、".join(name for name, _ in strongest),
            ))
        metadata = {
            "strategy": strategy["label"],
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "top_n": min(self.top_n, len(rankings)),
            "rebalance_days": self.rebalance_days,
            "execution": "每日收盘评估；主调仓或强信号提前换股于下一交易日开盘执行",
        }
        trading_dates = [item.date() for item in sorted(available_dates)]
        return rankings, metadata, trading_dates

    def _rank_lgbm(self, history: pd.DataFrame, as_of: date):
        model = self._get_lgbm_model()
        ranked, model_metadata = model.rank(history, as_of)
        target_weight = 1 / min(self.top_n, len(ranked))
        rankings = []
        for _, row in ranked.iterrows():
            factor_scores = {
                "LGBM排名": round(float(row["lgbm_rank"]) * 100, 2),
                "基础策略排名": round(float(row["baseline_rank"]) * 100, 2),
            }
            if "alpha158_lgbm_rank" in row:
                factor_scores["Alpha158模型排名"] = round(
                    float(row["alpha158_lgbm_rank"]) * 100, 2
                )
            if "barra_lgbm_rank" in row:
                factor_scores["Alpha158+Barra模型排名"] = round(
                    float(row["barra_lgbm_rank"]) * 100, 2
                )
            rankings.append(StrategyPick(
                rank=int(row["rank"]),
                code=str(row["code"]),
                name=self._display_name(row.get("name"), row["code"]),
                score=round(float(row["score"]), 2),
                price=round(float(row["close"]), 2),
                target_weight=round(target_weight, 4),
                factor_scores=factor_scores,
                reason=f"LGBM贡献：{row['lgbm_reason']}",
            ))
        metadata = {
            **model_metadata,
            "top_n": min(self.top_n, len(rankings)),
            "rebalance_days": self.rebalance_days,
        }
        available_dates = pd.DatetimeIndex(
            history.loc[history["date"] <= pd.Timestamp(as_of), "date"].dropna().unique()
        )
        trading_dates = [item.date() for item in sorted(available_dates)]
        return rankings, metadata, trading_dates

    @staticmethod
    def _display_name(value, code) -> str:
        if value is None or pd.isna(value) or not str(value).strip():
            return str(code)
        return str(value)

    def model_status(self) -> Dict:
        if not self.strategy_backend.startswith("lgbm_"):
            return {"backend": "contrarian", "available": True, "mode": "rule"}
        try:
            model = self._get_lgbm_model()
            return {
                "backend": "lightgbm",
                "available": True,
                "mode": "shadow" if self.strategy_backend == "lgbm_shadow" else "active",
                "model_version": model.metadata["model_version"],
                "model_sha256": getattr(
                    model, "artifact_sha256", model.metadata.get("model_sha256")
                ),
                "trained_through": self._trained_through(model.metadata),
            }
        except Exception as exc:
            return {
                "backend": "lightgbm",
                "available": False,
                "mode": "shadow" if self.strategy_backend == "lgbm_shadow" else "active",
                "error": str(exc),
            }

    @staticmethod
    def _trained_through(metadata: Dict) -> str | None:
        direct = metadata.get("max_training_label_end_date")
        if direct:
            return str(direct)
        audits = [
            metadata.get("baseline_model_audit") or {},
            metadata.get("candidate_model_audit") or {},
        ]
        values = [item.get("max_training_label_end_date") for item in audits]
        return max((str(value) for value in values if value), default=None)

    def _get_lgbm_model(self):
        if self.lgbm_model is None:
            self.lgbm_model = load_frozen_strategy_model(self.lgbm_model_dir)
        return self.lgbm_model
