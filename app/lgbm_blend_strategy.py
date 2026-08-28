from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestEngine
from app.barra_residual_factors import add_barra_residual_factors
from app.lgbm_features import build_alpha158_lite
from app.lgbm_strategy import (
    BASELINE_WEIGHTS,
    _feature_label,
    feature_schema_hash,
    file_sha256,
)


MODEL_VERSION = "lgbm_new_factors_blend_v1_candidate"
METADATA_FILENAME = "metadata.json"


class LgbmBlendStrategyModel:
    """Frozen Alpha158/Barra blend used only for forward paper trading."""

    def __init__(self, model_dir: Path):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("LGBM 融合模拟盘需要安装 lightgbm==4.6.0") from exc

        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise RuntimeError(f"LGBM 融合模型元数据不存在: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("model_version") != MODEL_VERSION:
            raise RuntimeError("LGBM 融合模型版本不受支持")
        if self.metadata.get("artifact_status") != "frozen_for_forward_shadow_only":
            raise RuntimeError("LGBM 融合模型不是冻结的前向影子工件")

        baseline_audit = dict(self.metadata.get("baseline_model_audit") or {})
        candidate_audit = dict(self.metadata.get("candidate_model_audit") or {})
        self.baseline_features = list(baseline_audit.get("feature_columns") or [])
        self.candidate_features = list(candidate_audit.get("feature_columns") or [])
        self._validate_feature_audit(self.baseline_features, baseline_audit, "Alpha158")
        self._validate_feature_audit(self.candidate_features, candidate_audit, "Alpha158+Barra")
        if self.candidate_features[: len(self.baseline_features)] != self.baseline_features:
            raise RuntimeError("融合模型的基础特征前缀与 Alpha158 模型不一致")

        training_as_of = pd.Timestamp(self.metadata["training_as_of_exclusive"])
        max_label_ends = [
            pd.Timestamp(baseline_audit["max_training_label_end_date"]),
            pd.Timestamp(candidate_audit["max_training_label_end_date"]),
        ]
        if not all(value < training_as_of for value in max_label_ends):
            raise RuntimeError("融合模型包含训练截止日尚未成熟的标签")
        if not baseline_audit.get("strictly_mature") or not candidate_audit.get("strictly_mature"):
            raise RuntimeError("融合模型成熟标签审计未通过")

        self.baseline_model = self._load_model(
            lgb,
            "baseline_model_file",
            "baseline_model_sha256",
            len(self.baseline_features),
        )
        self.candidate_model = self._load_model(
            lgb,
            "candidate_model_file",
            "candidate_model_sha256",
            len(self.candidate_features),
        )
        self.baseline_model_weight = float(self.metadata["alpha158_model_weight"])
        self.candidate_model_weight = float(self.metadata["alpha158_barra_model_weight"])
        self.rule_weight = float(self.metadata["rule_weight"])
        if not np.isclose(self.baseline_model_weight + self.candidate_model_weight, 1.0):
            raise RuntimeError("两个 LGBM 模型的融合权重之和必须为1")
        if not 0.0 <= self.rule_weight <= 1.0:
            raise RuntimeError("规则模型权重必须位于 [0, 1]")

        component_hashes = (
            str(self.metadata["baseline_model_sha256"])
            + str(self.metadata["candidate_model_sha256"])
        )
        self.artifact_sha256 = hashlib.sha256(component_hashes.encode("ascii")).hexdigest()

    @staticmethod
    def _validate_feature_audit(features: list[str], audit: Dict[str, Any], label: str) -> None:
        if not features or len(features) != int(audit.get("feature_count", -1)):
            raise RuntimeError(f"{label} 特征数量与冻结元数据不一致")
        if feature_schema_hash(features) != audit.get("feature_schema_sha256"):
            raise RuntimeError(f"{label} 特征模式哈希不一致")

    def _load_model(self, lgb, file_key: str, hash_key: str, feature_count: int):
        model_path = self.model_dir / str(self.metadata[file_key])
        if not model_path.exists() or file_sha256(model_path) != self.metadata[hash_key]:
            raise RuntimeError(f"融合模型文件缺失或SHA-256校验失败: {model_path.name}")
        model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        if model.num_feature() != feature_count:
            raise RuntimeError(f"融合模型特征数量与元数据不一致: {model_path.name}")
        return model

    def rank(self, history: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, Dict[str, Any]]:
        frame, base_features = build_alpha158_lite(history)
        frame, barra_features = add_barra_residual_factors(frame)
        candidate_features = base_features + barra_features
        if base_features != self.baseline_features:
            raise RuntimeError("当前 Alpha158 特征顺序与冻结模型不一致")
        if candidate_features != self.candidate_features:
            raise RuntimeError("当前 Alpha158+Barra 特征顺序与冻结模型不一致")

        available_dates = pd.DatetimeIndex(
            frame.loc[frame["date"].le(pd.Timestamp(as_of)), "date"].dropna().unique()
        )
        if available_dates.empty:
            raise ValueError("没有可用的 LGBM 融合信号日期")
        signal_date = available_dates.max()
        cross = frame[frame["date"].eq(signal_date)].copy()
        cross = cross[
            cross["tradestatus"].eq(1)
            & cross["is_st"].eq(0)
            & cross["close"].gt(1)
            & cross["amount_20"].ge(10_000_000)
        ].dropna(subset=[
            "ret_5", "ret_20", "ret_60", "volatility_20", "amount_20",
            "amount_60", "absolute_return_60", "ma_20",
        ])
        if "universe_member" in cross.columns:
            cross = cross[cross["universe_member"].fillna(False).astype(bool)]
        if len(cross) < 5:
            raise ValueError("LGBM 融合信号日可交易股票不足5只")

        baseline_matrix = cross[self.baseline_features].fillna(0.0).to_numpy(dtype=np.float32)
        candidate_matrix = cross[self.candidate_features].fillna(0.0).to_numpy(dtype=np.float32)
        baseline_prediction = pd.Series(
            self.baseline_model.predict(baseline_matrix), index=cross.index
        )
        candidate_prediction = pd.Series(
            self.candidate_model.predict(candidate_matrix), index=cross.index
        )
        cross["alpha158_lgbm_rank"] = baseline_prediction.rank(pct=True)
        cross["barra_lgbm_rank"] = candidate_prediction.rank(pct=True)
        blended = (
            cross["alpha158_lgbm_rank"].mul(self.baseline_model_weight)
            + cross["barra_lgbm_rank"].mul(self.candidate_model_weight)
        )
        cross["lgbm_rank"] = blended.rank(pct=True)

        baseline = BacktestEngine()._score(cross.copy(), BASELINE_WEIGHTS)
        cross["baseline_rank"] = baseline["score"].reindex(cross.index).rank(pct=True)
        cross["score"] = (
            cross["lgbm_rank"].mul(1.0 - self.rule_weight)
            + cross["baseline_rank"].mul(self.rule_weight)
        ).mul(100.0)

        baseline_contributions = self.baseline_model.predict(
            baseline_matrix, pred_contrib=True
        )[:, :-1]
        candidate_contributions = self.candidate_model.predict(
            candidate_matrix, pred_contrib=True
        )[:, :-1]
        cross["lgbm_reason"] = [
            self._reason(base_values, candidate_values)
            for base_values, candidate_values in zip(
                baseline_contributions, candidate_contributions
            )
        ]
        ranked = cross.sort_values(
            ["score", "amount_20"], ascending=[False, False]
        ).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        trained_through = max(
            self.metadata["baseline_model_audit"]["max_training_label_end_date"],
            self.metadata["candidate_model_audit"]["max_training_label_end_date"],
        )
        return ranked, {
            "strategy": "75/25 Alpha158-Barra LGBM 融合模拟盘",
            "strategy_id": MODEL_VERSION,
            "model_version": MODEL_VERSION,
            "model_created_at": self.metadata["created_at"],
            "model_sha256": self.artifact_sha256,
            "component_sha256s": {
                "alpha158": self.metadata["baseline_model_sha256"],
                "alpha158_barra": self.metadata["candidate_model_sha256"],
            },
            "trained_through": trained_through,
            "training_as_of_exclusive": self.metadata["training_as_of_exclusive"],
            "signal_date": signal_date.date().isoformat(),
            "feature_count": len(self.candidate_features),
            "rule_weight": self.rule_weight,
            "alpha158_model_weight": self.baseline_model_weight,
            "alpha158_barra_model_weight": self.candidate_model_weight,
            "strictly_mature": True,
            "strict_no_lookahead_certified": False,
            "execution": "收盘生成排名；触发模拟订单后于下一交易日开盘执行",
        }

    def _reason(self, baseline_values: np.ndarray, candidate_values: np.ndarray) -> str:
        base_index = int(np.argmax(np.abs(baseline_values)))
        candidate_index = int(np.argmax(np.abs(candidate_values)))
        return (
            f"Alpha158:{_feature_label(self.baseline_features[base_index])}、"
            f"融合:{_feature_label(self.candidate_features[candidate_index])}"
        )
