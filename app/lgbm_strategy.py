from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestEngine
from app.lgbm_features import build_alpha158_lite, mature_training_mask


MODEL_VERSION = "lgbm_ranker_top5_v1"
MODEL_FILENAME = "model.txt"
METADATA_FILENAME = "metadata.json"
RELEVANCE_LEVELS = 5
RANDOM_SEED = 20260827
WINNER_CONFIG = {
    "baseline_weight": 0.40,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 300,
    "n_estimators": 120,
    "feature_fraction": 0.70,
}
BASELINE_WEIGHTS = {
    "medium_reversal": 0.33,
    "inefficiency": 0.23,
    "liquidity_cooling": 0.19,
    "low_volatility": 0.05,
    "risk": 0.10,
    "value": 0.10,
}


def feature_schema_hash(feature_columns: list[str]) -> str:
    canonical = json.dumps(feature_columns, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_training_mask(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    cutoff = pd.Timestamp(as_of)
    mask = mature_training_mask(frame, cutoff)
    mask &= pd.to_datetime(frame["label_end_date"], errors="coerce").lt(cutoff)
    mask &= pd.to_datetime(frame["date"], errors="coerce").lt(cutoff)
    mask &= frame["tradestatus"].eq(1)
    mask &= frame["is_st"].eq(0)
    mask &= frame["close"].gt(1)
    mask &= frame["amount_20"].ge(10_000_000)
    if "universe_member" in frame.columns:
        mask &= frame["universe_member"].fillna(False).astype(bool)
    return mask


def build_ranker_training_set(
    frame: pd.DataFrame,
    feature_columns: list[str],
    as_of: pd.Timestamp,
    relevance_levels: int = RELEVANCE_LEVELS,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    if relevance_levels < 2:
        raise ValueError("relevance_levels 至少为2")
    train = frame.loc[eligible_training_mask(frame, as_of)].copy()
    counts = train.groupby("date")["code"].transform("size")
    train = train.loc[counts.ge(5)].sort_values(["date", "code"], kind="mergesort").copy()
    if train.empty:
        return train, np.array([], dtype=np.int32), []

    ranks = train["label_return"].groupby(train["date"]).rank(method="average").sub(1)
    denominators = train.groupby("date")["label_return"].transform("size").sub(1)
    percentiles = ranks.div(denominators)
    relevance = np.minimum(
        np.floor(percentiles.to_numpy(dtype=float) * relevance_levels),
        relevance_levels - 1,
    ).astype(np.int32)
    groups = train.groupby("date", sort=True).size().astype(int).tolist()
    if sum(groups) != len(train):
        raise RuntimeError("LightGBM 分组大小与训练行数不一致")
    if train[feature_columns].shape[1] != len(feature_columns):
        raise RuntimeError("LightGBM 特征列不一致")
    return train, relevance, groups


def native_lgbm_params(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.03,
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_data_in_leaf": int(config["min_child_samples"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "seed": RANDOM_SEED,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def train_frozen_model(
    history: pd.DataFrame,
    model_dir: Path,
    as_of: date,
    metadata_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("训练 LGBM 策略需要安装 lightgbm") from exc

    frame, feature_columns = build_alpha158_lite(history)
    cutoff = pd.Timestamp(as_of)
    train, relevance, groups = build_ranker_training_set(frame, feature_columns, cutoff)
    if train["date"].nunique() < 504 or len(train) < 10_000:
        raise RuntimeError("LGBM 冻结模型至少需要504个训练日和10000行成熟样本")

    params = native_lgbm_params(WINNER_CONFIG)
    dataset = lgb.Dataset(
        train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32),
        label=relevance,
        group=groups,
        feature_name=feature_columns,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        dataset,
        num_boost_round=WINNER_CONFIG["n_estimators"],
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    # Avoid LightGBM's native path handling, which rejects this Windows Unicode path.
    model_path.write_text(model.model_to_string(), encoding="utf-8")

    gains = model.feature_importance(importance_type="gain").astype(float)
    importance = sorted(
        (
            {"feature": feature_columns[index], "gain": float(value)}
            for index, value in enumerate(gains)
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    metadata: Dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "artifact_status": "frozen_for_forward_paper",
        "created_at": datetime.now().isoformat(),
        "backend": "lightgbm_native_lambdarank",
        "lightgbm_version": lgb.__version__,
        "model_file": MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "feature_schema": "alpha158_lite_v1",
        "feature_schema_sha256": feature_schema_hash(feature_columns),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "winner_config": WINNER_CONFIG,
        "model_params": params,
        "baseline_weights": BASELINE_WEIGHTS,
        "baseline_weight": WINNER_CONFIG["baseline_weight"],
        "training_as_of_exclusive": cutoff.date().isoformat(),
        "training_rows": len(train),
        "training_dates": int(train["date"].nunique()),
        "training_groups": len(groups),
        "training_signal_start": pd.Timestamp(train["date"].min()).date().isoformat(),
        "training_signal_end": pd.Timestamp(train["date"].max()).date().isoformat(),
        "max_training_label_end_date": pd.Timestamp(
            train["label_end_date"].max()
        ).date().isoformat(),
        "label_contract": "signal close; buy next open; exit at t+11 open; five relevance levels",
        "execution_contract": "Top5; next-open; 10-session main rebalance; 12bp one-way research cost",
        "research_gate_passed": True,
        "strict_no_lookahead_certified": False,
        "production_change": False,
        "top_feature_importance": importance[:20],
        "warnings": [
            "当前Top50回填历史存在成分与存续偏差",
            "当前版本前复权价格不是严格as-of价格",
            "该模型只允许用于冻结后的前向模拟盘",
        ],
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    metadata_path = model_dir / METADATA_FILENAME
    temporary_metadata_path = model_dir / f"{METADATA_FILENAME}.tmp"
    temporary_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_metadata_path.replace(metadata_path)
    return metadata


class LgbmStrategyModel:
    def __init__(self, model_dir: Path):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("LGBM 每日策略需要安装 lightgbm") from exc

        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise RuntimeError(f"LGBM 模型元数据不存在: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("model_version") != MODEL_VERSION:
            raise RuntimeError("LGBM 模型版本不受支持")
        self.feature_columns = list(self.metadata.get("feature_columns") or [])
        if feature_schema_hash(self.feature_columns) != self.metadata.get("feature_schema_sha256"):
            raise RuntimeError("LGBM 特征模式哈希不一致")
        model_path = self.model_dir / str(self.metadata.get("model_file", MODEL_FILENAME))
        if not model_path.exists() or file_sha256(model_path) != self.metadata.get("model_sha256"):
            raise RuntimeError("LGBM 模型文件缺失或SHA-256校验失败")
        self.model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        if self.model.num_feature() != len(self.feature_columns):
            raise RuntimeError("LGBM 模型特征数量与元数据不一致")

    def rank(self, history: pd.DataFrame, as_of: date) -> tuple[pd.DataFrame, Dict[str, Any]]:
        frame, computed_features = build_alpha158_lite(history)
        if computed_features != self.feature_columns:
            raise RuntimeError("当前特征顺序与冻结 LGBM 模型不一致")
        available_dates = pd.DatetimeIndex(
            frame.loc[frame["date"] <= pd.Timestamp(as_of), "date"].dropna().unique()
        )
        if available_dates.empty:
            raise ValueError("没有可用的 LGBM 信号日期")
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
        if len(cross) < 2:
            raise ValueError("LGBM 信号日可交易股票不足2只")

        matrix = cross[self.feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        prediction = self.model.predict(matrix)
        contributions = self.model.predict(matrix, pred_contrib=True)
        cross["lgbm_raw"] = prediction
        cross["lgbm_rank"] = pd.Series(prediction, index=cross.index).rank(pct=True)
        baseline = BacktestEngine()._score(cross.copy(), BASELINE_WEIGHTS)
        baseline_score = baseline["score"].reindex(cross.index)
        cross["baseline_rank"] = baseline_score.rank(pct=True)
        baseline_weight = float(self.metadata["baseline_weight"])
        cross["score"] = (
            cross["lgbm_rank"].mul(1 - baseline_weight)
            + cross["baseline_rank"].mul(baseline_weight)
        ).mul(100)
        cross["lgbm_reason"] = [
            self._reason(values[:-1]) for values in np.asarray(contributions)
        ]
        ranked = cross.sort_values(["score", "amount_20"], ascending=[False, False]).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        return ranked, {
            "strategy": "LGBM LambdaRank + 基础策略保护",
            "strategy_id": MODEL_VERSION,
            "model_version": self.metadata["model_version"],
            "model_created_at": self.metadata["created_at"],
            "model_sha256": self.metadata["model_sha256"],
            "trained_through": self.metadata["max_training_label_end_date"],
            "signal_date": signal_date.date().isoformat(),
            "feature_count": len(self.feature_columns),
            "baseline_weight": baseline_weight,
            "strict_no_lookahead_certified": False,
            "execution": "收盘生成排名；触发订单后于下一交易日开盘执行",
        }

    def _reason(self, contribution: np.ndarray) -> str:
        positive = np.flatnonzero(contribution > 0)
        candidates = positive if len(positive) >= 2 else np.arange(len(contribution))
        ordered = candidates[np.argsort(np.abs(contribution[candidates]))[::-1]][:2]
        return "、".join(_feature_label(self.feature_columns[index]) for index in ordered)


def load_frozen_strategy_model(model_dir: Path):
    metadata_path = Path(model_dir) / METADATA_FILENAME
    if not metadata_path.exists():
        raise RuntimeError(f"LGBM 模型元数据不存在: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_version = metadata.get("model_version")
    if model_version == MODEL_VERSION:
        return LgbmStrategyModel(model_dir)
    if model_version == "lgbm_new_factors_blend_v1_candidate":
        from app.lgbm_blend_strategy import LgbmBlendStrategyModel

        return LgbmBlendStrategyModel(model_dir)
    raise RuntimeError(f"LGBM 模型版本不受支持: {model_version}")


def _feature_label(name: str) -> str:
    labels = {
        "x_pb": "PB估值",
        "x_pe": "PE估值",
        "x_price_amount_corr_40": "40日价量相关",
        "x_volatility_10": "10日波动",
        "x_min_return_60": "60日下行尾部",
        "x_efficiency_120": "120日趋势效率",
        "x_volatility_120": "120日波动",
        "x_max_return_10": "10日上行尾部",
        "x_return_120": "120日收益",
        "x_min_return_20": "20日下行尾部",
    }
    return labels.get(name, name.removeprefix("x_").replace("_", " "))
