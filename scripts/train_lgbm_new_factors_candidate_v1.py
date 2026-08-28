from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.backtest_engine import BacktestEngine
from app.config import BASE_DIR
from app.lgbm_strategy import (
    BASELINE_WEIGHTS,
    WINNER_CONFIG,
    build_ranker_training_set,
    feature_schema_hash,
    file_sha256,
    native_lgbm_params,
)
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_new_factors_v1 import build_extended_feature_frame
from research_qlib_ridge_topk_v1 import SOURCE_OPTIMIZATION


MODEL_VERSION = "lgbm_new_factors_blend_v1_candidate"
BASELINE_MODEL_FILE = "alpha158_model.txt"
BARRA_MODEL_FILE = "alpha158_barra_model.txt"
METADATA_FILE = "metadata.json"
CANDIDATE_WEIGHT = 0.25
RULE_WEIGHT = 0.40
MODEL_DIR = BASE_DIR / "data" / "models" / MODEL_VERSION
RESEARCH_RESULT = BASE_DIR / "data" / "lgbm_new_factors_v1.json"
SIGNAL_OUTPUT = BASE_DIR / "data" / "lgbm_new_factors_forward_signal_v1.json"
REPORT_OUTPUT = BASE_DIR / "reports" / "lgbm_new_factors_forward_candidate_v1.md"


def blend_candidate_model_ranks(
    baseline_prediction: pd.Series,
    candidate_prediction: pd.Series,
    candidate_weight: float = CANDIDATE_WEIGHT,
) -> pd.Series:
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("新因子模型权重必须在 [0, 1]")
    baseline_rank = baseline_prediction.rank(pct=True)
    candidate_rank = candidate_prediction.rank(pct=True)
    blended = baseline_rank.mul(1.0 - candidate_weight).add(
        candidate_rank.mul(candidate_weight)
    )
    return blended.rank(pct=True)


def train_booster(
    frame: pd.DataFrame,
    feature_columns: list[str],
    training_as_of: pd.Timestamp,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("冻结新因子候选需要 lightgbm==4.6.0") from exc

    train, relevance, groups = build_ranker_training_set(
        frame, feature_columns, training_as_of
    )
    if train["date"].nunique() < 504 or len(train) < 10_000:
        raise RuntimeError("候选模型至少需要504个训练日和10000行成熟样本")
    dataset = lgb.Dataset(
        train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32),
        label=relevance,
        group=groups,
        feature_name=feature_columns,
        free_raw_data=True,
    )
    booster = lgb.train(
        native_lgbm_params(WINNER_CONFIG),
        dataset,
        num_boost_round=WINNER_CONFIG["n_estimators"],
    )
    max_label_end = pd.to_datetime(train["label_end_date"], errors="coerce").max()
    if not max_label_end < training_as_of:
        raise RuntimeError("候选模型使用了训练截止日尚未成熟的标签")
    gains = booster.feature_importance(importance_type="gain").astype(float)
    importance = sorted(
        (
            {"feature": feature_columns[index], "gain": float(gain)}
            for index, gain in enumerate(gains)
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    return booster, {
        "feature_count": len(feature_columns),
        "feature_schema_sha256": feature_schema_hash(feature_columns),
        "feature_columns": feature_columns,
        "training_rows": int(len(train)),
        "training_dates": int(train["date"].nunique()),
        "training_signal_start": pd.Timestamp(train["date"].min()).date().isoformat(),
        "training_signal_end": pd.Timestamp(train["date"].max()).date().isoformat(),
        "max_training_label_end_date": max_label_end.date().isoformat(),
        "strictly_mature": True,
        "top_feature_importance": importance[:25],
    }


def latest_candidate_rank(
    frame: pd.DataFrame,
    base_features: list[str],
    candidate_features: list[str],
    baseline_model,
    candidate_model,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    available = pd.DatetimeIndex(
        frame.loc[frame["date"].le(as_of), "date"].dropna().unique()
    )
    if available.empty:
        raise RuntimeError("没有可生成候选信号的交易日")
    signal_date = available.max()
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
        raise RuntimeError("信号日可交易股票不足5只")

    baseline_prediction = pd.Series(
        baseline_model.predict(
            cross[base_features].fillna(0.0).to_numpy(dtype=np.float32)
        ),
        index=cross.index,
    )
    candidate_prediction = pd.Series(
        candidate_model.predict(
            cross[candidate_features].fillna(0.0).to_numpy(dtype=np.float32)
        ),
        index=cross.index,
    )
    cross["alpha158_lgbm_rank"] = baseline_prediction.rank(pct=True)
    cross["barra_lgbm_rank"] = candidate_prediction.rank(pct=True)
    cross["lgbm_ensemble_rank"] = blend_candidate_model_ranks(
        baseline_prediction, candidate_prediction
    )
    rule = BacktestEngine()._score(cross.copy(), BASELINE_WEIGHTS)
    cross["baseline_rule_rank"] = rule["score"].reindex(cross.index).rank(pct=True)
    cross["score"] = (
        cross["lgbm_ensemble_rank"].mul(1.0 - RULE_WEIGHT)
        + cross["baseline_rule_rank"].mul(RULE_WEIGHT)
    ).mul(100.0)
    ranked = cross.sort_values(
        ["score", "amount_20"], ascending=[False, False]
    ).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["signal_date"] = signal_date
    return ranked


def main() -> None:
    research = json.loads(RESEARCH_RESULT.read_text(encoding="utf-8"))
    if not research.get("promotion_gate", {}).get("passed"):
        raise RuntimeError("新因子 LGBM 历史门禁未通过，拒绝冻结前向候选")
    if research.get("selected_feature_batch") != "alpha158_plus_barra":
        raise RuntimeError("研究选择结构与候选冻结结构不一致")
    if float(research.get("selected_candidate_weight", -1)) != CANDIDATE_WEIGHT:
        raise RuntimeError("研究选择权重与候选冻结权重不一致")

    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    latest_date = pd.Timestamp(history["date"].max())
    training_as_of = latest_date + timedelta(days=1)
    frame, feature_sets = build_extended_feature_frame(history)
    base_features = feature_sets["alpha158_baseline"]
    candidate_features = feature_sets["alpha158_plus_barra"]

    baseline_model, baseline_audit = train_booster(
        frame, base_features, training_as_of
    )
    candidate_model, candidate_audit = train_booster(
        frame, candidate_features, training_as_of
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = MODEL_DIR / BASELINE_MODEL_FILE
    candidate_path = MODEL_DIR / BARRA_MODEL_FILE
    baseline_path.write_text(baseline_model.model_to_string(), encoding="utf-8")
    candidate_path.write_text(candidate_model.model_to_string(), encoding="utf-8")

    ranked = latest_candidate_rank(
        frame,
        base_features,
        candidate_features,
        baseline_model,
        candidate_model,
        latest_date,
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "artifact_status": "frozen_for_forward_shadow_only",
        "created_at": pd.Timestamp.now().isoformat(),
        "training_as_of_exclusive": training_as_of.date().isoformat(),
        "model_config": WINNER_CONFIG,
        "rule_weight": RULE_WEIGHT,
        "alpha158_model_weight": 1.0 - CANDIDATE_WEIGHT,
        "alpha158_barra_model_weight": CANDIDATE_WEIGHT,
        "baseline_model_file": BASELINE_MODEL_FILE,
        "baseline_model_sha256": file_sha256(baseline_path),
        "candidate_model_file": BARRA_MODEL_FILE,
        "candidate_model_sha256": file_sha256(candidate_path),
        "baseline_model_audit": baseline_audit,
        "candidate_model_audit": candidate_audit,
        "research_result": str(RESEARCH_RESULT.relative_to(BASE_DIR)),
        "research_result_sha256": file_sha256(RESEARCH_RESULT),
        "universe_snapshot_run_id": snapshot_id,
        "execution_contract": "signal close; next-open; Top5; 10-session rebalance; 12bp one-way cost",
        "strict_no_lookahead_certified": False,
        "production_change": False,
        "warnings": warnings + [
            "Historical promotion used an already revealed 2020-2026 blocker window.",
            "The current Top50 universe is backfilled and has survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "This artifact is shadow-only until a new forward paper window is completed.",
        ],
    }
    metadata_path = MODEL_DIR / METADATA_FILE
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    picks = []
    for _, row in ranked.head(5).iterrows():
        name = row.get("name")
        picks.append({
            "rank": int(row["rank"]),
            "code": str(row["code"]),
            "name": str(row["code"]) if pd.isna(name) else str(name),
            "signal_date": pd.Timestamp(row["signal_date"]).date().isoformat(),
            "close": round(float(row["close"]), 2),
            "score": round(float(row["score"]), 2),
            "alpha158_lgbm_rank": round(float(row["alpha158_lgbm_rank"]) * 100, 2),
            "barra_lgbm_rank": round(float(row["barra_lgbm_rank"]) * 100, 2),
            "baseline_rule_rank": round(float(row["baseline_rule_rank"]) * 100, 2),
            "target_weight": 0.20,
        })
    signal = {
        "status": "forward_shadow_candidate_ready",
        "model_version": MODEL_VERSION,
        "signal_date": latest_date.date().isoformat(),
        "picks": picks,
        "model_dir": str(MODEL_DIR.relative_to(BASE_DIR)),
        "production_change": False,
        "warnings": metadata["warnings"],
    }
    SIGNAL_OUTPUT.write_text(
        json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_OUTPUT.write_text(
        "\n".join([
            "# 新因子 LGBM 前向影子候选 v1",
            "",
            "- 状态：模型已冻结，仅用于全新交易日影子观察",
            f"- 模型版本：`{MODEL_VERSION}`",
            f"- 训练截止（不含）：{training_as_of.date().isoformat()}",
            f"- 最新信号日：{latest_date.date().isoformat()}",
            "- 结构：40%规则分 + 60% LGBM组合分",
            "- LGBM组合：75% Alpha158 + 25% Alpha158+Barra",
            "- 执行：收盘信号，下一交易日开盘，Top5等权，10日主调仓，单边12bp",
            "- 生产修改：否",
            "",
            "## 最新影子 Top5",
            "",
            *[
                f"- {row['rank']}. {row['name']}（{row['code']}），{row['score']:.2f}分"
                for row in picks
            ],
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model_dir": str(MODEL_DIR),
        "metadata": metadata,
        "signal": signal,
        "report": str(REPORT_OUTPUT),
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
