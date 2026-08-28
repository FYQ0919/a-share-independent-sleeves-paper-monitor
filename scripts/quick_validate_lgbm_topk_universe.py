from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG, build_ranker_training_set, native_lgbm_params
from research_lgbm_topk_universe_v1 import (
    CANDIDATES,
    COST_BPS,
    SELECTION_END,
    SELECTION_START,
    load_history,
    snapshot_universes,
    validate_candidates,
)
from research_qlib_ridge_topk_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    research_summary,
    topk_dropout_backtest,
)


OUTPUT_PATH = BASE_DIR / "data" / "lgbm_topk_universe_quick_validation.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_topk_universe_quick_validation.md"


def fit_frozen_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.Series, dict]:
    import lightgbm as lgb

    as_of = pd.Timestamp(SELECTION_START)
    train, relevance, groups = build_ranker_training_set(
        frame, feature_columns, as_of
    )
    if train["date"].nunique() < 252 or len(train) < 5_000:
        raise RuntimeError("快速冻结LGBM的成熟训练样本不足")
    dataset = lgb.Dataset(
        train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32),
        label=relevance,
        group=groups,
        feature_name=feature_columns,
        free_raw_data=True,
    )
    model = lgb.train(
        native_lgbm_params(WINNER_CONFIG),
        dataset,
        num_boost_round=WINNER_CONFIG["n_estimators"],
    )
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    mask = frame["date"].between(
        pd.Timestamp(SELECTION_START), pd.Timestamp(SELECTION_END)
    )
    prediction.loc[mask] = model.predict(
        frame.loc[mask, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    )
    return prediction, {
        "training_as_of_exclusive": as_of.date().isoformat(),
        "training_rows": int(len(train)),
        "training_dates": int(train["date"].nunique()),
        "max_training_label_end_date": pd.Timestamp(
            train["label_end_date"].max()
        ).date().isoformat(),
        "monthly_refits": 0,
        "frozen_through_selection_window": True,
    }


def annual_priority_passes(row: dict, baseline: dict) -> bool:
    performance = row["selection"]["performance"]
    base_performance = baseline["selection"]["performance"]
    activity = row["selection"]["activity"]
    base_activity = baseline["selection"]["activity"]
    return bool(
        row["name"] != "pool50_top5"
        and performance["annual_return"] > base_performance["annual_return"]
        and performance["sharpe"] >= base_performance["sharpe"]
        and performance["max_drawdown"] >= base_performance["max_drawdown"] - 0.02
        and activity["annual_turnover"] <= base_activity["annual_turnover"] * 1.50
    )


def main() -> None:
    manifest = validate_candidates()
    universes, snapshot_audit = snapshot_universes()
    histories = {}
    coverage = {}
    scores = {}
    model_audits = {}
    for pool_size in (50, 100):
        histories[pool_size], coverage[pool_size] = load_history(universes[pool_size])
        history = histories[pool_size][
            histories[pool_size]["date"].le(pd.Timestamp(SELECTION_END))
        ].copy()
        frame, feature_columns = build_alpha158_lite(history)
        model_prediction, model_audits[pool_size] = fit_frozen_predictions(
            frame, feature_columns
        )
        baseline_prediction = baseline_scores(frame, SELECTION_START, SELECTION_END)
        scores[pool_size] = blend_scores(
            frame,
            model_prediction,
            baseline_prediction,
            WINNER_CONFIG["baseline_weight"],
        )

    candidates = []
    for name, config in CANDIDATES.items():
        pool_size = int(config["pool_size"])
        history = histories[pool_size][
            histories[pool_size]["date"].le(pd.Timestamp(SELECTION_END))
        ].copy()
        result = topk_dropout_backtest(
            history,
            scores[pool_size],
            SELECTION_START,
            SELECTION_END,
            top_k=int(config["top_k"]),
            n_drop=1,
            rebalance_days=10,
            cost_bps=COST_BPS,
        )
        candidates.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
        })

    baseline = next(row for row in candidates if row["name"] == "pool50_top5")
    eligible = [row for row in candidates if annual_priority_passes(row, baseline)]
    winner = max(
        eligible,
        key=lambda row: (
            row["selection"]["performance"]["annual_return"],
            row["selection"]["performance"]["sharpe"],
            row["selection"]["performance"]["max_drawdown"],
        ),
    ) if eligible else baseline
    candidates.sort(
        key=lambda row: row["selection"]["performance"]["annual_return"],
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "quick_directional_candidate" if eligible else "quick_directional_rejected",
        "objective": "increase annual return while keeping Sharpe and limiting drawdown deterioration to 2 percentage points",
        "window": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
        "method": "single LGBM fit using only mature labels before 2018, frozen for all of 2018-2019",
        "candidate_manifest_sha256": manifest,
        "snapshot": snapshot_audit,
        "coverage": {str(key): value for key, value in coverage.items()},
        "model_audit": {str(key): value for key, value in model_audits.items()},
        "winner": winner,
        "annual_priority_gate_passed": bool(eligible),
        "candidates": candidates,
        "production_change": False,
        "warnings": [
            "This is a fast directional validation, not the monthly expanding production-equivalent research run.",
            "The universe uses the current 2026-08-27 snapshot backfilled through history and has survivorship bias.",
            "The result is restricted to 2018-2019 and does not open the repeatedly revealed 2020-2026 window.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# LGBM TopK/股票池年化优先快速验证",
        "",
        f"- 状态：{'发现方向性候选，未部署' if eligible else '没有候选通过'}",
        f"- 冠军：`{winner['name']}`",
        "- 训练：仅使用2018年前成熟标签，随后冻结模型",
        "- 评测：2018-2019",
        "- 生产修改：否",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | 年化换手 |",
        "|---|---:|---:|---:|---:|",
        *[
            "| {name} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} | {turnover:.2f}x |".format(
                name=row["name"],
                annual=row["selection"]["performance"]["annual_return"],
                drawdown=row["selection"]["performance"]["max_drawdown"],
                sharpe=row["selection"]["performance"]["sharpe"],
                turnover=row["selection"]["activity"]["annual_turnover"],
            )
            for row in candidates
        ],
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "winner": winner,
        "annual_priority_gate_passed": bool(eligible),
        "coverage": payload["coverage"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
