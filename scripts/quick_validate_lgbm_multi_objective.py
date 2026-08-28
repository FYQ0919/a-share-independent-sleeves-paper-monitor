from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from app.lgbm_strategy import (
    WINNER_CONFIG,
    build_ranker_training_set,
    eligible_training_mask,
    native_lgbm_params,
)
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_multi_objective_v1 import (
    CANDIDATES,
    DIAGNOSTIC_END,
    DIAGNOSTIC_START,
    LARGE_LOSS_THRESHOLD,
    auxiliary_lgbm_params,
    build_multihead_scores,
    exposure_summary,
    multihead_portfolio_backtest,
)
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
)


FREEZE_DATE = date(2020, 1, 2)
OUTPUT_PATH = BASE_DIR / "data" / "lgbm_multi_objective_quick_validation.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_multi_objective_quick_validation.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_multi_objective_quick_validation_curves.csv"


def fit_frozen_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    freeze_date: date,
) -> tuple[pd.DataFrame, dict]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("快速验证需要 lightgbm==4.6.0") from exc

    cutoff = pd.Timestamp(freeze_date)
    rank_train, relevance, groups = build_ranker_training_set(
        frame, feature_columns, cutoff
    )
    auxiliary_train = frame.loc[eligible_training_mask(frame, cutoff)].copy()
    if rank_train.empty or auxiliary_train.empty:
        raise RuntimeError("冻结日前成熟训练样本不足")
    rank_set = lgb.Dataset(
        rank_train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32),
        label=relevance,
        group=groups,
        feature_name=feature_columns,
        free_raw_data=True,
    )
    rank_model = lgb.train(
        native_lgbm_params(WINNER_CONFIG),
        rank_set,
        num_boost_round=WINNER_CONFIG["n_estimators"],
    )
    auxiliary_matrix = auxiliary_train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    return_set = lgb.Dataset(
        auxiliary_matrix,
        label=auxiliary_train["label_return"].clip(-0.20, 0.20).to_numpy(dtype=np.float32),
        feature_name=feature_columns,
        free_raw_data=False,
    )
    downside_set = lgb.Dataset(
        auxiliary_matrix,
        label=auxiliary_train["label_return"].le(LARGE_LOSS_THRESHOLD).astype(np.int32),
        feature_name=feature_columns,
        free_raw_data=False,
    )
    return_model = lgb.train(
        auxiliary_lgbm_params("regression_l1"), return_set, num_boost_round=100
    )
    downside_model = lgb.train(
        auxiliary_lgbm_params("binary"), downside_set, num_boost_round=100
    )

    prediction_mask = frame["date"].ge(cutoff)
    prediction_index = frame.index[prediction_mask]
    matrix = frame.loc[prediction_mask, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    predictions = pd.DataFrame(
        {
            "rank_prediction": rank_model.predict(matrix),
            "expected_return": return_model.predict(matrix),
            "downside_probability": downside_model.predict(matrix),
        },
        index=prediction_index,
    ).reindex(frame.index)
    max_rank_label_end = pd.to_datetime(rank_train["label_end_date"], errors="coerce").max()
    max_aux_label_end = pd.to_datetime(auxiliary_train["label_end_date"], errors="coerce").max()
    return predictions, {
        "freeze_date": cutoff.date().isoformat(),
        "rank_training_rows": int(len(rank_train)),
        "auxiliary_training_rows": int(len(auxiliary_train)),
        "training_dates": int(auxiliary_train["date"].nunique()),
        "max_rank_label_end_date": max_rank_label_end.date().isoformat(),
        "max_auxiliary_label_end_date": max_aux_label_end.date().isoformat(),
        "strictly_mature": bool(max_rank_label_end < cutoff and max_aux_label_end < cutoff),
        "large_loss_rate": float(auxiliary_train["label_return"].le(LARGE_LOSS_THRESHOLD).mean()),
        "prediction_rows": int(prediction_mask.sum()),
    }


def main() -> None:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
    frame, feature_columns = build_alpha158_lite(history)
    predictions, audit = fit_frozen_predictions(frame, feature_columns, FREEZE_DATE)
    if not audit["strictly_mature"]:
        raise RuntimeError("冻结模型使用了未成熟标签")

    baseline_prediction = baseline_scores(frame, DIAGNOSTIC_START, DIAGNOSTIC_END)
    current_scores = blend_scores(
        frame,
        predictions["rank_prediction"],
        baseline_prediction,
        WINNER_CONFIG["baseline_weight"],
    )
    auxiliary = predictions[["expected_return", "downside_probability"]]
    baseline_config = CANDIDATES["current_rank_top5"]
    pareto_config = CANDIDATES["multihead_pareto_top5"]
    baseline_scores_frame = build_multihead_scores(current_scores, auxiliary, baseline_config)
    pareto_scores_frame = build_multihead_scores(current_scores, auxiliary, pareto_config)
    baseline_result = multihead_portfolio_backtest(
        history, baseline_scores_frame, baseline_config,
        DIAGNOSTIC_START, DIAGNOSTIC_END,
    )
    pareto_result = multihead_portfolio_backtest(
        history, pareto_scores_frame, pareto_config,
        DIAGNOSTIC_START, DIAGNOSTIC_END,
    )
    baseline_summary = research_summary(
        baseline_result, DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    pareto_summary = research_summary(
        pareto_result, DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    baseline_perf = baseline_summary["performance"]
    pareto_perf = pareto_summary["performance"]
    baseline_calmar = baseline_perf["annual_return"] / abs(baseline_perf["max_drawdown"])
    pareto_calmar = pareto_perf["annual_return"] / abs(pareto_perf["max_drawdown"])
    baseline_periods = pd.DataFrame(baseline_result["return_periods"])
    pareto_periods = pd.DataFrame(pareto_result["return_periods"])
    balanced_returns = (
        baseline_periods["strategy_return"].astype(float).mul(0.50)
        + pareto_periods["strategy_return"].astype(float).mul(0.50)
    )
    balanced_equity = (1 + balanced_returns).cumprod()
    balanced_curve_values = pd.Series(
        np.r_[1.0, balanced_equity.to_numpy(dtype=float)]
    )
    balanced_drawdown = balanced_curve_values.div(
        balanced_curve_values.cummax()
    ).sub(1)
    balanced_volatility = float(balanced_returns.std(ddof=0) * np.sqrt(252))
    balanced_annual = float(
        balanced_equity.iloc[-1] ** (252 / len(balanced_returns)) - 1
    )
    balanced_sharpe = float(
        balanced_returns.mean() * 252 / balanced_volatility
    ) if balanced_volatility > 0 else 0.0
    balanced_metrics = {
        "allocation": "50% current frozen LGBM + 50% multi-objective LGBM",
        "total_return": float(balanced_equity.iloc[-1] - 1),
        "annual_return": balanced_annual,
        "max_drawdown": float(balanced_drawdown.min()),
        "sharpe": balanced_sharpe,
        "calmar": balanced_annual / abs(float(balanced_drawdown.min())),
        "annual_return_retention": balanced_annual / baseline_perf["annual_return"],
        "relative_drawdown_improvement": 1
        - abs(float(balanced_drawdown.min())) / abs(baseline_perf["max_drawdown"]),
        "status": "revealed-diagnostic research candidate; forward paper required",
    }
    checks = {
        "annual_return_not_lower": pareto_perf["annual_return"] >= baseline_perf["annual_return"],
        "maximum_drawdown_not_worse": pareto_perf["max_drawdown"] >= baseline_perf["max_drawdown"],
        "sharpe_not_lower": pareto_perf["sharpe"] >= baseline_perf["sharpe"],
        "calmar_not_lower": pareto_calmar >= baseline_calmar,
        "turnover_not_higher_10pct": pareto_summary["activity"]["annual_turnover"]
        <= baseline_summary["activity"]["annual_turnover"] * 1.10,
    }
    passed = all(checks.values())
    curves = pd.concat(
        {
            "multi_objective_lgbm": normalized_curve(pareto_result, "strategy_return"),
            "current_lgbm_top5": normalized_curve(baseline_result, "strategy_return"),
            "pool_benchmark": normalized_curve(baseline_result, "benchmark_return"),
        },
        axis=1,
        join="inner",
    )
    curves["balanced_50_50"] = balanced_curve_values.to_numpy()
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "quick_validation_passed_not_deployed" if passed else "quick_validation_failed",
        "method": "single frozen pre-2020 three-head LGBM; no post-2020 refit",
        "window": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        "freeze_date": FREEZE_DATE.isoformat(),
        "strategy_config": pareto_config,
        "execution_contract": "signal close; next-open execution; Top5; 10-session rebalance; 12bp one-way cost",
        "current_lgbm_top5": baseline_summary,
        "multi_objective_lgbm": pareto_summary,
        "current_calmar": baseline_calmar,
        "multi_objective_calmar": pareto_calmar,
        "balanced_50_50_research_candidate": balanced_metrics,
        "current_exposure": exposure_summary(baseline_result),
        "multi_objective_exposure": exposure_summary(pareto_result),
        "gate": {"passed": passed, "checks": checks},
        "causality_audit": audit,
        "universe_snapshot_run_id": snapshot_id,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "production_change": False,
        "warnings": warnings + [
            "This is a fast frozen-model diagnostic, not the expanding monthly production protocol.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "2020-2026 has already been revealed and cannot be used to retune the weights.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        "\n".join([
            "# LGBM 年化与回撤快速冻结验证",
            "",
            f"- 状态：{'快速门禁通过，未部署' if passed else '快速门禁未通过'}",
            f"- 冻结日期：{FREEZE_DATE}",
            f"- 验证区间：{DIAGNOSTIC_START} 至 {DIAGNOSTIC_END}",
            f"- 当前LGBM：年化 {baseline_perf['annual_return']:.2%}，回撤 {baseline_perf['max_drawdown']:.2%}，夏普 {baseline_perf['sharpe']:.3f}，Calmar {baseline_calmar:.3f}",
            f"- 多目标LGBM：年化 {pareto_perf['annual_return']:.2%}，回撤 {pareto_perf['max_drawdown']:.2%}，夏普 {pareto_perf['sharpe']:.3f}，Calmar {pareto_calmar:.3f}",
            f"- 50/50均衡候选：年化 {balanced_metrics['annual_return']:.2%}，回撤 {balanced_metrics['max_drawdown']:.2%}，夏普 {balanced_metrics['sharpe']:.3f}，Calmar {balanced_metrics['calmar']:.3f}",
            "- 生产修改：否",
            "",
            "模型只使用冻结日前完整成熟的标签，2020年后不重训。本结果用于快速结构验证，不替代前向模拟盘。",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "baseline": baseline_summary,
        "multi_objective": pareto_summary,
        "baseline_calmar": baseline_calmar,
        "multi_objective_calmar": pareto_calmar,
        "balanced_50_50_research_candidate": balanced_metrics,
        "gate": payload["gate"],
        "causality_audit": audit,
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
