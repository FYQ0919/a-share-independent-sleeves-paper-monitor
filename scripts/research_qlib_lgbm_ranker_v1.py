from __future__ import annotations

from datetime import date, datetime
import hashlib
import json

import numpy as np
import pandas as pd

from app.config import BASE_DIR
from app.lgbm_strategy import (
    build_ranker_training_set as _build_ranker_training_set,
    eligible_training_mask as _eligible_training_mask,
    native_lgbm_params,
)
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_ridge_topk_v1 import (
    DATA_START,
    LABEL_HORIZON,
    SELECTION_END,
    SELECTION_START,
    SOURCE_OPTIMIZATION,
    TEST_END,
    TEST_START,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    production_baseline,
    research_summary,
    topk_dropout_backtest,
)


OUTPUT_PATH = BASE_DIR / "data" / "qlib_lgbm_ranker_research_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "qlib_lgbm_ranker_research_v1.md"
CURVES_PATH = BASE_DIR / "reports" / "qlib_lgbm_ranker_2020_curves.csv"
RELEVANCE_LEVELS = 5
RANDOM_SEED = 20260827

# Frozen before the 2020 diagnostic. The small grid limits model-selection
# degrees of freedom on a short history with only a few market regimes.
CANDIDATES = {
    "lgbm_l7_blend40": {
        "baseline_weight": 0.40,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 300,
        "n_estimators": 120,
        "feature_fraction": 0.70,
    },
    "lgbm_l15_blend40": {
        "baseline_weight": 0.40,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 300,
        "n_estimators": 160,
        "feature_fraction": 0.70,
    },
    "lgbm_l15_blend20": {
        "baseline_weight": 0.20,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 300,
        "n_estimators": 160,
        "feature_fraction": 0.70,
    },
}


def validate_candidates() -> str:
    expected = {
        "baseline_weight", "num_leaves", "max_depth", "min_child_samples",
        "n_estimators", "feature_fraction",
    }
    for name, config in CANDIDATES.items():
        if set(config) != expected:
            raise RuntimeError(f"{name} 参数不完整")
        if not 0 <= config["baseline_weight"] <= 1:
            raise RuntimeError(f"{name} baseline_weight 越界")
        if config["num_leaves"] > 2 ** config["max_depth"]:
            raise RuntimeError(f"{name} num_leaves 超过深度容量")
        if config["min_child_samples"] < 200:
            raise RuntimeError(f"{name} 叶节点样本过少")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def eligible_training_mask(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    return _eligible_training_mask(frame, as_of)


def build_ranker_training_set(
    frame: pd.DataFrame,
    feature_columns: list[str],
    as_of: pd.Timestamp,
    relevance_levels: int = RELEVANCE_LEVELS,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    return _build_ranker_training_set(frame, feature_columns, as_of, relevance_levels)


def lgbm_model_params(config: dict) -> dict:
    return native_lgbm_params(config)


def fit_expanding_lgbm_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    prediction_start: date,
    prediction_end: date,
) -> tuple[pd.Series, dict]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "缺少 lightgbm。请安装项目 research 可选依赖后重新运行："
            "python -m pip install -e .[research]"
        ) from exc

    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).unique()))
    prediction_dates = dates[
        (dates >= pd.Timestamp(prediction_start)) & (dates <= pd.Timestamp(prediction_end))
    ]
    params = lgbm_model_params(config)
    maturity_audit = []
    importance_rows = []

    grouped_months = pd.Series(prediction_dates, index=prediction_dates).groupby(
        [prediction_dates.year, prediction_dates.month]
    )
    for month, month_dates in grouped_months:
        first_prediction_date = pd.Timestamp(month_dates.iloc[0])
        train, relevance, groups = build_ranker_training_set(
            frame, feature_columns, first_prediction_date
        )
        if train["date"].nunique() < 252 or len(train) < 5_000:
            continue

        train_set = lgb.Dataset(
            train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32),
            label=relevance,
            group=groups,
            feature_name=feature_columns,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            train_set,
            num_boost_round=config["n_estimators"],
        )
        month_mask = frame["date"].isin(pd.DatetimeIndex(month_dates.values))
        predictions.loc[month_mask] = model.predict(
            frame.loc[month_mask, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        )
        importance_rows.append(
            model.feature_importance(importance_type="gain").astype(float)
        )
        maturity_audit.append({
            "month": f"{month[0]:04d}-{month[1]:02d}",
            "prediction_start": first_prediction_date.date().isoformat(),
            "max_training_label_end_date": pd.to_datetime(
                train["label_end_date"]
            ).max().date().isoformat(),
            "gap_sessions": LABEL_HORIZON + 1,
            "training_rows": int(len(train)),
            "training_dates": int(train["date"].nunique()),
            "group_count": len(groups),
            "max_relevance": int(relevance.max()),
        })

    mean_gain = (
        np.mean(np.vstack(importance_rows), axis=0)
        if importance_rows else np.zeros(len(feature_columns))
    )
    importance = sorted(
        (
            {"feature": feature_columns[index], "mean_gain": float(value)}
            for index, value in enumerate(mean_gain)
        ),
        key=lambda row: row["mean_gain"],
        reverse=True,
    )
    return predictions, {
        "model_params": params,
        "refit_months": len(maturity_audit),
        "maturity_audit": maturity_audit,
        "top_feature_importance": importance[:20],
    }


def write_dependency_blocked(message: str) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "dependency_blocked",
        "backtest_executed": False,
        "strategy": "Alpha158-lite expanding LightGBM LambdaRank + Top5 Dropout",
        "reason": message,
        "required_dependency": "lightgbm==4.6.0",
        "install_command": "python -m pip install -e .[research]",
        "production_change": False,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        f"""# LightGBM LambdaRank 研究 v1

- 状态：**依赖阻断，尚未执行回测**
- 原因：{message}
- 需要依赖：`lightgbm==4.6.0`
- 安装命令：`python -m pip install -e .[research]`
- 生产策略修改：否

LGBM 训练、成熟标签分组、冻结候选和样本外门禁代码已经就绪；本文件不包含任何替代模型或推测收益。
""",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def selection_folds(result: dict) -> dict:
    return {
        "2018_stress": research_summary(result, date(2018, 1, 1), date(2018, 12, 31)),
        "2019_recovery": research_summary(result, date(2019, 1, 1), date(2019, 12, 31)),
    }


def candidate_rank(row: dict) -> tuple:
    fold_sharpes = [fold["performance"]["sharpe"] for fold in row["selection_folds"].values()]
    performance = row["selection"]["performance"]
    activity = row["selection"]["activity"]
    return (
        min(fold_sharpes),
        float(np.median(fold_sharpes)),
        performance["sharpe"],
        performance["annual_return"],
        -activity["annual_turnover"],
        performance["max_drawdown"],
    )


def export_test_curves(winner_result: dict, production_result: dict) -> int:
    curves = pd.concat(
        {
            "lgbm_ranker_top5": normalized_curve(winner_result, "strategy_return"),
            "production_baseline": normalized_curve(production_result, "strategy_return"),
            "pool_benchmark": normalized_curve(winner_result, "benchmark_return"),
        },
        axis=1,
        join="inner",
    )
    if curves.empty or not np.allclose(curves.iloc[0].to_numpy(), 1.0):
        raise RuntimeError("LGBM 2020曲线无法对齐到共同起点")
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVES_PATH, encoding="utf-8-sig", float_format="%.10f")
    return len(curves)


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history = history[history["date"] <= pd.Timestamp(TEST_END)].copy()
    codes = sorted(history["code"].unique().tolist())
    frame, feature_columns = build_alpha158_lite(history)
    baseline_prediction = baseline_scores(frame, SELECTION_START, TEST_END)

    candidate_rows = []
    selection_prediction_cache = {}
    for name, config in CANDIDATES.items():
        model_key = tuple(
            (key, value) for key, value in config.items() if key != "baseline_weight"
        )
        if model_key not in selection_prediction_cache:
            selection_prediction_cache[model_key] = fit_expanding_lgbm_predictions(
                frame, feature_columns, config, SELECTION_START, SELECTION_END
            )
        prediction, audit = selection_prediction_cache[model_key]
        scores = blend_scores(
            frame, prediction, baseline_prediction, config["baseline_weight"]
        )
        result = topk_dropout_backtest(history, scores, SELECTION_START, SELECTION_END)
        candidate_rows.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
            "selection_folds": selection_folds(result),
            "model_refits": audit["refit_months"],
        })

    winner = max(candidate_rows, key=candidate_rank)
    winner_config = winner["config"]
    prediction, model_audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, winner_config, SELECTION_START, TEST_END
    )
    winner_scores = blend_scores(
        frame, prediction, baseline_prediction, winner_config["baseline_weight"]
    )
    winner_result = topk_dropout_backtest(history, winner_scores, TEST_START, TEST_END)
    winner_test = research_summary(winner_result, TEST_START, TEST_END)

    production = production_baseline(history, codes, include_test_result=True)
    production_result = production.pop("_test_result")
    production_test = production["test"]
    curve_rows = export_test_curves(winner_result, production_result)
    checks = {
        "annual_return_improved_vs_production": (
            winner_test["performance"]["annual_return"]
            > production_test["performance"]["annual_return"]
        ),
        "sharpe_improved_vs_production": (
            winner_test["performance"]["sharpe"]
            > production_test["performance"]["sharpe"]
        ),
        "drawdown_not_worse_by_3pp": (
            winner_test["performance"]["max_drawdown"]
            >= production_test["performance"]["max_drawdown"] - 0.03
        ),
        "annual_turnover_not_higher": (
            winner_test["activity"]["annual_turnover"]
            <= production_test["activity"]["annual_turnover"]
        ),
    }
    passed = all(checks.values())
    candidate_rows.sort(key=candidate_rank, reverse=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "strategy": "Alpha158-lite expanding LightGBM LambdaRank + Top5 Dropout",
        "model_backend": "lightgbm",
        "windows": {
            "data_and_model_warmup": [DATA_START.isoformat(), "2017-12-31"],
            "candidate_selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_online_diagnostic": [TEST_START.isoformat(), TEST_END.isoformat()],
            "2021_plus": "not_evaluated",
        },
        "label": (
            "five-level cross-sectional relevance from signal-close features and "
            "10-session next-open-to-open returns"
        ),
        "portfolio": "Top5, evaluate every 10 sessions, drop at most 1, 12 bps one-way cost",
        "universe_snapshot_run_id": snapshot_id,
        "candidate_manifest_sha256": manifest,
        "candidate_count": len(CANDIDATES),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "selection_rule": (
            "Freeze three candidates, rank on 2018-2019 by worst yearly Sharpe, median yearly "
            "Sharpe, aggregate Sharpe/return, lower turnover, and drawdown; only then inspect 2020."
        ),
        "production_baseline": production,
        "selection_winner": winner,
        "winner_model_audit": model_audit,
        "winner_test": winner_test,
        "test_gate": {"passed": passed, "checks": checks},
        "production_change": False,
        "curve_data": str(CURVES_PATH.relative_to(BASE_DIR)),
        "curve_rows": curve_rows,
        "promotion_blockers": [
            "2020窗口已在先前研究中揭示，只能作为复用诊断而非全新盲测",
            "冻结的当前Top50历史回填存在成分、上市和存续偏差",
            "当前版本前复权价格不满足严格as-of价格门禁",
            "财务字段缺少逐字段available_date",
            "需要冻结后至少126个交易日前向模拟盘",
        ],
        "candidates": candidate_rows,
        "warnings": warnings,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        f"""# LightGBM LambdaRank + Alpha158-lite + Top5 Dropout v1

- 状态：研究结果，**未修改生产策略**
- 冻结股票池：`{snapshot_id}`
- 候选哈希：`{manifest}`
- 训练选择冠军：`{winner['name']}`
- 2018-2019选择年化：{winner['selection']['performance']['annual_return']:.2%}
- 2018-2019选择夏普：{winner['selection']['performance']['sharpe']:.3f}
- 2020复用诊断年化：{winner_test['performance']['annual_return']:.2%}
- 2020复用诊断夏普：{winner_test['performance']['sharpe']:.3f}
- 2020生产基线年化：{production_test['performance']['annual_return']:.2%}
- 2020生产基线夏普：{production_test['performance']['sharpe']:.3f}
- 2020策略年化换手：{winner_test['activity']['annual_turnover']:.2f}x
- 2020生产基线年化换手：{production_test['activity']['annual_turnover']:.2f}x
- 测试门禁：{'通过但仅限研究' if passed else '未通过'}

## 约束

- 每月扩展窗口重训；每行标签结束日不得晚于预测月首日。
- 2020已经在此前研究中查看，只能作为诊断，不能重新用于调参。
- 当前Top50回填和当前版本前复权价格仍阻止严格无前视认证。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "curves": str(CURVES_PATH),
        "winner": winner,
        "winner_test": winner_test,
        "production_test": production_test,
        "test_gate": payload["test_gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "缺少 lightgbm" not in str(exc):
            raise
        write_dependency_blocked(str(exc))
        raise SystemExit(2) from None
