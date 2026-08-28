from __future__ import annotations

from datetime import date, datetime
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

from app.alpha101_extra_factors import add_alpha101_extra_factors
from app.barra_residual_factors import add_barra_residual_factors
from app.config import BASE_DIR
from app.lgbm_features import build_alpha158_lite
from app.lgbm_strategy import (
    WINNER_CONFIG,
    build_ranker_training_set,
    native_lgbm_params,
)
from app.qlib_multiscale_factors import add_qlib_multiscale_factors
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_multi_objective_v1 import (
    CANDIDATES as PORTFOLIO_CANDIDATES,
    multihead_portfolio_backtest,
)
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    normalized_curve,
    research_summary,
)


SELECTION_FREEZE_DATE = date(2018, 1, 1)
SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_FREEZE_DATE = date(2020, 1, 2)
DIAGNOSTIC_START = date(2020, 1, 1)
DIAGNOSTIC_END = date(2026, 8, 25)

FEATURE_BATCH_NAMES = (
    "alpha158_baseline",
    "alpha158_plus_alpha101",
    "alpha158_plus_qlib_multiscale",
    "alpha158_plus_barra",
    "alpha158_plus_all_new",
)
MODEL_BLEND_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

OUTPUT_PATH = BASE_DIR / "data" / "lgbm_new_factors_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_new_factors_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_new_factors_v1_curves.csv"


def build_extended_feature_frame(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame, base_features = build_alpha158_lite(history)
    frame, alpha101_features = add_alpha101_extra_factors(frame)
    frame, qlib_features = add_qlib_multiscale_factors(frame)
    frame, barra_features = add_barra_residual_factors(frame)
    feature_sets = {
        "alpha158_baseline": base_features,
        "alpha158_plus_alpha101": base_features + alpha101_features,
        "alpha158_plus_qlib_multiscale": base_features + qlib_features,
        "alpha158_plus_barra": base_features + barra_features,
        "alpha158_plus_all_new": (
            base_features + alpha101_features + qlib_features + barra_features
        ),
    }
    if tuple(feature_sets) != FEATURE_BATCH_NAMES:
        raise RuntimeError("LGBM 新因子候选顺序发生变化")
    for name, features in feature_sets.items():
        if len(features) != len(set(features)):
            raise RuntimeError(f"{name} 存在重复特征")
        missing = sorted(set(features).difference(frame.columns))
        if missing:
            raise RuntimeError(f"{name} 缺少特征列: {', '.join(missing)}")
    return frame, feature_sets


def fit_frozen_rank_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    freeze_date: date,
) -> tuple[pd.Series, dict]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("新因子 LGBM 策略需要 lightgbm==4.6.0") from exc

    cutoff = pd.Timestamp(freeze_date)
    train, relevance, groups = build_ranker_training_set(
        frame, feature_columns, cutoff
    )
    if train.empty or len(groups) == 0:
        raise RuntimeError(f"{freeze_date} 前成熟训练样本不足")
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
    prediction_mask = frame["date"].ge(cutoff)
    matrix = frame.loc[prediction_mask, feature_columns].fillna(0.0).to_numpy(
        dtype=np.float32
    )
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    predictions.loc[prediction_mask] = model.predict(matrix)
    max_label_end = pd.to_datetime(train["label_end_date"], errors="coerce").max()
    gains = model.feature_importance(importance_type="gain").astype(float)
    importance = sorted(
        (
            {"feature": feature_columns[index], "gain": float(gain)}
            for index, gain in enumerate(gains)
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    return predictions, {
        "freeze_date": cutoff.date().isoformat(),
        "feature_count": len(feature_columns),
        "training_rows": int(len(train)),
        "training_dates": int(train["date"].nunique()),
        "max_training_label_end_date": max_label_end.date().isoformat(),
        "strictly_mature": bool(max_label_end < cutoff),
        "prediction_rows": int(prediction_mask.sum()),
        "top_feature_importance": importance[:25],
    }


def blend_model_predictions(
    frame: pd.DataFrame,
    baseline_predictions: pd.Series,
    candidate_predictions: pd.Series,
    candidate_weight: float,
) -> pd.Series:
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("新因子模型融合权重必须在 [0, 1]")
    baseline_rank = baseline_predictions.groupby(frame["date"]).rank(pct=True)
    candidate_ranked = candidate_predictions.groupby(frame["date"]).rank(pct=True)
    return baseline_rank.mul(1.0 - candidate_weight).add(
        candidate_ranked.mul(candidate_weight)
    )


def evaluate_predictions(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    predictions: pd.Series,
    start: date,
    end: date,
) -> tuple[dict, dict]:
    rule_scores = baseline_scores(frame, start, end)
    scores = blend_scores(
        frame,
        predictions,
        rule_scores,
        WINNER_CONFIG["baseline_weight"],
    )
    scores["expected_return"] = 0.0
    scores["downside_probability"] = 0.0
    result = multihead_portfolio_backtest(
        history,
        scores,
        PORTFOLIO_CANDIDATES["current_rank_top5"],
        start,
        end,
        rebalance_days=10,
        n_drop=1,
        cost_bps=12.0,
    )
    return research_summary(result, start, end), result


def evaluate_frozen_model(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    feature_columns: list[str],
    freeze_date: date,
    start: date,
    end: date,
) -> tuple[dict, dict, dict, pd.Series]:
    predictions, audit = fit_frozen_rank_predictions(
        frame, feature_columns, freeze_date
    )
    if not audit["strictly_mature"]:
        raise RuntimeError("冻结 LGBM 使用了预测日尚未成熟的标签")
    summary, result = evaluate_predictions(
        history, frame, predictions, start, end
    )
    return summary, audit, result, predictions


def candidate_rank(row: dict) -> tuple[float, ...]:
    performance = row["selection"]["performance"]
    activity = row["selection"]["activity"]
    return (
        float(performance["sharpe"]),
        float(performance["annual_return"]),
        float(performance["max_drawdown"]),
        float(performance["information_ratio"]),
        -float(activity["annual_turnover"]),
    )


def promotion_checks(candidate: dict, baseline: dict) -> dict[str, bool]:
    candidate_perf = candidate["performance"]
    baseline_perf = baseline["performance"]
    return {
        "annual_return_not_lower": (
            candidate_perf["annual_return"] >= baseline_perf["annual_return"]
        ),
        "sharpe_not_lower": candidate_perf["sharpe"] >= baseline_perf["sharpe"],
        "max_drawdown_not_worse": (
            candidate_perf["max_drawdown"] >= baseline_perf["max_drawdown"]
        ),
        "annual_turnover_not_higher_10pct": (
            candidate["activity"]["annual_turnover"]
            <= baseline["activity"]["annual_turnover"] * 1.10
        ),
    }


def main() -> None:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
    frame, feature_sets = build_extended_feature_frame(history)

    selection_rows = []
    selection_predictions: dict[str, pd.Series] = {}
    for name, features in feature_sets.items():
        summary, audit, _, predictions = evaluate_frozen_model(
            history,
            frame,
            features,
            SELECTION_FREEZE_DATE,
            SELECTION_START,
            SELECTION_END,
        )
        selection_rows.append({
            "name": name,
            "feature_count": len(features),
            "selection": summary,
            "causality_audit": audit,
        })
        selection_predictions[name] = predictions

    winner = max(selection_rows, key=candidate_rank)
    winner_name = str(winner["name"])
    blend_selection_rows = []
    for weight in MODEL_BLEND_WEIGHTS:
        predictions = blend_model_predictions(
            frame,
            selection_predictions["alpha158_baseline"],
            selection_predictions[winner_name],
            weight,
        )
        summary, _ = evaluate_predictions(
            history, frame, predictions, SELECTION_START, SELECTION_END
        )
        blend_selection_rows.append({
            "name": f"baseline_{1 - weight:.2f}_new_{weight:.2f}",
            "candidate_weight": weight,
            "selection": summary,
        })
    blend_winner = max(blend_selection_rows, key=candidate_rank)
    selected_blend_weight = float(blend_winner["candidate_weight"])

    diagnostic_names = ["alpha158_baseline"]
    if winner_name != "alpha158_baseline":
        diagnostic_names.append(winner_name)

    diagnostics = {}
    diagnostic_results = {}
    diagnostic_predictions: dict[str, pd.Series] = {}
    for name in diagnostic_names:
        summary, audit, result, predictions = evaluate_frozen_model(
            history,
            frame,
            feature_sets[name],
            DIAGNOSTIC_FREEZE_DATE,
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
        )
        diagnostics[name] = {
            "feature_count": len(feature_sets[name]),
            "diagnostic": summary,
            "causality_audit": audit,
        }
        diagnostic_results[name] = result
        diagnostic_predictions[name] = predictions

    blended_predictions = blend_model_predictions(
        frame,
        diagnostic_predictions["alpha158_baseline"],
        diagnostic_predictions[winner_name],
        selected_blend_weight,
    )
    blend_summary, blend_result = evaluate_predictions(
        history,
        frame,
        blended_predictions,
        DIAGNOSTIC_START,
        DIAGNOSTIC_END,
    )
    blend_name = "selected_model_blend"
    diagnostics[blend_name] = {
        "baseline_feature_count": len(feature_sets["alpha158_baseline"]),
        "candidate_feature_count": len(feature_sets[winner_name]),
        "candidate_feature_batch": winner_name,
        "candidate_weight": selected_blend_weight,
        "diagnostic": blend_summary,
        "causality_audit": {
            "strictly_mature": all(
                diagnostics[name]["causality_audit"]["strictly_mature"]
                for name in diagnostic_names
            ),
            "component_freeze_date": DIAGNOSTIC_FREEZE_DATE.isoformat(),
        },
    }
    diagnostic_results[blend_name] = blend_result

    baseline = diagnostics["alpha158_baseline"]["diagnostic"]
    candidate = diagnostics[blend_name]["diagnostic"]
    checks = promotion_checks(candidate, baseline)
    passed = selected_blend_weight > 0 and all(checks.values())

    curves = pd.concat(
        {
            name: normalized_curve(result, "strategy_return")
            for name, result in diagnostic_results.items()
        },
        axis=1,
        join="inner",
    )
    for name in list(curves.columns):
        curves[f"{name}_drawdown"] = curves[name].div(curves[name].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "candidate_passed_not_deployed" if passed else "candidate_rejected",
        "objective": "improve annual return, Sharpe and drawdown with unchanged execution",
        "feature_batch_names": list(FEATURE_BATCH_NAMES),
        "feature_counts": {name: len(features) for name, features in feature_sets.items()},
        "model_contract": {
            "model": "deterministic LightGBM LambdaRank",
            "model_config": WINNER_CONFIG,
            "selection_freeze_date": SELECTION_FREEZE_DATE.isoformat(),
            "selection_window": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "diagnostic_freeze_date": DIAGNOSTIC_FREEZE_DATE.isoformat(),
            "diagnostic_window": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
            "execution": "signal close; next-open; Top5; 10-session rebalance; 12bp one-way cost",
        },
        "selection_candidates": selection_rows,
        "selected_feature_batch": winner_name,
        "model_blend_weights": list(MODEL_BLEND_WEIGHTS),
        "blend_selection_candidates": blend_selection_rows,
        "selected_candidate_weight": selected_blend_weight,
        "diagnostics": diagnostics,
        "promotion_gate": {"passed": passed, "checks": checks},
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "universe_snapshot_run_id": snapshot_id,
        "production_change": False,
        "warnings": warnings + [
            "2020-2026 is already revealed and is only a deployment blocker.",
            "The current Top50 universe is backfilled and has survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "A passing historical gate still requires a new forward paper window.",
        ],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# 新因子 LGBM 策略 v1",
        "",
        f"- 状态：{'候选通过，尚未部署' if passed else '候选未通过，不部署'}",
        f"- 2018-2019 选择结构：`{winner_name}`",
        f"- 2018-2019 选择新因子模型权重：{selected_blend_weight:.0%}",
        "- 生产修改：否",
        "",
        "## 2018-2019 结构选择",
        "",
        "| 特征结构 | 特征数 | 年化 | 最大回撤 | 夏普 | 年化换手 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selection_rows:
        performance = row["selection"]["performance"]
        activity = row["selection"]["activity"]
        report.append(
            f"| `{row['name']}` | {row['feature_count']} | "
            f"{performance['annual_return']:.2%} | {performance['max_drawdown']:.2%} | "
            f"{performance['sharpe']:.3f} | {activity['annual_turnover']:.2f}x |"
        )
    report.extend([
        "",
        "## 2018-2019 模型分数融合",
        "",
        "| 新因子模型权重 | 年化 | 最大回撤 | 夏普 | 年化换手 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in blend_selection_rows:
        performance = row["selection"]["performance"]
        activity = row["selection"]["activity"]
        report.append(
            f"| {row['candidate_weight']:.0%} | {performance['annual_return']:.2%} | "
            f"{performance['max_drawdown']:.2%} | {performance['sharpe']:.3f} | "
            f"{activity['annual_turnover']:.2f}x |"
        )
    report.extend([
        "",
        "## 2020-2026 已揭盲否决检查",
        "",
        "| 策略 | 年化 | 最大回撤 | 夏普 | 年化换手 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, row in diagnostics.items():
        performance = row["diagnostic"]["performance"]
        activity = row["diagnostic"]["activity"]
        report.append(
            f"| `{name}` | {performance['annual_return']:.2%} | "
            f"{performance['max_drawdown']:.2%} | {performance['sharpe']:.3f} | "
            f"{activity['annual_turnover']:.2f}x |"
        )
    report.extend([
        "",
        "2020-2026 不参与特征结构选择或参数修改。只有全部晋级条件同时满足，候选才允许进入新的前向模拟盘观察。",
    ])
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "curve": str(CURVE_PATH),
        "selected_feature_batch": winner_name,
        "selected_candidate_weight": selected_blend_weight,
        "selection_candidates": selection_rows,
        "blend_selection_candidates": blend_selection_rows,
        "diagnostics": diagnostics,
        "promotion_gate": payload["promotion_gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
