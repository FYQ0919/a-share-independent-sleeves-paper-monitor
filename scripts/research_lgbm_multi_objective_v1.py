from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.backtest_engine import BacktestEngine
from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG, eligible_training_mask
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_dynamic_k_cash_v1 import adjusted_holdings
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
)


SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_START = date(2020, 1, 1)
DIAGNOSTIC_END = date(2026, 8, 25)
COST_BPS = 12.0
ROUND_TRIP_COST = COST_BPS * 2 / 10_000
RETURN_HURDLE = 0.003
LARGE_LOSS_THRESHOLD = -0.06
RANDOM_SEED = 20260828

OUTPUT_PATH = BASE_DIR / "data" / "lgbm_multi_objective_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_multi_objective_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_multi_objective_curves.csv"
REBALANCE_PATH = BASE_DIR / "reports" / "lgbm_multi_objective_rebalances.csv"

# Frozen before the revealed 2020-2026 diagnostic. The grid changes only the
# portfolio use of two auxiliary heads; all model and label parameters are fixed.
CANDIDATES = {
    "current_rank_top5": {
        "return_weight": 0.0,
        "safety_weight": 0.0,
        "use_hurdle": False,
        "max_k": 5,
        "min_k": 5,
        "risk_budget": None,
        "minimum_exposure": 1.0,
    },
    "multihead_pareto_top5": {
        "return_weight": 0.025,
        "safety_weight": 0.05,
        "use_hurdle": False,
        "max_k": 5,
        "min_k": 5,
        "risk_budget": None,
        "minimum_exposure": 1.0,
    },
    "multihead_soft_top5": {
        "return_weight": 0.15,
        "safety_weight": 0.10,
        "use_hurdle": False,
        "max_k": 5,
        "min_k": 5,
        "risk_budget": None,
        "minimum_exposure": 1.0,
    },
    "multihead_opportunity": {
        "return_weight": 0.15,
        "safety_weight": 0.10,
        "use_hurdle": True,
        "max_k": 8,
        "min_k": 3,
        "risk_budget": None,
        "minimum_exposure": 1.0,
    },
    "multihead_risk_budget": {
        "return_weight": 0.15,
        "safety_weight": 0.15,
        "use_hurdle": True,
        "max_k": 8,
        "min_k": 3,
        "risk_budget": 0.06,
        "minimum_exposure": 0.65,
    },
}


def validate_candidates() -> str:
    expected = {
        "return_weight", "safety_weight", "use_hurdle", "max_k", "min_k",
        "risk_budget", "minimum_exposure",
    }
    for name, config in CANDIDATES.items():
        if set(config) != expected:
            raise RuntimeError(f"{name} 多目标配置不完整")
        if config["return_weight"] + config["safety_weight"] >= 1:
            raise RuntimeError(f"{name} 辅助头权重过高")
        if not 0 < config["min_k"] <= config["max_k"]:
            raise RuntimeError(f"{name} 持股数量范围无效")
        if not 0 < config["minimum_exposure"] <= 1:
            raise RuntimeError(f"{name} 最低仓位无效")
    if list(CANDIDATES) != [
        "current_rank_top5", "multihead_pareto_top5", "multihead_soft_top5",
        "multihead_opportunity", "multihead_risk_budget",
    ]:
        raise RuntimeError("多目标候选顺序已改变")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def auxiliary_lgbm_params(objective: str) -> dict:
    params = {
        "objective": objective,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.70,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "seed": RANDOM_SEED,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    params["metric"] = "l1" if objective == "regression_l1" else "binary_logloss"
    return params


def fit_expanding_auxiliary_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    prediction_start: date,
    prediction_end: date,
) -> tuple[pd.DataFrame, dict]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("多目标策略需要 lightgbm==4.6.0") from exc

    output = pd.DataFrame(
        {"expected_return": np.nan, "downside_probability": np.nan},
        index=frame.index,
    )
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dropna().unique()))
    prediction_dates = dates[
        (dates >= pd.Timestamp(prediction_start)) & (dates <= pd.Timestamp(prediction_end))
    ]
    quarters = pd.Series(prediction_dates, index=prediction_dates).groupby(
        [prediction_dates.year, prediction_dates.quarter]
    )
    audits = []
    for (year, quarter), quarter_dates in quarters:
        first_prediction = pd.Timestamp(quarter_dates.iloc[0])
        train = frame.loc[eligible_training_mask(frame, first_prediction)].copy()
        if train["date"].nunique() < 252 or len(train) < 5_000:
            continue
        matrix = train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        returns = train["label_return"].clip(-0.20, 0.20).to_numpy(dtype=np.float32)
        downside = train["label_return"].le(LARGE_LOSS_THRESHOLD).astype(np.int32).to_numpy()
        return_set = lgb.Dataset(
            matrix,
            label=returns,
            feature_name=feature_columns,
            free_raw_data=False,
        )
        downside_set = lgb.Dataset(
            matrix,
            label=downside,
            feature_name=feature_columns,
            free_raw_data=False,
        )
        return_model = lgb.train(
            auxiliary_lgbm_params("regression_l1"), return_set, num_boost_round=100
        )
        downside_model = lgb.train(
            auxiliary_lgbm_params("binary"), downside_set, num_boost_round=100
        )
        quarter_index = pd.DatetimeIndex(quarter_dates.to_numpy())
        mask = frame["date"].isin(quarter_index)
        predict_matrix = frame.loc[mask, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        output.loc[mask, "expected_return"] = return_model.predict(predict_matrix)
        output.loc[mask, "downside_probability"] = downside_model.predict(predict_matrix)
        max_label_end = pd.to_datetime(train["label_end_date"], errors="coerce").max()
        audits.append({
            "quarter": f"{year:04d}-Q{quarter}",
            "prediction_start": first_prediction.date().isoformat(),
            "max_training_label_end_date": max_label_end.date().isoformat(),
            "strictly_mature": bool(max_label_end < first_prediction),
            "training_rows": int(len(train)),
            "training_dates": int(train["date"].nunique()),
            "large_loss_rate": float(downside.mean()),
        })
    return output, {
        "refit_quarters": len(audits),
        "return_objective": "regression_l1 on clipped absolute 10-session return",
        "downside_objective": f"binary probability label_return <= {LARGE_LOSS_THRESHOLD:.2%}",
        "maturity_audit": audits,
    }


def build_multihead_scores(
    current_scores: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    output = current_scores.copy()
    expected = predictions["expected_return"].reindex(output.index)
    downside = predictions["downside_probability"].reindex(output.index)
    expected_rank = expected.groupby(output["date"]).rank(pct=True)
    safety_rank = downside.groupby(output["date"]).rank(pct=True, ascending=False)
    auxiliary_weight = config["return_weight"] + config["safety_weight"]
    output["score"] = (
        output["score"].mul(1 - auxiliary_weight)
        + expected_rank.mul(config["return_weight"])
        + safety_rank.mul(config["safety_weight"])
    )
    output["expected_return"] = expected
    output["downside_probability"] = downside
    output.loc[expected.isna() | downside.isna(), "score"] = np.nan
    return output


def opportunity_target(
    cross: pd.DataFrame,
    planned_holdings: list[str],
    config: dict,
    n_drop: int = 1,
) -> tuple[list[str], list[str], float]:
    ranked = cross.dropna(subset=["score"]).sort_values("score", ascending=False)
    if config["use_hurdle"]:
        ranked = ranked[
            ranked["expected_return"].gt(RETURN_HURDLE)
            & ranked["downside_probability"].lt(0.10)
        ]
    target_k = min(int(config["max_k"]), len(ranked))
    if target_k < int(config["min_k"]):
        return [], list(planned_holdings), 0.0

    score = ranked.set_index("code")["score"]
    mandatory_drops = [code for code in planned_holdings if code not in score.index]
    surviving = [code for code in planned_holdings if code in score.index]
    target, optional_drops = adjusted_holdings(score, surviving, target_k, n_drop=n_drop)
    dropped = mandatory_drops + [code for code in optional_drops if code not in mandatory_drops]
    selected = ranked.set_index("code").reindex(target)
    risk_budget = config["risk_budget"]
    if risk_budget is None or selected.empty:
        exposure = 1.0
    else:
        predicted_risk = float(selected["downside_probability"].mean())
        exposure = float(np.clip(
            float(risk_budget) / max(predicted_risk, 1e-6),
            float(config["minimum_exposure"]),
            1.0,
        ))
    return target, dropped, exposure


def multihead_portfolio_backtest(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict,
    start: date,
    end: date,
    rebalance_days: int = 10,
    n_drop: int = 1,
    cost_bps: float = COST_BPS,
) -> dict:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    open_prices = data.pivot(index="date", columns="code", values="open").reindex(dates)
    period_returns = open_prices.shift(-1).div(open_prices).sub(1).replace([np.inf, -np.inf], np.nan)
    benchmark_returns = period_returns.mean(axis=1, skipna=True).fillna(0.0)
    score_lookup = scores.set_index(["date", "code"])[
        ["score", "expected_return", "downside_probability"]
    ]

    pending_targets: dict[int, dict] = {}
    planned_holdings: list[str] = []
    targets = pd.DataFrame(index=dates, columns=open_prices.columns, dtype=float)
    records = []
    for signal_index, signal_date in enumerate(dates):
        if signal_index in pending_targets:
            planned_holdings = list(pending_targets.pop(signal_index)["holdings"])
        if signal_index % rebalance_days != 0:
            continue
        execution_index = signal_index + 1
        if execution_index >= len(dates) - 1:
            break
        try:
            cross = score_lookup.loc[signal_date].dropna(subset=["score"]).reset_index()
        except KeyError:
            continue
        cross = cross[cross["code"].isin(open_prices.columns)]
        holdings, dropped, exposure = opportunity_target(
            cross, planned_holdings, config, n_drop=n_drop
        )
        execution_date = dates[execution_index]
        targets.loc[execution_date, :] = 0.0
        if holdings and exposure > 0:
            targets.loc[execution_date, holdings] = exposure / len(holdings)
        pending_targets[execution_index] = {"holdings": holdings}
        selected = cross.set_index("code").reindex(holdings)
        records.append({
            "signal_date": signal_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "target_k": len(holdings),
            "target_exposure": exposure,
            "holdings": holdings,
            "dropped": dropped,
            "mean_expected_return": float(selected["expected_return"].mean()) if holdings else None,
            "mean_downside_probability": float(selected["downside_probability"].mean()) if holdings else None,
        })

    current_weights = pd.Series(0.0, index=open_prices.columns)
    daily_returns = pd.Series(0.0, index=dates)
    turnovers = pd.Series(0.0, index=dates)
    exposures = pd.Series(0.0, index=dates)
    for trade_date in dates:
        target = targets.loc[trade_date]
        if target.notna().any():
            target = target.fillna(0.0)
            turnovers.loc[trade_date] = target.sub(current_weights).abs().sum()
            current_weights = target
        exposures.loc[trade_date] = float(current_weights.sum())
        returns_today = period_returns.loc[trade_date].fillna(0.0)
        gross_return = float(current_weights.mul(returns_today).sum())
        daily_returns.loc[trade_date] = gross_return
        growth = 1 + gross_return
        if current_weights.sum() > 0 and growth > 0:
            current_weights = current_weights.mul(1 + returns_today).div(growth)

    costs = turnovers * cost_bps / 10_000
    strategy_returns = daily_returns - costs
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    metrics = BacktestEngine._metrics(
        strategy_returns, benchmark_returns, strategy_equity,
        benchmark_equity, turnovers, costs,
    )
    return_periods = [
        {
            "period_start": dates[index].date().isoformat(),
            "period_end": dates[index + 1].date().isoformat(),
            "strategy_return": float(strategy_returns.iloc[index]),
            "benchmark_return": float(benchmark_returns.iloc[index]),
            "turnover": float(turnovers.iloc[index]),
            "cost": float(costs.iloc[index]),
        }
        for index in range(len(dates) - 1)
    ]
    return {
        "metrics": metrics,
        "return_periods": return_periods,
        "rebalances": records,
        "exposure": exposures,
        "config": {**config, "rebalance_days": rebalance_days, "cost_bps": cost_bps},
    }


def exposure_summary(result: dict) -> dict:
    exposure = result["exposure"]
    records = result["rebalances"]
    return {
        "average_exposure": float(exposure.mean()),
        "minimum_exposure": float(exposure.min()),
        "cash_days": int(exposure.lt(0.01).sum()),
        "average_target_k": float(np.mean([row["target_k"] for row in records])) if records else 0.0,
        "average_selected_downside_probability": float(np.nanmean([
            row["mean_downside_probability"] for row in records
            if row["mean_downside_probability"] is not None
        ])) if records else 0.0,
    }


def candidate_passes(row: dict, baseline: dict) -> bool:
    performance = row["selection"]["performance"]
    base = baseline["selection"]["performance"]
    return bool(
        row["name"] != "current_rank_top5"
        and performance["annual_return"] >= base["annual_return"] * 0.99
        and abs(performance["max_drawdown"]) <= abs(base["max_drawdown"]) * 0.97
        and performance["sharpe"] >= base["sharpe"]
    )


def candidate_rank(row: dict) -> tuple:
    performance = row["selection"]["performance"]
    calmar = performance["annual_return"] / max(abs(performance["max_drawdown"]), 1e-9)
    return (
        calmar,
        performance["sharpe"],
        performance["annual_return"],
        -row["selection"]["activity"]["annual_turnover"],
    )


def run_predictions(history: pd.DataFrame, start: date, end: date):
    frame, features = build_alpha158_lite(history)
    baseline = baseline_scores(frame, start, end)
    rank_prediction, rank_audit = fit_expanding_lgbm_predictions(
        frame, features, WINNER_CONFIG, start, end
    )
    current_scores = blend_scores(
        frame, rank_prediction, baseline, WINNER_CONFIG["baseline_weight"]
    )
    auxiliary, auxiliary_audit = fit_expanding_auxiliary_predictions(
        frame, features, start, end
    )
    return current_scores, auxiliary, rank_audit, auxiliary_audit


def export_curves(winner_result: dict, baseline_result: dict) -> int:
    curves = pd.concat(
        {
            "multi_objective_lgbm": normalized_curve(winner_result, "strategy_return"),
            "current_lgbm_top5": normalized_curve(baseline_result, "strategy_return"),
            "pool_benchmark": normalized_curve(baseline_result, "benchmark_return"),
        },
        axis=1,
        join="inner",
    )
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")
    return len(curves)


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    selection_history = history[history["date"].le(pd.Timestamp(SELECTION_END))].copy()
    current_scores, auxiliary, rank_audit, auxiliary_audit = run_predictions(
        selection_history, SELECTION_START, SELECTION_END
    )

    candidates = []
    selection_results = {}
    for name, config in CANDIDATES.items():
        scores = build_multihead_scores(current_scores, auxiliary, config)
        result = multihead_portfolio_backtest(
            selection_history, scores, config, SELECTION_START, SELECTION_END
        )
        selection_results[name] = result
        candidates.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
            "exposure": exposure_summary(result),
        })
    baseline = next(row for row in candidates if row["name"] == "current_rank_top5")
    eligible = [row for row in candidates if candidate_passes(row, baseline)]
    winner = max(eligible, key=candidate_rank) if eligible else baseline
    winner["selection_gate_passed"] = bool(eligible)

    diagnostic = None
    diagnostic_checks = {"selection_candidate_exists": bool(eligible)}
    curve_rows = 0
    if eligible:
        diagnostic_history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
        current_scores, auxiliary, diagnostic_rank_audit, diagnostic_aux_audit = run_predictions(
            diagnostic_history, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        baseline_scores_frame = build_multihead_scores(
            current_scores, auxiliary, CANDIDATES["current_rank_top5"]
        )
        winner_scores = build_multihead_scores(current_scores, auxiliary, winner["config"])
        baseline_result = multihead_portfolio_backtest(
            diagnostic_history, baseline_scores_frame,
            CANDIDATES["current_rank_top5"], DIAGNOSTIC_START, DIAGNOSTIC_END,
        )
        winner_result = multihead_portfolio_backtest(
            diagnostic_history, winner_scores, winner["config"],
            DIAGNOSTIC_START, DIAGNOSTIC_END,
        )
        baseline_summary = research_summary(
            baseline_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        winner_summary = research_summary(
            winner_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        base_perf = baseline_summary["performance"]
        winner_perf = winner_summary["performance"]
        diagnostic_checks.update({
            "annual_return_retained_95pct": winner_perf["annual_return"] >= base_perf["annual_return"] * 0.95,
            "drawdown_improved_3pct": abs(winner_perf["max_drawdown"]) <= abs(base_perf["max_drawdown"]) * 0.97,
            "sharpe_not_lower": winner_perf["sharpe"] >= base_perf["sharpe"],
            "turnover_not_higher_15pct": winner_summary["activity"]["annual_turnover"] <= baseline_summary["activity"]["annual_turnover"] * 1.15,
        })
        curve_rows = export_curves(winner_result, baseline_result)
        pd.DataFrame(winner_result["rebalances"]).to_csv(
            REBALANCE_PATH, index=False, encoding="utf-8-sig"
        )
        diagnostic = {
            "current_lgbm_top5": baseline_summary,
            "multi_objective_lgbm": winner_summary,
            "current_exposure": exposure_summary(baseline_result),
            "multi_objective_exposure": exposure_summary(winner_result),
            "rank_audit": {
                "refit_months": diagnostic_rank_audit["refit_months"],
                "first": diagnostic_rank_audit["maturity_audit"][0],
                "last": diagnostic_rank_audit["maturity_audit"][-1],
            },
            "auxiliary_audit": {
                "refit_quarters": diagnostic_aux_audit["refit_quarters"],
                "first": diagnostic_aux_audit["maturity_audit"][0],
                "last": diagnostic_aux_audit["maturity_audit"][-1],
            },
        }

    passed = bool(eligible) and all(diagnostic_checks.values())
    candidates.sort(key=candidate_rank, reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "candidate_passed_not_deployed" if passed else "research_only_not_promoted",
        "strategy": "three-head LGBM: relative rank + absolute return + large-loss probability",
        "objective": "retain annual return while reducing maximum drawdown and preserving Sharpe",
        "windows": {
            "selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_diagnostic": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        },
        "labels": {
            "rank": "next-open to t+11-open cross-sectional relevance",
            "absolute_return": "clipped next-open to t+11-open return",
            "large_loss": f"probability return <= {LARGE_LOSS_THRESHOLD:.2%}",
        },
        "fixed_thresholds": {
            "round_trip_cost": ROUND_TRIP_COST,
            "absolute_return_hurdle": RETURN_HURDLE,
            "maximum_large_loss_probability": 0.10,
        },
        "execution_contract": "signal close; next-open execution; K=0..8; 10-session rebalance; 12bp one-way cost; cash earns zero",
        "candidate_manifest_sha256": manifest,
        "universe_snapshot_run_id": snapshot_id,
        "winner": winner,
        "selection_model_audit": {
            "rank_refits": rank_audit["refit_months"],
            "auxiliary_refits": auxiliary_audit["refit_quarters"],
            "rank_first": rank_audit["maturity_audit"][0],
            "rank_last": rank_audit["maturity_audit"][-1],
            "auxiliary_first": auxiliary_audit["maturity_audit"][0],
            "auxiliary_last": auxiliary_audit["maturity_audit"][-1],
        },
        "diagnostic": diagnostic,
        "promotion_gate": {"passed": passed, "checks": diagnostic_checks},
        "candidates": candidates,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)) if curve_rows else None,
        "rebalance_data": str(REBALANCE_PATH.relative_to(BASE_DIR)) if curve_rows else None,
        "production_change": False,
        "warnings": warnings + [
            "2020-2026 is already revealed and is used only as a deployment blocker.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "Auxiliary probabilities are historical model estimates, not guaranteed probabilities.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# LGBM 年化与回撤多目标策略 v1",
        "",
        f"- 状态：{'候选通过，未部署' if passed else '未通过晋级门禁'}",
        f"- 选择冠军：`{winner['name']}`",
        f"- 选择期年化：{winner['selection']['performance']['annual_return']:.2%}",
        f"- 选择期最大回撤：{winner['selection']['performance']['max_drawdown']:.2%}",
        f"- 选择期夏普：{winner['selection']['performance']['sharpe']:.3f}",
        f"- 选择期平均仓位：{winner['exposure']['average_exposure']:.1%}",
        "- 生产修改：否",
        "",
        "## 2018-2019 选择结果",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | Calmar | 平均仓位 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        perf = row["selection"]["performance"]
        report_lines.append(
            f"| `{row['name']}` | {perf['annual_return']:.2%} | {perf['max_drawdown']:.2%} | "
            f"{perf['sharpe']:.3f} | {perf['annual_return']/abs(perf['max_drawdown']):.3f} | "
            f"{row['exposure']['average_exposure']:.1%} |"
        )
    if diagnostic:
        report_lines.extend(["", "## 2020-2026 已揭盲部署检查", ""])
        for key, label in (("current_lgbm_top5", "当前LGBM"), ("multi_objective_lgbm", "多目标LGBM")):
            perf = diagnostic[key]["performance"]
            report_lines.append(
                f"- {label}：年化 {perf['annual_return']:.2%}，最大回撤 {perf['max_drawdown']:.2%}，夏普 {perf['sharpe']:.3f}。"
            )
    report_lines.extend([
        "",
        "2020-2026 不参与候选排序或阈值修改。当前结果仍受历史股票池回填和复权时点偏差影响。",
    ])
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "winner": winner,
        "diagnostic": diagnostic,
        "promotion_gate": payload["promotion_gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
