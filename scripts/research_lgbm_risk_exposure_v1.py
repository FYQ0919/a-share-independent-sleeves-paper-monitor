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

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_START = date(2020, 1, 1)
DIAGNOSTIC_END = date(2026, 8, 25)

OUTPUT_PATH = BASE_DIR / "data" / "lgbm_risk_exposure_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_risk_exposure_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_risk_exposure_curves.csv"
EXPOSURE_PATH = BASE_DIR / "reports" / "lgbm_risk_exposure_diagnostics.csv"

RISK_WEIGHTS = {
    "risk_beta_60": 0.15,
    "risk_downside_beta_120": 0.22,
    "risk_residual_vol_60": 0.15,
    "risk_market_corr_60": 0.10,
    "risk_cvar_60": 0.15,
    "risk_drawdown_60": 0.13,
    "risk_gap_tail_20": 0.05,
    "risk_illiquidity_60": 0.05,
}

CANDIDATES = {
    "current_lgbm": {"stress_risk_weight": 0.00},
    "stress_risk25": {"stress_risk_weight": 0.25},
    "stress_risk50": {"stress_risk_weight": 0.50},
    "stress_risk75": {"stress_risk_weight": 0.75},
}


def validate_research_config() -> str:
    if not np.isclose(sum(RISK_WEIGHTS.values()), 1.0):
        raise RuntimeError("风险因子权重之和必须为1")
    candidate_weights = [row["stress_risk_weight"] for row in CANDIDATES.values()]
    if candidate_weights != [0.0, 0.25, 0.5, 0.75]:
        raise RuntimeError("风险覆盖候选必须保持冻结的小网格")
    canonical = json.dumps(
        {"risk_weights": RISK_WEIGHTS, "candidates": CANDIDATES},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _wide_to_rows(wide: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    stacked = wide.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(frame["date"]), frame["code"].astype(str)],
        names=stacked.index.names,
    )
    return stacked.reindex(keys).to_numpy(dtype=float)


def build_causal_risk_exposures(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    close = output.pivot(index="date", columns="code", values="close").sort_index()
    open_price = output.pivot(index="date", columns="code", values="open").reindex(close.index)
    amount = output.pivot(index="date", columns="code", values="amount").reindex(close.index)
    returns = close.pct_change(fill_method=None)
    market_return = returns.mean(axis=1, skipna=True)
    market_variance = market_return.rolling(60, min_periods=30).var(ddof=0)
    beta = returns.rolling(60, min_periods=30).cov(market_return).div(
        market_variance.replace(0, np.nan), axis=0
    )

    down_market = market_return.where(market_return.lt(0))
    down_returns = returns.where(market_return.lt(0), axis=0)
    down_variance = down_market.rolling(120, min_periods=20).var(ddof=0)
    downside_beta = down_returns.rolling(120, min_periods=20).cov(down_market).div(
        down_variance.replace(0, np.nan), axis=0
    )
    residual = returns.sub(beta.mul(market_return, axis=0))
    residual_volatility = residual.rolling(60, min_periods=30).std(ddof=0)
    market_correlation = returns.rolling(60, min_periods=30).corr(market_return)
    cvar = returns.rolling(60, min_periods=30).quantile(0.10).mul(-1)
    drawdown = close.div(close.rolling(60, min_periods=30).max()).sub(1).mul(-1)
    gap = open_price.div(close.shift(1)).sub(1)
    gap_tail = gap.rolling(20, min_periods=10).quantile(0.10).mul(-1)
    illiquidity = returns.abs().div(amount.replace(0, np.nan)).mul(1_000_000_000)
    illiquidity = illiquidity.rolling(60, min_periods=30).mean()

    risk_wide = {
        "risk_beta_60": beta,
        "risk_downside_beta_120": downside_beta,
        "risk_residual_vol_60": residual_volatility,
        "risk_market_corr_60": market_correlation,
        "risk_cvar_60": cvar,
        "risk_drawdown_60": drawdown,
        "risk_gap_tail_20": gap_tail,
        "risk_illiquidity_60": illiquidity,
    }
    for name, values in risk_wide.items():
        output[name] = _wide_to_rows(values, output)
        percentile = output[name].groupby(output["date"]).rank(pct=True)
        output[f"defensive_{name}"] = 1 - percentile
    output["defensive_score"] = sum(
        output[f"defensive_{name}"].mul(weight)
        for name, weight in RISK_WEIGHTS.items()
    )

    pool_equity = (1 + market_return.fillna(0.0)).cumprod()
    trend_120 = pool_equity.div(
        pool_equity.rolling(120, min_periods=60).mean()
    ).sub(1)
    volatility_20 = market_return.rolling(20, min_periods=10).std(ddof=0).mul(
        np.sqrt(252)
    )
    volatility_threshold = volatility_20.shift(1).rolling(
        252, min_periods=126
    ).quantile(0.75)
    breadth_20 = close.pct_change(20, fill_method=None).gt(0).mean(axis=1)
    regime = pd.DataFrame({
        "market_return": market_return,
        "trend_120": trend_120,
        "volatility_20": volatility_20,
        "volatility_threshold": volatility_threshold,
        "breadth_20": breadth_20,
    })
    regime["trend_stress"] = regime["trend_120"].lt(0)
    regime["volatility_stress"] = regime["volatility_20"].gt(
        regime["volatility_threshold"]
    )
    regime["breadth_stress"] = regime["breadth_20"].lt(0.40)
    regime["stress_votes"] = regime[
        ["trend_stress", "volatility_stress", "breadth_stress"]
    ].sum(axis=1)
    regime["market_stress"] = regime["stress_votes"].eq(3)
    output["market_stress"] = pd.to_datetime(output["date"]).map(
        regime["market_stress"]
    ).fillna(False).astype(bool)
    output["stress_votes"] = pd.to_datetime(output["date"]).map(
        regime["stress_votes"]
    ).fillna(0).astype(int)
    return output, regime


def blend_risk_scores(
    current_scores: pd.DataFrame,
    risk_frame: pd.DataFrame,
    stress_risk_weight: float,
) -> pd.DataFrame:
    output = current_scores.copy()
    stress = risk_frame["market_stress"].astype(float)
    weight = stress.mul(float(stress_risk_weight))
    defensive = risk_frame["defensive_score"].where(
        risk_frame["defensive_score"].notna(), current_scores["score"]
    )
    output["score"] = current_scores["score"].mul(1 - weight).add(
        defensive.mul(weight)
    )
    output.loc[current_scores["score"].isna(), "score"] = np.nan
    return output


def selection_folds(result: dict) -> dict:
    return {
        "2018": research_summary(result, date(2018, 1, 1), date(2018, 12, 31)),
        "2019": research_summary(result, date(2019, 1, 1), date(2019, 12, 31)),
    }


def selection_passes(candidate: dict, baseline: dict) -> bool:
    candidate_perf = candidate["selection"]["performance"]
    baseline_perf = baseline["selection"]["performance"]
    return bool(
        candidate["config"]["stress_risk_weight"] > 0
        and candidate_perf["annual_return"] >= baseline_perf["annual_return"] * 0.95
        and abs(candidate_perf["max_drawdown"]) <= abs(baseline_perf["max_drawdown"]) * 0.95
        and candidate_perf["sharpe"] >= baseline_perf["sharpe"]
        and candidate["selection_folds"]["2018"]["performance"]["max_drawdown"]
        >= baseline["selection_folds"]["2018"]["performance"]["max_drawdown"]
    )


def selection_rank(row: dict) -> tuple:
    performance = row["selection"]["performance"]
    calmar = performance["annual_return"] / abs(performance["max_drawdown"])
    return (
        calmar,
        performance["sharpe"],
        performance["annual_return"],
        performance["max_drawdown"],
    )


def portfolio_risk_summary(result: dict, risk_frame: pd.DataFrame) -> dict[str, float]:
    lookup = risk_frame.set_index(["date", "code"])
    rows = []
    for rebalance in result.get("rebalances", []):
        signal_date = pd.Timestamp(rebalance["signal_date"])
        for code in rebalance["holdings"]:
            key = (signal_date, str(code))
            if key not in lookup.index:
                continue
            item = lookup.loc[key]
            rows.append({name: float(item[name]) for name in RISK_WEIGHTS})
    exposure = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    if exposure.empty:
        return {name: 0.0 for name in RISK_WEIGHTS}
    return {name: float(exposure[name].mean()) for name in RISK_WEIGHTS}


def export_curves(risk_result: dict, baseline_result: dict) -> int:
    curves = pd.concat(
        {
            "risk_exposure_lgbm": normalized_curve(risk_result, "strategy_return"),
            "current_lgbm": normalized_curve(baseline_result, "strategy_return"),
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
    manifest = validate_research_config()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)

    selection_history = history[history["date"].le(pd.Timestamp(SELECTION_END))].copy()
    selection_frame, feature_columns = build_alpha158_lite(selection_history)
    selection_risk, selection_regime = build_causal_risk_exposures(selection_frame)
    selection_baseline = baseline_scores(
        selection_frame, SELECTION_START, SELECTION_END
    )
    selection_prediction, selection_audit = fit_expanding_lgbm_predictions(
        selection_frame,
        feature_columns,
        WINNER_CONFIG,
        SELECTION_START,
        SELECTION_END,
    )
    current_selection_scores = blend_scores(
        selection_frame,
        selection_prediction,
        selection_baseline,
        WINNER_CONFIG["baseline_weight"],
    )

    candidates = []
    selection_results = {}
    for name, config in CANDIDATES.items():
        scores = blend_risk_scores(
            current_selection_scores,
            selection_risk,
            config["stress_risk_weight"],
        )
        result = topk_dropout_backtest(
            selection_history, scores, SELECTION_START, SELECTION_END
        )
        selection_results[name] = result
        candidates.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
            "selection_folds": selection_folds(result),
        })
    baseline_row = next(row for row in candidates if row["name"] == "current_lgbm")
    eligible = [row for row in candidates if selection_passes(row, baseline_row)]
    winner = max(eligible, key=selection_rank) if eligible else baseline_row
    winner["selection_gate_passed"] = bool(eligible)

    diagnostic = None
    promotion_checks = {"selection_candidate_exists": bool(eligible)}
    curve_rows = 0
    if eligible:
        diagnostic_history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
        diagnostic_frame, diagnostic_features = build_alpha158_lite(diagnostic_history)
        if diagnostic_features != feature_columns:
            raise RuntimeError("选择期和诊断期特征模式不一致")
        diagnostic_risk, diagnostic_regime = build_causal_risk_exposures(diagnostic_frame)
        diagnostic_baseline = baseline_scores(
            diagnostic_frame, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        diagnostic_prediction, diagnostic_audit = fit_expanding_lgbm_predictions(
            diagnostic_frame,
            feature_columns,
            WINNER_CONFIG,
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
        )
        current_diagnostic_scores = blend_scores(
            diagnostic_frame,
            diagnostic_prediction,
            diagnostic_baseline,
            WINNER_CONFIG["baseline_weight"],
        )
        current_result = topk_dropout_backtest(
            diagnostic_history,
            current_diagnostic_scores,
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
        )
        risk_scores = blend_risk_scores(
            current_diagnostic_scores,
            diagnostic_risk,
            winner["config"]["stress_risk_weight"],
        )
        risk_result = topk_dropout_backtest(
            diagnostic_history, risk_scores, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        baseline_diagnostic = research_summary(
            current_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        risk_diagnostic = research_summary(
            risk_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        base_perf = baseline_diagnostic["performance"]
        risk_perf = risk_diagnostic["performance"]
        promotion_checks.update({
            "annual_return_retained_90pct": (
                risk_perf["annual_return"] >= base_perf["annual_return"] * 0.90
            ),
            "drawdown_improved_10pct": (
                abs(risk_perf["max_drawdown"]) <= abs(base_perf["max_drawdown"]) * 0.90
            ),
            "sharpe_not_lower": risk_perf["sharpe"] >= base_perf["sharpe"],
            "turnover_not_higher_10pct": (
                risk_diagnostic["activity"]["annual_turnover"]
                <= baseline_diagnostic["activity"]["annual_turnover"] * 1.10
            ),
        })
        curve_rows = export_curves(risk_result, current_result)
        diagnostic = {
            "current_lgbm": baseline_diagnostic,
            "risk_exposure_lgbm": risk_diagnostic,
            "current_exposures": portfolio_risk_summary(current_result, diagnostic_risk),
            "risk_controlled_exposures": portfolio_risk_summary(risk_result, diagnostic_risk),
            "stress_day_fraction": float(
                diagnostic_regime.loc[
                    pd.Timestamp(DIAGNOSTIC_START):pd.Timestamp(DIAGNOSTIC_END),
                    "market_stress",
                ].mean()
            ),
            "model_audit": {
                "refit_months": diagnostic_audit["refit_months"],
                "first": diagnostic_audit["maturity_audit"][0],
                "last": diagnostic_audit["maturity_audit"][-1],
            },
        }

    passed = bool(eligible) and all(promotion_checks.values())
    candidates.sort(key=selection_rank, reverse=True)
    selection_dates = selection_regime.loc[
        pd.Timestamp(SELECTION_START):pd.Timestamp(SELECTION_END)
    ].copy()
    selection_dates.to_csv(EXPOSURE_PATH, encoding="utf-8-sig", float_format="%.10f")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "candidate_passed_not_deployed" if passed else "research_only_not_promoted",
        "strategy": "LGBM Top5 with conditional stock-level risk exposure ranking",
        "windows": {
            "selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_diagnostic": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        },
        "execution_contract": "signal close; next-open execution; Top5; 10-session rebalance; 12bp one-way cost",
        "regime_rule": "stress only when all three hold: pool trend below MA120, vol20 above lagged rolling 75th percentile, and 20-day breadth below 40%",
        "risk_weights": RISK_WEIGHTS,
        "candidate_manifest_sha256": manifest,
        "universe_snapshot_run_id": snapshot_id,
        "winner": winner,
        "selection_stress_day_fraction": float(selection_dates["market_stress"].mean()),
        "selection_model_audit": {
            "refit_months": selection_audit["refit_months"],
            "first": selection_audit["maturity_audit"][0],
            "last": selection_audit["maturity_audit"][-1],
        },
        "diagnostic": diagnostic,
        "promotion_gate": {"passed": passed, "checks": promotion_checks},
        "candidates": candidates,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)) if curve_rows else None,
        "curve_rows": curve_rows,
        "regime_diagnostics": str(EXPOSURE_PATH.relative_to(BASE_DIR)),
        "production_change": False,
        "warnings": warnings + [
            "2020-2026 has already been revealed and is a historical diagnostic, not a fresh blind test.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Risk factors change stock selection but do not reduce total portfolio exposure or use leverage.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    selection_perf = winner["selection"]["performance"]
    report_lines = [
        "# LGBM 风险暴露选股研究 v1",
        "",
        f"- 状态：{'历史门禁通过，未部署' if passed else '未通过晋级门禁'}",
        f"- 选择冠军：`{winner['name']}`",
        f"- 压力期风险排名权重：{winner['config']['stress_risk_weight']:.0%}",
        f"- 选择期年化：{selection_perf['annual_return']:.2%}",
        f"- 选择期最大回撤：{selection_perf['max_drawdown']:.2%}",
        f"- 选择期夏普：{selection_perf['sharpe']:.3f}",
        f"- 选择期压力日占比：{payload['selection_stress_day_fraction']:.1%}",
        f"- 生产修改：否",
        "",
        "## 2018-2019选择期候选",
        "",
        "| 候选 | 压力期风险权重 | 年化 | 最大回撤 | 夏普 |",
        "|---|---:|---:|---:|---:|",
        *[
            "| {name} | {weight:.0%} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} |".format(
                name=row["name"],
                weight=row["config"]["stress_risk_weight"],
                annual=row["selection"]["performance"]["annual_return"],
                drawdown=row["selection"]["performance"]["max_drawdown"],
                sharpe=row["selection"]["performance"]["sharpe"],
            )
            for row in sorted(candidates, key=lambda item: item["config"]["stress_risk_weight"])
        ],
    ]
    if diagnostic:
        base_perf = diagnostic["current_lgbm"]["performance"]
        risk_perf = diagnostic["risk_exposure_lgbm"]["performance"]
        report_lines.extend([
            "",
            "## 2020-2026历史诊断",
            "",
            f"- 年化：{base_perf['annual_return']:.2%} -> {risk_perf['annual_return']:.2%}",
            f"- 最大回撤：{base_perf['max_drawdown']:.2%} -> {risk_perf['max_drawdown']:.2%}",
            f"- 夏普：{base_perf['sharpe']:.3f} -> {risk_perf['sharpe']:.3f}",
            f"- 压力日占比：{diagnostic['stress_day_fraction']:.1%}",
        ])
    report_lines.extend([
        "",
        "所有风险暴露只使用信号日及以前数据；下一交易日开盘执行。当前结果不自动覆盖冻结模型或模拟盘。",
    ])
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
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
