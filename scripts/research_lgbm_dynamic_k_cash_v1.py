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
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_risk_exposure_v1 import build_causal_risk_exposures
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

OUTPUT_PATH = BASE_DIR / "data" / "lgbm_dynamic_k_cash_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_dynamic_k_cash_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_dynamic_k_cash_curves.csv"
REBALANCE_PATH = BASE_DIR / "reports" / "lgbm_dynamic_k_cash_rebalances.csv"

CANDIDATES = {
    "current_top5": {
        "normal_k": 5, "stress_k": 5,
        "normal_exposure": 1.00, "stress_exposure": 1.00,
        "risk_rebalance": False,
    },
    "always_top10": {
        "normal_k": 10, "stress_k": 10,
        "normal_exposure": 1.00, "stress_exposure": 1.00,
        "risk_rebalance": False,
    },
    "stress_top10_full": {
        "normal_k": 5, "stress_k": 10,
        "normal_exposure": 1.00, "stress_exposure": 1.00,
        "risk_rebalance": False,
    },
    "stress_top10_cash75": {
        "normal_k": 5, "stress_k": 10,
        "normal_exposure": 1.00, "stress_exposure": 0.75,
        "risk_rebalance": False,
    },
    "stress_top15_cash70": {
        "normal_k": 5, "stress_k": 15,
        "normal_exposure": 1.00, "stress_exposure": 0.70,
        "risk_rebalance": False,
    },
    "stress_full_cash": {
        "normal_k": 5, "stress_k": 0,
        "normal_exposure": 1.00, "stress_exposure": 0.00,
        "risk_rebalance": False,
    },
}


def validate_candidates() -> str:
    for name, config in CANDIDATES.items():
        if config["normal_k"] < 0 or config["stress_k"] < 0:
            raise RuntimeError(f"{name} 持股数量不能为负")
        if not 0 <= config["normal_exposure"] <= 1:
            raise RuntimeError(f"{name} 正常仓位越界")
        if not 0 <= config["stress_exposure"] <= 1:
            raise RuntimeError(f"{name} 压力仓位越界")
    if list(CANDIDATES) != [
        "current_top5", "always_top10", "stress_top10_full",
        "stress_top10_cash75", "stress_top15_cash70", "stress_full_cash",
    ]:
        raise RuntimeError("动态持股候选必须保持冻结顺序")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adjusted_holdings(
    cross: pd.Series,
    planned_holdings: list[str],
    target_k: int,
    n_drop: int = 1,
) -> tuple[list[str], list[str]]:
    if target_k <= 0:
        return [], list(planned_holdings)
    ranked_codes = cross.index.astype(str).tolist()
    if not planned_holdings:
        return ranked_codes[:target_k], []
    available = [code for code in planned_holdings if code in cross.index]
    missing = [code for code in planned_holdings if code not in cross.index]
    if target_k > len(planned_holdings):
        target = list(available)
        target.extend(code for code in ranked_codes if code not in target)
        return target[:target_k], missing
    if target_k < len(planned_holdings):
        retained = sorted(available, key=lambda code: float(cross.loc[code]), reverse=True)
        target = retained[:target_k]
        target.extend(code for code in ranked_codes if code not in target)
        target = target[:target_k]
        return target, [code for code in planned_holdings if code not in target]

    ranked_holdings = sorted(available, key=lambda code: float(cross.loc[code]))
    challengers = [code for code in ranked_codes if code not in planned_holdings]
    drop_count = min(n_drop, len(ranked_holdings) + len(missing), len(challengers))
    dropped = missing[:drop_count]
    dropped.extend(ranked_holdings[: max(0, drop_count - len(dropped))])
    target = [code for code in planned_holdings if code not in dropped]
    target.extend(challengers[:drop_count])
    if len(target) < target_k:
        target.extend(code for code in ranked_codes if code not in target)
    return target[:target_k], dropped


def dynamic_portfolio_backtest(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    regime: pd.DataFrame,
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
    period_returns = open_prices.shift(-1).div(open_prices).sub(1).replace(
        [np.inf, -np.inf], np.nan
    )
    if "universe_member" in data.columns:
        membership = data.pivot(
            index="date", columns="code", values="universe_member"
        ).reindex(index=dates, columns=open_prices.columns).fillna(False).astype(bool)
        period_returns = period_returns.where(membership)
    benchmark_returns = period_returns.mean(axis=1, skipna=True).fillna(0.0)
    score_lookup = scores.set_index(["date", "code"])["score"]
    stress_lookup = regime["market_stress"].reindex(dates).fillna(False).astype(bool)

    pending_targets: dict[int, dict] = {}
    planned_holdings: list[str] = []
    planned_exposure = 0.0
    targets = pd.DataFrame(index=dates, columns=open_prices.columns, dtype=float)
    records = []
    last_signal_stress = None
    for signal_index, signal_date in enumerate(dates):
        if signal_index in pending_targets:
            materialized = pending_targets.pop(signal_index)
            planned_holdings = list(materialized["holdings"])
            planned_exposure = float(materialized["exposure"])
        stress = bool(stress_lookup.loc[signal_date])
        scheduled = signal_index % rebalance_days == 0
        regime_changed = last_signal_stress is not None and stress != last_signal_stress
        risk_trigger = bool(config["risk_rebalance"] and regime_changed)
        if not scheduled and not risk_trigger:
            last_signal_stress = stress
            continue
        execution_index = signal_index + 1
        if execution_index >= len(dates) - 1:
            break
        target_k = int(config["stress_k"] if stress else config["normal_k"])
        target_exposure = float(
            config["stress_exposure"] if stress else config["normal_exposure"]
        )
        try:
            cross = score_lookup.loc[signal_date].dropna().sort_values(ascending=False)
        except KeyError:
            last_signal_stress = stress
            continue
        cross = cross[cross.index.isin(open_prices.columns)]
        if target_k > 0 and len(cross) < target_k:
            last_signal_stress = stress
            continue
        target_codes, dropped = adjusted_holdings(
            cross, planned_holdings, target_k, n_drop=n_drop
        )
        execution_date = dates[execution_index]
        targets.loc[execution_date, :] = 0.0
        if target_codes and target_exposure > 0:
            targets.loc[execution_date, target_codes] = target_exposure / len(target_codes)
        pending_targets[execution_index] = {
            "holdings": target_codes,
            "exposure": target_exposure,
        }
        records.append({
            "signal_date": signal_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "trigger": "risk_transition" if risk_trigger and not scheduled else "scheduled",
            "market_stress": stress,
            "target_k": target_k,
            "target_exposure": target_exposure,
            "holdings": target_codes,
            "dropped": dropped,
            "previous_exposure": planned_exposure,
        })
        last_signal_stress = stress

    current_weights = pd.Series(0.0, index=open_prices.columns)
    daily_returns = pd.Series(0.0, index=dates)
    turnovers = pd.Series(0.0, index=dates)
    active = pd.Series(False, index=dates)
    exposures = pd.Series(0.0, index=dates)
    for trade_date in dates:
        target = targets.loc[trade_date]
        if target.notna().any():
            target = target.fillna(0.0)
            turnovers.loc[trade_date] = target.sub(current_weights).abs().sum()
            current_weights = target
        exposures.loc[trade_date] = float(current_weights.sum())
        active.loc[trade_date] = current_weights.sum() > 0
        returns_today = period_returns.loc[trade_date].fillna(0.0)
        gross_return = float(current_weights.mul(returns_today).sum())
        daily_returns.loc[trade_date] = gross_return
        growth = 1 + gross_return
        if current_weights.sum() > 0 and growth > 0:
            current_weights = current_weights.mul(1 + returns_today).div(growth)

    costs = turnovers * cost_bps / 10_000
    strategy_returns = (daily_returns - costs).where(active | costs.gt(0), 0.0)
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    metrics = BacktestEngine._metrics(
        strategy_returns,
        benchmark_returns,
        strategy_equity,
        benchmark_equity,
        turnovers,
        costs,
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
        "config": {
            **config,
            "rebalance_days": rebalance_days,
            "n_drop": n_drop,
            "cost_bps": cost_bps,
        },
    }


def candidate_passes(row: dict, baseline: dict) -> bool:
    performance = row["selection"]["performance"]
    base = baseline["selection"]["performance"]
    return bool(
        row["name"] != "current_top5"
        and performance["annual_return"] >= base["annual_return"] * 0.90
        and abs(performance["max_drawdown"]) <= abs(base["max_drawdown"]) * 0.90
        and performance["sharpe"] >= base["sharpe"]
    )


def candidate_rank(row: dict) -> tuple:
    performance = row["selection"]["performance"]
    calmar = performance["annual_return"] / abs(performance["max_drawdown"])
    return (
        calmar,
        performance["sharpe"],
        performance["annual_return"],
        -row["selection"]["activity"]["annual_turnover"],
    )


def result_exposure_summary(result: dict) -> dict:
    exposure = result["exposure"]
    records = result["rebalances"]
    return {
        "average_exposure": float(exposure.mean()),
        "minimum_exposure": float(exposure.min()),
        "cash_days": int(exposure.lt(0.01).sum()),
        "risk_transition_count": sum(
            row["trigger"] == "risk_transition" for row in records
        ),
        "average_target_k": float(np.mean([row["target_k"] for row in records])) if records else 0.0,
    }


def export_curves(winner_result: dict, baseline_result: dict) -> int:
    curves = pd.concat(
        {
            "dynamic_k_cash": normalized_curve(winner_result, "strategy_return"),
            "current_top5": normalized_curve(baseline_result, "strategy_return"),
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


def run_scores(history: pd.DataFrame, start: date, end: date):
    frame, feature_columns = build_alpha158_lite(history)
    risk_frame, regime = build_causal_risk_exposures(frame)
    baseline = baseline_scores(frame, start, end)
    prediction, audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, start, end
    )
    scores = blend_scores(
        frame, prediction, baseline, WINNER_CONFIG["baseline_weight"]
    )
    return scores, regime, audit


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    selection_history = history[history["date"].le(pd.Timestamp(SELECTION_END))].copy()
    selection_scores, selection_regime, selection_audit = run_scores(
        selection_history, SELECTION_START, SELECTION_END
    )

    candidates = []
    selection_results = {}
    for name, config in CANDIDATES.items():
        result = dynamic_portfolio_backtest(
            selection_history,
            selection_scores,
            selection_regime,
            config,
            SELECTION_START,
            SELECTION_END,
        )
        selection_results[name] = result
        candidates.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
            "exposure": result_exposure_summary(result),
        })
    baseline_row = next(row for row in candidates if row["name"] == "current_top5")
    eligible = [row for row in candidates if candidate_passes(row, baseline_row)]
    winner = max(eligible, key=candidate_rank) if eligible else baseline_row
    winner["selection_gate_passed"] = bool(eligible)

    diagnostic = None
    checks = {"selection_candidate_exists": bool(eligible)}
    curve_rows = 0
    if eligible:
        diagnostic_history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
        diagnostic_scores, diagnostic_regime, diagnostic_audit = run_scores(
            diagnostic_history, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        baseline_result = dynamic_portfolio_backtest(
            diagnostic_history,
            diagnostic_scores,
            diagnostic_regime,
            CANDIDATES["current_top5"],
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
        )
        winner_result = dynamic_portfolio_backtest(
            diagnostic_history,
            diagnostic_scores,
            diagnostic_regime,
            winner["config"],
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
        )
        baseline_summary = research_summary(
            baseline_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        winner_summary = research_summary(
            winner_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        base_perf = baseline_summary["performance"]
        winner_perf = winner_summary["performance"]
        checks.update({
            "annual_return_retained_85pct": (
                winner_perf["annual_return"] >= base_perf["annual_return"] * 0.85
            ),
            "drawdown_improved_15pct": (
                abs(winner_perf["max_drawdown"]) <= abs(base_perf["max_drawdown"]) * 0.85
            ),
            "sharpe_not_lower": winner_perf["sharpe"] >= base_perf["sharpe"],
            "turnover_not_higher_20pct": (
                winner_summary["activity"]["annual_turnover"]
                <= baseline_summary["activity"]["annual_turnover"] * 1.20
            ),
        })
        curve_rows = export_curves(winner_result, baseline_result)
        pd.DataFrame(winner_result["rebalances"]).to_csv(
            REBALANCE_PATH, index=False, encoding="utf-8-sig"
        )
        diagnostic = {
            "current_top5": baseline_summary,
            "dynamic_k_cash": winner_summary,
            "current_exposure": result_exposure_summary(baseline_result),
            "dynamic_exposure": result_exposure_summary(winner_result),
            "model_audit": {
                "refit_months": diagnostic_audit["refit_months"],
                "first": diagnostic_audit["maturity_audit"][0],
                "last": diagnostic_audit["maturity_audit"][-1],
            },
        }

    passed = bool(eligible) and all(checks.values())
    candidates.sort(key=candidate_rank, reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "candidate_passed_not_deployed" if passed else "research_only_not_promoted",
        "strategy": "LGBM with dynamic holding count and explicit cash exposure",
        "windows": {
            "selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_diagnostic": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        },
        "execution_contract": "signal close; next-open execution; 10-session stock rebalance; risk transitions execute next open; 12bp one-way cost",
        "candidate_manifest_sha256": manifest,
        "universe_snapshot_run_id": snapshot_id,
        "winner": winner,
        "selection_model_audit": {
            "refit_months": selection_audit["refit_months"],
            "first": selection_audit["maturity_audit"][0],
            "last": selection_audit["maturity_audit"][-1],
        },
        "diagnostic": diagnostic,
        "promotion_gate": {"passed": passed, "checks": checks},
        "candidates": candidates,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)) if curve_rows else None,
        "rebalance_data": str(REBALANCE_PATH.relative_to(BASE_DIR)) if curve_rows else None,
        "production_change": False,
        "warnings": warnings + [
            "2020-2026 has already been revealed and is a historical diagnostic, not a fresh blind test.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Cash earns zero interest in this backtest.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    perf = winner["selection"]["performance"]
    report = [
        "# LGBM 动态持股数与现金仓位研究 v1",
        "",
        f"- 状态：{'历史门禁通过，未部署' if passed else '未通过晋级门禁'}",
        f"- 选择冠军：`{winner['name']}`",
        f"- 选择期年化：{perf['annual_return']:.2%}",
        f"- 选择期最大回撤：{perf['max_drawdown']:.2%}",
        f"- 选择期夏普：{perf['sharpe']:.3f}",
        f"- 选择期平均仓位：{winner['exposure']['average_exposure']:.1%}",
        "- 生产修改：否",
        "",
        "## 2018-2019选择期候选",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | 平均仓位 |",
        "|---|---:|---:|---:|---:|",
        *[
            "| {name} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} | {exposure:.1%} |".format(
                name=row["name"],
                annual=row["selection"]["performance"]["annual_return"],
                drawdown=row["selection"]["performance"]["max_drawdown"],
                sharpe=row["selection"]["performance"]["sharpe"],
                exposure=row["exposure"]["average_exposure"],
            )
            for row in candidates
        ],
    ]
    if diagnostic:
        base_perf = diagnostic["current_top5"]["performance"]
        dynamic_perf = diagnostic["dynamic_k_cash"]["performance"]
        report.extend([
            "",
            "## 2020-2026历史诊断",
            "",
            f"- 年化：{base_perf['annual_return']:.2%} -> {dynamic_perf['annual_return']:.2%}",
            f"- 最大回撤：{base_perf['max_drawdown']:.2%} -> {dynamic_perf['max_drawdown']:.2%}",
            f"- 夏普：{base_perf['sharpe']:.3f} -> {dynamic_perf['sharpe']:.3f}",
            f"- 平均仓位：{diagnostic['dynamic_exposure']['average_exposure']:.1%}",
            f"- 空仓交易日：{diagnostic['dynamic_exposure']['cash_days']}",
        ])
    report.extend([
        "",
        "所有风险切换在收盘确认，并于下一交易日开盘执行；现金收益按0计算。当前结果不自动修改模拟盘。",
    ])
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
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
