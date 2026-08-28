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
from research_lgbm_dynamic_k_cash_v1 import adjusted_holdings
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
TOP_K = 5
N_DROP = 1
COST_BPS = 12.0

OUTPUT_PATH = BASE_DIR / "data" / "lgbm_dynamic_frequency_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_dynamic_frequency_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_dynamic_frequency_curves.csv"
REBALANCE_PATH = BASE_DIR / "reports" / "lgbm_dynamic_frequency_rebalances.csv"

# This grid is frozen before opening the repeatedly revealed 2020+ diagnostic.
CANDIDATES = {
    "fixed_10": {
        "use_volatility": False,
        "use_rank_trigger": False,
        "high_interval": 10,
        "normal_interval": 10,
        "calm_interval": 10,
        "min_hold": 10,
        "max_interval": 10,
        "rank_buffer": 20,
        "score_advantage": 0.15,
    },
    "vol_5_10": {
        "use_volatility": True,
        "use_rank_trigger": False,
        "high_interval": 5,
        "normal_interval": 10,
        "calm_interval": 10,
        "min_hold": 5,
        "max_interval": 10,
        "rank_buffer": 20,
        "score_advantage": 0.15,
    },
    "vol_5_10_15": {
        "use_volatility": True,
        "use_rank_trigger": False,
        "high_interval": 5,
        "normal_interval": 10,
        "calm_interval": 15,
        "min_hold": 5,
        "max_interval": 15,
        "rank_buffer": 20,
        "score_advantage": 0.15,
    },
    "rank_5_15": {
        "use_volatility": False,
        "use_rank_trigger": True,
        "high_interval": 15,
        "normal_interval": 15,
        "calm_interval": 15,
        "min_hold": 5,
        "max_interval": 15,
        "rank_buffer": 20,
        "score_advantage": 0.15,
    },
    "hybrid": {
        "use_volatility": True,
        "use_rank_trigger": True,
        "high_interval": 5,
        "normal_interval": 10,
        "calm_interval": 15,
        "min_hold": 5,
        "max_interval": 15,
        "rank_buffer": 20,
        "score_advantage": 0.15,
    },
}


def validate_candidates() -> str:
    expected_order = ["fixed_10", "vol_5_10", "vol_5_10_15", "rank_5_15", "hybrid"]
    if list(CANDIDATES) != expected_order:
        raise RuntimeError("动态交易频率候选顺序已改变")
    expected_keys = {
        "use_volatility", "use_rank_trigger", "high_interval", "normal_interval",
        "calm_interval", "min_hold", "max_interval", "rank_buffer", "score_advantage",
    }
    for name, config in CANDIDATES.items():
        if set(config) != expected_keys:
            raise RuntimeError(f"{name} 参数不完整")
        intervals = [
            config["high_interval"], config["normal_interval"],
            config["calm_interval"], config["min_hold"], config["max_interval"],
        ]
        if min(intervals) < 1 or config["min_hold"] > config["max_interval"]:
            raise RuntimeError(f"{name} 调仓间隔无效")
        if config["rank_buffer"] < TOP_K or config["score_advantage"] < 0:
            raise RuntimeError(f"{name} 排名触发参数无效")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_causal_volatility_regime(history: pd.DataFrame) -> pd.DataFrame:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    close = data.pivot(index="date", columns="code", values="close").sort_index()
    pool_return = close.pct_change(fill_method=None).mean(axis=1, skipna=True)
    lagged_vol20 = pool_return.shift(1).rolling(20, min_periods=20).std(ddof=0).mul(
        np.sqrt(252)
    )
    threshold_source = lagged_vol20.shift(1)
    high_threshold = threshold_source.rolling(252, min_periods=126).quantile(0.75)
    low_threshold = threshold_source.rolling(252, min_periods=126).quantile(0.25)
    regime = pd.DataFrame({
        "pool_return": pool_return,
        "lagged_vol20": lagged_vol20,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
    })
    regime["volatility_regime"] = "normal"
    regime.loc[lagged_vol20.gt(high_threshold), "volatility_regime"] = "high"
    regime.loc[lagged_vol20.lt(low_threshold), "volatility_regime"] = "calm"
    return regime


def target_interval(config: dict, volatility_regime: str) -> int:
    if not config["use_volatility"]:
        return int(config["max_interval"])
    if volatility_regime == "high":
        return int(config["high_interval"])
    if volatility_regime == "calm":
        return int(config["calm_interval"])
    return int(config["normal_interval"])


def rank_deteriorated(
    cross: pd.Series,
    holdings: list[str],
    rank_buffer: int,
    score_advantage: float,
) -> tuple[bool, dict]:
    if not holdings:
        return False, {"worst_holding_rank": None, "challenger_advantage": None}
    available = [code for code in holdings if code in cross.index]
    if len(available) < len(holdings):
        return True, {"worst_holding_rank": None, "challenger_advantage": None}
    ranks = pd.Series(np.arange(1, len(cross) + 1), index=cross.index)
    worst_code = min(available, key=lambda code: float(cross.loc[code]))
    challengers = [code for code in cross.index.astype(str) if code not in holdings]
    advantage = (
        float(cross.loc[challengers[0]] - cross.loc[worst_code]) if challengers else 0.0
    )
    worst_rank = int(ranks.loc[worst_code])
    triggered = worst_rank > int(rank_buffer) or advantage > float(score_advantage)
    return triggered, {
        "worst_holding_rank": worst_rank,
        "challenger_advantage": advantage,
    }


def dynamic_frequency_backtest(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    regime: pd.DataFrame,
    config: dict,
    start: date,
    end: date,
    top_k: int = TOP_K,
    n_drop: int = N_DROP,
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
    score_lookup = scores.copy()
    score_lookup["date"] = pd.to_datetime(score_lookup["date"])
    score_lookup = score_lookup.set_index(["date", "code"])["score"]
    regime_lookup = regime["volatility_regime"].reindex(dates).fillna("normal")

    pending_targets: dict[int, list[str]] = {}
    planned_holdings: list[str] = []
    last_signal_index: int | None = None
    targets = pd.DataFrame(index=dates, columns=open_prices.columns, dtype=float)
    records: list[dict] = []

    for signal_index, signal_date in enumerate(dates):
        if signal_index in pending_targets:
            planned_holdings = pending_targets.pop(signal_index)
        try:
            cross = score_lookup.loc[signal_date].dropna().sort_values(ascending=False)
        except KeyError:
            continue
        cross.index = cross.index.astype(str)
        cross = cross[cross.index.isin(open_prices.columns.astype(str))]
        if len(cross) < top_k:
            continue

        volatility_regime = str(regime_lookup.loc[signal_date])
        interval = target_interval(config, volatility_regime)
        elapsed = signal_index - last_signal_index if last_signal_index is not None else None
        trigger = None
        rank_audit = {"worst_holding_rank": None, "challenger_advantage": None}
        if not planned_holdings:
            trigger = "scheduled"
        elif elapsed is not None and elapsed >= interval:
            if config["use_volatility"] and volatility_regime == "high":
                trigger = "high_vol"
            elif (
                (config["use_volatility"] and volatility_regime == "calm")
                or config["use_rank_trigger"]
            ) and interval >= int(config["max_interval"]):
                trigger = "max_interval"
            else:
                trigger = "scheduled"
        elif (
            config["use_rank_trigger"]
            and elapsed is not None
            and elapsed >= int(config["min_hold"])
        ):
            deteriorated, rank_audit = rank_deteriorated(
                cross,
                planned_holdings,
                int(config["rank_buffer"]),
                float(config["score_advantage"]),
            )
            if deteriorated:
                trigger = "rank_deterioration"
        if trigger is None:
            continue

        execution_index = signal_index + 1
        if execution_index >= len(dates) - 1:
            break
        target_codes, dropped = adjusted_holdings(
            cross, planned_holdings, top_k, n_drop=n_drop
        )
        execution_date = dates[execution_index]
        targets.loc[execution_date, :] = 0.0
        targets.loc[execution_date, target_codes] = 1 / len(target_codes)
        pending_targets[execution_index] = target_codes
        records.append({
            "signal_date": signal_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "trigger": trigger,
            "volatility_regime": volatility_regime,
            "target_interval": interval,
            "elapsed_signal_sessions": elapsed,
            "holdings": target_codes,
            "dropped": dropped,
            **rank_audit,
        })
        last_signal_index = signal_index

    current_weights = pd.Series(0.0, index=open_prices.columns)
    daily_returns = pd.Series(0.0, index=dates)
    turnovers = pd.Series(0.0, index=dates)
    active = pd.Series(False, index=dates)
    for trade_date in dates:
        target = targets.loc[trade_date]
        if target.notna().any():
            target = target.fillna(0.0)
            turnovers.loc[trade_date] = target.sub(current_weights).abs().sum()
            current_weights = target
        active.loc[trade_date] = current_weights.sum() > 0
        returns_today = period_returns.loc[trade_date].fillna(0.0)
        gross_return = float(current_weights.mul(returns_today).sum())
        daily_returns.loc[trade_date] = gross_return
        growth = 1 + gross_return
        if current_weights.sum() > 0 and growth > 0:
            current_weights = current_weights.mul(1 + returns_today).div(growth)

    costs = turnovers.mul(cost_bps / 10_000)
    strategy_returns = (daily_returns - costs).where(active, 0.0)
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
        "config": {**config, "top_k": top_k, "n_drop": n_drop, "cost_bps": cost_bps},
    }


def candidate_passes(row: dict, baseline: dict) -> bool:
    performance = row["selection"]["performance"]
    base_performance = baseline["selection"]["performance"]
    activity = row["selection"]["activity"]
    base_activity = baseline["selection"]["activity"]
    return bool(
        row["name"] != "fixed_10"
        and performance["annual_return"] >= base_performance["annual_return"] * 0.95
        and abs(performance["max_drawdown"]) <= abs(base_performance["max_drawdown"]) * 0.90
        and performance["sharpe"] >= base_performance["sharpe"]
        and activity["annual_turnover"] <= base_activity["annual_turnover"] * 1.50
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


def schedule_summary(result: dict) -> dict:
    records = result["rebalances"]
    intervals = [
        row["elapsed_signal_sessions"]
        for row in records
        if row["elapsed_signal_sessions"] is not None
    ]
    triggers = pd.Series([row["trigger"] for row in records], dtype="object").value_counts()
    return {
        "rebalance_count": len(records),
        "average_signal_interval": float(np.mean(intervals)) if intervals else 0.0,
        "minimum_signal_interval": int(min(intervals)) if intervals else 0,
        "maximum_signal_interval": int(max(intervals)) if intervals else 0,
        "trigger_counts": {str(key): int(value) for key, value in triggers.items()},
    }


def run_scores(history: pd.DataFrame, start: date, end: date) -> tuple[pd.DataFrame, dict]:
    frame, feature_columns = build_alpha158_lite(history)
    baseline = baseline_scores(frame, start, end)
    prediction, audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, start, end
    )
    scores = blend_scores(frame, prediction, baseline, WINNER_CONFIG["baseline_weight"])
    return scores, audit


def export_curves(dynamic_result: dict, baseline_result: dict) -> int:
    curves = pd.concat(
        {
            "dynamic_frequency_lgbm": normalized_curve(dynamic_result, "strategy_return"),
            "fixed_10_lgbm": normalized_curve(baseline_result, "strategy_return"),
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
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()

    selection_history = history[history["date"].le(pd.Timestamp(SELECTION_END))].copy()
    selection_scores, selection_audit = run_scores(
        selection_history, SELECTION_START, SELECTION_END
    )
    selection_regime = build_causal_volatility_regime(selection_history)
    candidates = []
    selection_results = {}
    for name, config in CANDIDATES.items():
        result = dynamic_frequency_backtest(
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
            "schedule": schedule_summary(result),
        })

    baseline_row = next(row for row in candidates if row["name"] == "fixed_10")
    reference_result = topk_dropout_backtest(
        selection_history, selection_scores, SELECTION_START, SELECTION_END
    )
    fixed_periods = pd.DataFrame(selection_results["fixed_10"]["return_periods"])
    reference_periods = pd.DataFrame(reference_result["return_periods"])
    baseline_reproduction = {
        "strategy_returns_match": bool(np.allclose(
            fixed_periods["strategy_return"], reference_periods["strategy_return"], atol=1e-12
        )),
        "turnover_matches": bool(np.allclose(
            fixed_periods["turnover"], reference_periods["turnover"], atol=1e-12
        )),
        "rebalance_count_match": (
            len(selection_results["fixed_10"]["rebalances"])
            == len(reference_result["rebalances"])
        ),
    }
    if not all(baseline_reproduction.values()):
        raise RuntimeError("固定10日动态回测未能复现当前LGBM Top5基线")

    eligible = [row for row in candidates if candidate_passes(row, baseline_row)]
    winner = max(eligible, key=candidate_rank) if eligible else baseline_row
    winner["selection_gate_passed"] = bool(eligible)

    diagnostic = None
    curve_rows = 0
    checks = {"selection_candidate_exists": bool(eligible)}
    if eligible:
        diagnostic_scores, diagnostic_audit = run_scores(
            history, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        diagnostic_regime = build_causal_volatility_regime(history)
        baseline_result = dynamic_frequency_backtest(
            history,
            diagnostic_scores,
            diagnostic_regime,
            CANDIDATES["fixed_10"],
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
        )
        winner_result = dynamic_frequency_backtest(
            history,
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
        base_performance = baseline_summary["performance"]
        winner_performance = winner_summary["performance"]
        checks.update({
            "annual_return_retained_95pct": (
                winner_performance["annual_return"] >= base_performance["annual_return"] * 0.95
            ),
            "drawdown_improved_10pct": (
                abs(winner_performance["max_drawdown"])
                <= abs(base_performance["max_drawdown"]) * 0.90
            ),
            "sharpe_not_lower": winner_performance["sharpe"] >= base_performance["sharpe"],
            "turnover_not_higher_50pct": (
                winner_summary["activity"]["annual_turnover"]
                <= baseline_summary["activity"]["annual_turnover"] * 1.50
            ),
        })
        curve_rows = export_curves(winner_result, baseline_result)
        pd.DataFrame(winner_result["rebalances"]).to_csv(
            REBALANCE_PATH, index=False, encoding="utf-8-sig"
        )
        diagnostic = {
            "fixed_10": baseline_summary,
            "dynamic_frequency": winner_summary,
            "fixed_schedule": schedule_summary(baseline_result),
            "dynamic_schedule": schedule_summary(winner_result),
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
        "status": "historical_gate_passed_not_deployed" if passed else "research_only_not_promoted",
        "strategy": "expanding LGBM Top5 with causal dynamic rebalance frequency",
        "windows": {
            "candidate_selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_deployment_blocker": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        },
        "execution_contract": (
            "signal at close; frequency decision uses lagged pool volatility; execution at next open; "
            "Top5; at most one ordinary replacement per event; 12bp one-way cost on actual weight changes"
        ),
        "candidate_manifest_sha256": manifest,
        "universe_snapshot_run_id": snapshot_id,
        "baseline_reproduction": baseline_reproduction,
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
        "forward_paper_required_sessions": 126,
        "warnings": warnings + [
            "2020-2026 has been repeatedly revealed and is only a deployment blocker, not a fresh holdout.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "Valuation fields do not have certified per-field availability timestamps.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# LGBM 动态交易频率研究 v1",
        "",
        f"- 状态：{'历史门禁通过，未部署' if passed else '未通过晋级门禁'}",
        f"- 选择冠军：`{winner['name']}`",
        "- 选择期：2018-01-01 至 2019-12-31",
        "- 生产修改：否",
        "",
        "## 2018-2019冻结候选",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | 年化换手 | 平均调仓间隔 |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            "| {name} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} | {turnover:.2f}x | {interval:.2f}日 |".format(
                name=row["name"],
                annual=row["selection"]["performance"]["annual_return"],
                drawdown=row["selection"]["performance"]["max_drawdown"],
                sharpe=row["selection"]["performance"]["sharpe"],
                turnover=row["selection"]["activity"]["annual_turnover"],
                interval=row["schedule"]["average_signal_interval"],
            )
            for row in candidates
        ],
    ]
    if diagnostic:
        base = diagnostic["fixed_10"]["performance"]
        dynamic = diagnostic["dynamic_frequency"]["performance"]
        report.extend([
            "",
            "## 2020-2026已揭示区间阻断验证",
            "",
            f"- 年化：{base['annual_return']:.2%} -> {dynamic['annual_return']:.2%}",
            f"- 最大回撤：{base['max_drawdown']:.2%} -> {dynamic['max_drawdown']:.2%}",
            f"- 夏普：{base['sharpe']:.3f} -> {dynamic['sharpe']:.3f}",
            f"- 平均调仓间隔：{diagnostic['fixed_schedule']['average_signal_interval']:.2f} -> {diagnostic['dynamic_schedule']['average_signal_interval']:.2f}日",
        ])
    report.extend([
        "",
        "波动率只使用截至信号日前一日的收盘收益；所有交易在下一交易日开盘发生。当前结果不会自动修改模拟盘。",
    ])
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
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
