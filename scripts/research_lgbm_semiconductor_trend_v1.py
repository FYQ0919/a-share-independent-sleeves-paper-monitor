from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.backtest_engine import BacktestEngine
from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_dynamic_k_cash_v1 import adjusted_holdings
from research_lgbm_early_excess_v1 import build_scores, run_portfolio
from research_lgbm_new_factors_v1 import (
    blend_model_predictions,
    build_extended_feature_frame,
)
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    normalized_curve,
    research_summary,
)


START = date(2020, 1, 2)
END = date(2026, 8, 25)
COST_BPS = 12.0
TOP_K = 5
REBALANCE_DAYS = 10

# These are the semiconductor-equipment names that were already present in the
# frozen production Top50. The list is not expanded with hindsight winners.
EQUIPMENT_CODES = {"688037", "688120", "688200", "688630"}

OUTPUT_JSON = BASE_DIR / "data" / "lgbm_semiconductor_trend_v1.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "lgbm_semiconductor_trend_v1.md"
OUTPUT_CURVES = BASE_DIR / "reports" / "lgbm_semiconductor_trend_v1_curves.csv"
OUTPUT_REBALANCES = BASE_DIR / "reports" / "lgbm_semiconductor_trend_v1_rebalances.csv"


def build_trend_expert(
    frame: pd.DataFrame,
    model_prediction: pd.Series,
    rule_score: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a trailing-data-only trend overlay and its equipment regime."""
    working = frame[["date", "code", "ret_20", "ret_60", "amount_20", "amount_60"]].copy()
    working["date"] = pd.to_datetime(working["date"])
    working["amount_expansion"] = (
        working["amount_20"].div(working["amount_60"].replace(0.0, np.nan)).sub(1.0)
    )

    working["ret20_rank"] = working.groupby("date")["ret_20"].rank(pct=True)
    working["ret60_rank"] = working.groupby("date")["ret_60"].rank(pct=True)
    working["amount_rank"] = working.groupby("date")["amount_expansion"].rank(pct=True)
    working["individual_trend"] = (
        working["ret20_rank"].mul(0.45)
        + working["ret60_rank"].mul(0.35)
        + working["amount_rank"].mul(0.20)
    )
    working["equipment_member"] = working["code"].astype(str).isin(EQUIPMENT_CODES)

    market = working.groupby("date").agg(
        market_ret20=("ret_20", "median"),
        market_ret60=("ret_60", "median"),
    )
    equipment = (
        working.loc[working["equipment_member"]]
        .groupby("date")
        .agg(
            equipment_count=("ret_20", "count"),
            equipment_ret20=("ret_20", "median"),
            equipment_ret60=("ret_60", "median"),
            equipment_breadth=("ret_20", lambda values: float(values.gt(0.0).mean())),
        )
    )
    regime = market.join(equipment, how="left")
    regime["relative_ret20"] = regime["equipment_ret20"] - regime["market_ret20"]
    regime["relative_ret60"] = regime["equipment_ret60"] - regime["market_ret60"]
    regime["strong_equipment"] = (
        regime["equipment_count"].ge(3)
        & regime["equipment_breadth"].ge(0.75)
        & regime["relative_ret20"].gt(0.0)
        & regime["relative_ret60"].gt(0.0)
    ).fillna(False)

    strong = working["date"].map(regime["strong_equipment"]).fillna(False)
    working["trend_expert"] = working["individual_trend"].mul(0.75)
    working.loc[strong & working["equipment_member"], "trend_expert"] += 0.25
    working.loc[~strong, "trend_expert"] = np.nan

    model_rank = model_prediction.groupby(frame["date"]).rank(pct=True)
    baseline = build_scores(
        frame,
        model_prediction,
        model_prediction,
        rule_score,
        model_weight=0.0,
        rule_weight=WINNER_CONFIG["baseline_weight"],
    )
    dynamic = baseline.copy()
    active_score = (
        model_rank.mul(0.60)
        + rule_score.mul(0.20)
        + working["trend_expert"].mul(0.20)
    )
    active_rows = strong & baseline["score"].notna() & active_score.notna()
    dynamic.loc[active_rows, "score"] = active_score.loc[active_rows]
    dynamic["strong_equipment"] = strong.to_numpy(dtype=bool)
    dynamic["equipment_member"] = working["equipment_member"].to_numpy(dtype=bool)
    return dynamic, regime


def cap_ranked_equipment(cross: pd.Series, cap: int | None) -> pd.Series:
    if cap is None:
        return cross
    kept: list[str] = []
    equipment_count = 0
    for code in cross.index.astype(str):
        if code in EQUIPMENT_CODES:
            if equipment_count >= cap:
                continue
            equipment_count += 1
        kept.append(code)
    return cross.reindex(kept)


def dynamic_backtest(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    dynamic_drop: bool,
    equipment_cap: int | None,
) -> dict:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))]
    open_prices = data.pivot(index="date", columns="code", values="open").reindex(dates)
    period_returns = (
        open_prices.shift(-1).div(open_prices).sub(1.0).replace([np.inf, -np.inf], np.nan)
    )
    benchmark_returns = period_returns.mean(axis=1, skipna=True).fillna(0.0)
    lookup_columns = ["score"]
    if "strong_equipment" in scores:
        lookup_columns.append("strong_equipment")
    score_lookup = scores.set_index(["date", "code"])[lookup_columns]

    pending_targets: dict[int, list[str]] = {}
    planned_holdings: list[str] = []
    targets = pd.DataFrame(index=dates, columns=open_prices.columns, dtype=float)
    records: list[dict] = []
    for signal_index, signal_date in enumerate(dates):
        if signal_index in pending_targets:
            planned_holdings = pending_targets.pop(signal_index)
        if signal_index % REBALANCE_DAYS != 0:
            continue
        execution_index = signal_index + 1
        if execution_index >= len(dates) - 1:
            continue
        try:
            cross_frame = score_lookup.loc[signal_date].dropna(subset=["score"])
        except KeyError:
            continue
        cross_frame = cross_frame[cross_frame.index.isin(open_prices.columns)]
        cross = cross_frame["score"].sort_values(ascending=False)
        cross = cap_ranked_equipment(cross, equipment_cap)
        if len(cross) < TOP_K:
            continue
        strong = bool(cross_frame["strong_equipment"].iloc[0]) if dynamic_drop else False
        n_drop = 2 if strong else 1
        target_codes, dropped = adjusted_holdings(
            cross, planned_holdings, TOP_K, n_drop=n_drop
        )

        execution_date = dates[execution_index]
        targets.loc[execution_date, :] = 0.0
        targets.loc[execution_date, target_codes] = 1.0 / len(target_codes)
        pending_targets[execution_index] = target_codes
        records.append({
            "signal_date": signal_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "strong_equipment": strong,
            "n_drop": n_drop,
            "holdings": target_codes,
            "equipment_holdings": [code for code in target_codes if code in EQUIPMENT_CODES],
            "dropped": dropped,
        })

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
        active.loc[trade_date] = current_weights.sum() > 0.0
        returns_today = period_returns.loc[trade_date].fillna(0.0)
        gross_return = float(current_weights.mul(returns_today).sum())
        daily_returns.loc[trade_date] = gross_return
        growth = 1.0 + gross_return
        if current_weights.sum() > 0.0 and growth > 0.0:
            current_weights = current_weights.mul(1.0 + returns_today).div(growth)

    costs = turnovers.mul(COST_BPS / 10_000.0)
    strategy_returns = daily_returns.sub(costs).where(active, 0.0)
    strategy_equity = strategy_returns.add(1.0).cumprod()
    benchmark_equity = benchmark_returns.add(1.0).cumprod()
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
        "config": {
            "top_k": TOP_K,
            "rebalance_days": REBALANCE_DAYS,
            "base_n_drop": 1,
            "strong_equipment_n_drop": 2 if dynamic_drop else 1,
            "equipment_cap": equipment_cap,
            "cost_bps": COST_BPS,
        },
    }


def return_series(result: dict) -> pd.Series:
    rows = pd.DataFrame(result["return_periods"])
    return pd.Series(
        rows["strategy_return"].to_numpy(dtype=float),
        index=pd.to_datetime(rows["period_start"]),
    )


def period_table(result: dict) -> dict[str, dict]:
    windows = {
        "2020_2023": (date(2020, 1, 2), date(2023, 12, 31)),
        "2024_2025": (date(2024, 1, 1), date(2025, 12, 31)),
        "2026_ytd": (date(2026, 1, 1), END),
        "2026_june": (date(2026, 6, 1), date(2026, 7, 1)),
        "full": (START, END),
    }
    return {name: research_summary(result, start, end) for name, (start, end) in windows.items()}


def compact_performance(summary: dict) -> dict:
    performance = summary["performance"]
    activity = summary["activity"]
    return {
        "total_return": performance["total_return"],
        "annual_return": performance["annual_return"],
        "sharpe": performance["sharpe"],
        "max_drawdown": performance["max_drawdown"],
        "annual_turnover": activity["annual_turnover"],
    }


def main() -> None:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    frozen_codes = {str(item["code"]) for item in universe}
    if not EQUIPMENT_CODES.issubset(frozen_codes):
        raise RuntimeError("Frozen Top50 no longer contains the fixed equipment group")

    history, _, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(END))].copy()
    frame, feature_sets = build_extended_feature_frame(history)
    frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    print("fit expanding Alpha158-lite model", flush=True)
    base_prediction, base_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_baseline"],
        WINNER_CONFIG,
        START,
        END,
    )
    print("fit expanding Alpha158+Barra model", flush=True)
    barra_prediction, barra_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_plus_barra"],
        WINNER_CONFIG,
        START,
        END,
    )
    model_prediction = blend_model_predictions(
        frame, base_prediction, barra_prediction, candidate_weight=0.25
    )
    rule_score = baseline_scores(frame, START, END)
    baseline_scores_frame = build_scores(
        frame,
        base_prediction,
        barra_prediction,
        rule_score,
        model_weight=0.25,
        rule_weight=0.40,
    )
    trend_scores, regime = build_trend_expert(frame, model_prediction, rule_score)

    official_baseline = run_portfolio(
        history,
        baseline_scores_frame,
        top_k=TOP_K,
        rebalance_days=REBALANCE_DAYS,
        n_drop=1,
        start=START,
        end=END,
    )
    replay_baseline = dynamic_backtest(
        history, baseline_scores_frame, dynamic_drop=False, equipment_cap=None
    )
    baseline_error = float(
        return_series(official_baseline).sub(return_series(replay_baseline)).abs().max()
    )
    if baseline_error > 1e-12:
        raise RuntimeError(f"Baseline replay mismatch: {baseline_error:.3e}")

    trend_nocap = dynamic_backtest(
        history, trend_scores, dynamic_drop=True, equipment_cap=None
    )
    trend_cap2 = dynamic_backtest(
        history, trend_scores, dynamic_drop=True, equipment_cap=2
    )
    variants = {
        "current_lgbm": replay_baseline,
        "trend_expert_nocap": trend_nocap,
        "trend_expert_cap2": trend_cap2,
    }
    summaries = {
        name: {period: compact_performance(summary) for period, summary in period_table(result).items()}
        for name, result in variants.items()
    }

    curves = pd.concat(
        {name: normalized_curve(result, "strategy_return") for name, result in variants.items()},
        axis=1,
        join="inner",
    )
    curves.index.name = "date"
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")

    rebalance_rows: list[dict] = []
    for name, result in variants.items():
        for row in result["rebalances"]:
            rebalance_rows.append({
                "variant": name,
                **row,
                "holdings": ",".join(row["holdings"]),
                "equipment_holdings": ",".join(row.get("equipment_holdings", [])),
                "dropped": ",".join(row["dropped"]),
            })
    pd.DataFrame(rebalance_rows).to_csv(OUTPUT_REBALANCES, index=False, encoding="utf-8-sig")

    signal_regime = regime.reindex(pd.to_datetime([row["signal_date"] for row in trend_nocap["rebalances"]]))
    strong_signal_count = int(signal_regime["strong_equipment"].fillna(False).sum())
    full_base = summaries["current_lgbm"]["full"]
    full_cap = summaries["trend_expert_cap2"]["full"]
    gate = {
        "baseline_replay_exact": baseline_error <= 1e-12,
        "annual_return_retained": full_cap["annual_return"] >= full_base["annual_return"] * 0.98,
        "drawdown_not_worse": abs(full_cap["max_drawdown"]) <= abs(full_base["max_drawdown"]) * 1.05,
        "sharpe_not_lower": full_cap["sharpe"] >= full_base["sharpe"],
    }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "historical_diagnostic_only_not_deployed",
        "strategy": "current LGBM plus fixed semiconductor-equipment trend expert",
        "universe_snapshot_run_id": snapshot_id,
        "equipment_codes": sorted(EQUIPMENT_CODES),
        "execution_contract": (
            "close signal; next-open execution; Top5 equal weight; 10-session rebalance; "
            "12bp one-way stock cost"
        ),
        "overlay": {
            "strong_regime": (
                "at least 3 equipment names; 20-day positive breadth >= 75%; "
                "equipment median 20/60-day return above frozen-pool median"
            ),
            "strong_score": "60% blended LGBM + 20% defensive rule + 20% trend expert",
            "normal_score": "60% blended LGBM + 40% defensive rule",
            "strong_max_replacements": 2,
            "normal_max_replacements": 1,
            "trend_expert": (
                "75% x (45% ret20 rank + 35% ret60 rank + 20% amount expansion rank) "
                "+ 25% equipment membership in strong regime"
            ),
        },
        "model_audit": {
            "alpha158_lite_refits": base_audit["refit_months"],
            "alpha158_barra_refits": barra_audit["refit_months"],
            "first_base_refit": base_audit["maturity_audit"][0],
            "last_base_refit": base_audit["maturity_audit"][-1],
            "all_base_labels_strictly_mature": all(
                pd.Timestamp(row["max_training_label_end_date"]) < pd.Timestamp(row["prediction_start"])
                for row in base_audit["maturity_audit"]
            ),
            "all_barra_labels_strictly_mature": all(
                pd.Timestamp(row["max_training_label_end_date"]) < pd.Timestamp(row["prediction_start"])
                for row in barra_audit["maturity_audit"]
            ),
        },
        "strong_equipment_signal_events": strong_signal_count,
        "summaries": summaries,
        "gate": {"passed": all(gate.values()), "checks": gate},
        "baseline_replay_max_daily_error": baseline_error,
        "artifacts": {
            "curves": str(OUTPUT_CURVES.relative_to(BASE_DIR)),
            "rebalances": str(OUTPUT_REBALANCES.relative_to(BASE_DIR)),
        },
        "production_change": False,
        "warnings": [
            *warnings,
            "The frozen current Top50 is backfilled and has constituent and survivorship bias.",
            "The equipment code map is static and incomplete; most equipment leaders are outside the frozen Top50.",
            "The 2020-2026 interval and June 2026 episode were already revealed, so this is not fresh out-of-sample evidence.",
            "No parameter was selected from 2026 outcomes, but the research hypothesis was prompted by the revealed June episode.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Semiconductor trend expert diagnostic v1",
        "",
        "- Status: historical diagnostic only; production unchanged.",
        f"- Frozen universe: {snapshot_id}.",
        f"- Strong equipment signal events: {strong_signal_count}.",
        f"- Baseline replay max daily error: {baseline_error:.3e}.",
        "",
        "| Window | Variant | Total return | Annual return | Sharpe | Max drawdown | Annual turnover |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period in ("2020_2023", "2024_2025", "2026_ytd", "2026_june", "full"):
        for name in variants:
            row = summaries[name][period]
            lines.append(
                f"| {period} | {name} | {row['total_return']:.2%} | "
                f"{row['annual_return']:.2%} | {row['sharpe']:.3f} | "
                f"{row['max_drawdown']:.2%} | {row['annual_turnover']:.2f}x |"
            )
    lines.extend([
        "",
        f"Gate: {'PASS' if payload['gate']['passed'] else 'FAIL'}; no deployment regardless of historical result.",
        "",
        "The overlay uses only trailing values available at the signal close. Model refits retain the existing label-maturity gap.",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_JSON),
        "report": str(OUTPUT_REPORT),
        "curves": str(OUTPUT_CURVES),
        "rebalances": str(OUTPUT_REBALANCES),
        "summaries": summaries,
        "gate": payload["gate"],
        "baseline_replay_max_daily_error": baseline_error,
        "strong_equipment_signal_events": strong_signal_count,
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
