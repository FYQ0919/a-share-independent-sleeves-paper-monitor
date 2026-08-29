from __future__ import annotations

from datetime import date, datetime
import html
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

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from app.master_reproduction import prediction_metrics
from app.return_decomposition_factors import (
    SPECS as RETURN_FACTOR_SPECS,
    add_return_decomposition_factors,
)
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_multi_objective_v1 import (
    CANDIDATES as PORTFOLIO_CANDIDATES,
    multihead_portfolio_backtest,
)
from research_lgbm_new_factors_v1 import (
    blend_model_predictions,
    build_extended_feature_frame,
)
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    normalized_curve,
    research_summary,
)


START = date(2020, 1, 2)
END = date(2026, 8, 25)
NEW_FACTOR = "x_overnight_momentum_5"
CURRENT_BARRA_WEIGHT = 0.25
OVERLAY_WEIGHT = 0.15
REBALANCE_DAYS = 10
N_DROP = 1
COST_BPS = 12.0

OUTPUT_JSON = BASE_DIR / "data" / "overnight_factor_lgbm_2020_2026.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "overnight_factor_lgbm_2020_2026.md"
OUTPUT_CURVES = BASE_DIR / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
OUTPUT_SVG = BASE_DIR / "reports" / "overnight_factor_lgbm_2020_2026.svg"

STRATEGY_LABELS = {
    "previous_lgbm": "Previous LGBM",
    "lgbm_factor_feature": "LGBM + factor feature",
    "lgbm_factor_overlay": "LGBM + fixed 15% overlay",
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def maturity_checks(audit: dict) -> dict[str, bool]:
    rows = audit.get("maturity_audit", [])
    return {
        row["month"]: pd.Timestamp(row["max_training_label_end_date"])
        < pd.Timestamp(row["prediction_start"])
        for row in rows
    }


def fixed_factor_overlay(
    frame: pd.DataFrame,
    current_scores: pd.DataFrame,
    factor: str = NEW_FACTOR,
    weight: float = OVERLAY_WEIGHT,
) -> pd.DataFrame:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("overlay weight must be in [0, 1]")
    factor_values = pd.to_numeric(frame[factor], errors="coerce")
    factor_rank = factor_values.groupby(frame["date"], sort=False).rank(pct=True)
    output = current_scores.copy()
    output["score"] = output["score"].mul(1.0 - weight).add(
        factor_rank.mul(weight)
    )
    output.loc[current_scores["score"].isna() | factor_rank.isna(), "score"] = np.nan
    return output


def run_portfolio(history: pd.DataFrame, scores: pd.DataFrame) -> dict:
    prepared = scores.copy()
    prepared["expected_return"] = 0.0
    prepared["downside_probability"] = 0.0
    return multihead_portfolio_backtest(
        history,
        prepared,
        PORTFOLIO_CANDIDATES["current_rank_top5"],
        START,
        END,
        rebalance_days=REBALANCE_DAYS,
        n_drop=N_DROP,
        cost_bps=COST_BPS,
    )


def period_windows() -> dict[str, tuple[date, date]]:
    windows = {
        "full_2020_2026": (START, END),
        "2020_2023": (START, date(2023, 12, 31)),
        "2024_2026": (date(2024, 1, 1), END),
    }
    for year in range(2020, 2027):
        windows[str(year)] = (
            max(START, date(year, 1, 1)),
            min(END, date(year, 12, 31)),
        )
    return windows


def summarize_result(result: dict) -> dict[str, dict]:
    return {
        name: research_summary(result, start, end)
        for name, (start, end) in period_windows().items()
        if start <= end
    }


def promotion_checks(candidate: dict, previous: dict) -> dict[str, bool]:
    candidate_full = candidate["full_2020_2026"]
    previous_full = previous["full_2020_2026"]
    return {
        "full_cagr_not_lower": (
            candidate_full["performance"]["annual_return"]
            >= previous_full["performance"]["annual_return"]
        ),
        "full_sharpe_not_lower": (
            candidate_full["performance"]["sharpe"]
            >= previous_full["performance"]["sharpe"]
        ),
        "full_drawdown_not_worse": (
            candidate_full["performance"]["max_drawdown"]
            >= previous_full["performance"]["max_drawdown"]
        ),
        "turnover_not_higher_10pct": (
            candidate_full["activity"]["annual_turnover"]
            <= previous_full["activity"]["annual_turnover"] * 1.10
        ),
        "2020_2023_cagr_not_lower": (
            candidate["2020_2023"]["performance"]["annual_return"]
            >= previous["2020_2023"]["performance"]["annual_return"]
        ),
        "2024_2026_cagr_not_lower": (
            candidate["2024_2026"]["performance"]["annual_return"]
            >= previous["2024_2026"]["performance"]["annual_return"]
        ),
    }


def build_curves(results: dict[str, dict]) -> pd.DataFrame:
    curves = pd.concat(
        {
            name: normalized_curve(result, "strategy_return")
            for name, result in results.items()
        },
        axis=1,
        join="inner",
    ).dropna()
    if curves.empty or not np.allclose(curves.iloc[0].to_numpy(dtype=float), 1.0):
        raise RuntimeError("strategy curves do not share a normalized 1.00 start")
    for name in results:
        curves[f"{name}_drawdown"] = curves[name].div(curves[name].cummax()).sub(1.0)
    curves.index.name = "date"
    return curves


def render_svg(curves: pd.DataFrame, path: Path) -> None:
    columns = list(STRATEGY_LABELS)
    width, height = 1120, 630
    left, right, top, bottom = 82, 235, 72, 64
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = curves[columns].to_numpy(dtype=float)
    low = min(0.90, float(np.nanmin(values)))
    high = max(1.10, float(np.nanmax(values)))
    padding = max((high - low) * 0.06, 0.05)
    low, high = max(0.0, low - padding), high + padding

    def x(index: int) -> float:
        return left + chart_width * index / max(len(curves) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / max(high - low, 1e-12)

    colors = {
        "previous_lgbm": "#176B87",
        "lgbm_factor_feature": "#B45309",
        "lgbm_factor_overlay": "#2F855A",
    }
    dash = {
        "previous_lgbm": "",
        "lgbm_factor_feature": ' stroke-dasharray="8 5"',
        "lgbm_factor_overlay": ' stroke-dasharray="3 4"',
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        '<text x="82" y="30" font-family="Segoe UI, Arial" font-size="20" font-weight="600" fill="#1F2933">Overnight factor LGBM comparison, 2020-2026</text>',
        '<text x="82" y="51" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Monthly expanding fit; close signal; next-open execution; Top5; 10-session rebalance; one replacement; 12 bps one-way</text>',
    ]
    for tick in np.linspace(low, high, 6):
        yy = y(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{left + chart_width}" y2="{yy:.2f}" stroke="#D9E2E8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 9}" y="{yy + 4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{tick:.2f}x</text>'
        )
    dates = pd.DatetimeIndex(curves.index)
    for year in range(dates.min().year, dates.max().year + 1):
        target = pd.Timestamp(year, 1, 1)
        index = int(np.argmin(np.abs((dates - target).days)))
        xx = x(index)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + chart_height}" stroke="#EEF0F2" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.2f}" y="{height - 37}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{year}</text>'
        )
    endpoints = []
    for column in columns:
        points = " ".join(
            f"{x(index):.2f},{y(float(value)):.2f}"
            for index, value in enumerate(curves[column])
        )
        parts.append(
            f'<polyline data-series="{column}" points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"{dash[column]}/>'
        )
        endpoints.append([column, y(float(curves[column].iloc[-1]))])
    endpoints.sort(key=lambda item: item[1])
    minimum_gap = 20.0
    for index in range(1, len(endpoints)):
        endpoints[index][1] = max(endpoints[index][1], endpoints[index - 1][1] + minimum_gap)
    overflow = endpoints[-1][1] - (top + chart_height - 6)
    if overflow > 0:
        for item in endpoints:
            item[1] -= overflow
    for column, label_y in endpoints:
        value = float(curves[column].iloc[-1])
        actual_y = y(value)
        x_end = left + chart_width
        parts.append(
            f'<circle cx="{x_end:.2f}" cy="{actual_y:.2f}" r="3" fill="{colors[column]}"/>'
        )
        parts.append(
            f'<line x1="{x_end + 4:.2f}" y1="{actual_y:.2f}" x2="{x_end + 17:.2f}" y2="{label_y:.2f}" stroke="{colors[column]}" stroke-width="1"/>'
        )
        label = html.escape(f"{STRATEGY_LABELS[column]}  {value:.2f}x")
        parts.append(
            f'<text data-end-label="{column}" x="{x_end + 23:.2f}" y="{label_y + 4:.2f}" font-family="Segoe UI, Arial" font-size="11" fill="#1F2933">{label}</text>'
        )
    parts.extend([
        f'<text x="{left + chart_width / 2:.2f}" y="{height - 14}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Signal date</text>',
        f'<text x="18" y="{top + chart_height / 2:.2f}" transform="rotate(-90 18 {top + chart_height / 2:.2f})" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Normalized net value (x)</text>',
        "</svg>",
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def _pct(value: float) -> str:
    return f"{value:.2%}"


def write_report(payload: dict) -> None:
    lines = [
        "# Overnight factor optimization of the current LGBM strategy",
        "",
        f"- Window: {payload['window']['start']} to {payload['window']['end']}",
        "- The 2020-2026 period has already been revealed. Results are historical diagnostics, not untouched out-of-sample evidence.",
        "- Previous architecture: 40% rule protection + 60% model rank; model rank is 75% Alpha158-lite + 25% Alpha158+Barra.",
        f"- New factor: `{NEW_FACTOR}` = {RETURN_FACTOR_SPECS[NEW_FACTOR]['formula']}.",
        "- Execution: close signal, next-open trade, Top5, rebalance every 10 sessions, at most one replacement, 12 bps one-way cost.",
        "",
        "## Full-period result",
        "",
        "| Strategy | Terminal value | CAGR | Sharpe | Max drawdown | Information ratio | Annual turnover | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in STRATEGY_LABELS.items():
        block = payload["periods"][name]["full_2020_2026"]
        perf, activity = block["performance"], block["activity"]
        lines.append(
            f"| {label} | {1 + perf['total_return']:.2f}x | {_pct(perf['annual_return'])} | "
            f"{perf['sharpe']:.3f} | {_pct(perf['max_drawdown'])} | "
            f"{perf['information_ratio']:.3f} | {activity['annual_turnover']:.2f}x | "
            f"{activity['total_cost_return_units']:.2%} |"
        )
    lines.extend([
        "",
        "## Subperiod stability",
        "",
        "| Period | Strategy | CAGR | Sharpe | Max drawdown |",
        "|---|---|---:|---:|---:|",
    ])
    for period in ("2020_2023", "2024_2026"):
        for name, label in STRATEGY_LABELS.items():
            perf = payload["periods"][name][period]["performance"]
            lines.append(
                f"| {period.replace('_', '-')} | {label} | {_pct(perf['annual_return'])} | "
                f"{perf['sharpe']:.3f} | {_pct(perf['max_drawdown'])} |"
            )
    lines.extend([
        "",
        "## Calendar-year CAGR",
        "",
        "| Year | Previous LGBM | LGBM + feature | Fixed overlay |",
        "|---|---:|---:|---:|",
    ])
    for year in map(str, range(2020, 2027)):
        values = [
            payload["periods"][name][year]["performance"]["annual_return"]
            for name in STRATEGY_LABELS
        ]
        lines.append(f"| {year} | {_pct(values[0])} | {_pct(values[1])} | {_pct(values[2])} |")
    lines.extend([
        "",
        "## Prediction diagnostics",
        "",
        "| Score | Mean IC | ICIR | Mean RankIC | RankIC IR |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in STRATEGY_LABELS.items():
        metrics = payload["prediction_metrics"][name]
        lines.append(
            f"| {label} | {metrics['mean_ic']:.4f} | {metrics['ic_ir']:.4f} | "
            f"{metrics['mean_rank_ic']:.4f} | {metrics['rank_ic_ir']:.4f} |"
        )
    lines.extend([
        "",
        "## Historical gate",
        "",
    ])
    for name in ("lgbm_factor_feature", "lgbm_factor_overlay"):
        checks = payload["historical_gate"][name]
        lines.append(
            f"- {STRATEGY_LABELS[name]}: {'pass' if all(checks.values()) else 'fail'}; "
            + ", ".join(f"{key}={'pass' if value else 'fail'}" for key, value in checks.items())
        )
    lines.extend([
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Audit and limitations",
        "",
        f"- Monthly refits: {payload['training_audit']['refit_months']} per branch; all mature-label checks passed: {payload['training_audit']['all_strictly_mature']}.",
        f"- New factor top-20 mean-gain rank: {payload['new_factor_model_gain']['top20_rank'] or 'outside top 20'}.",
        "- Current Top50 membership is backfilled and therefore has survivorship and constituent bias.",
        "- Current-vintage adjusted prices are not strict point-in-time prices.",
        "- Repeated inspection of 2020-2026 creates validation overfitting risk. A new forward paper-trading window is required before deployment.",
        "- Production model, paper account, scheduler and notification configuration were not changed.",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(END))].copy()
    frame, feature_sets = build_extended_feature_frame(history)
    frame, factor_features = add_return_decomposition_factors(frame)
    if NEW_FACTOR not in factor_features:
        raise RuntimeError(f"missing factor: {NEW_FACTOR}")
    frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    print("training Alpha158-lite monthly expanding branch", flush=True)
    base_prediction, base_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_baseline"],
        WINNER_CONFIG,
        START,
        END,
    )
    print("training Alpha158+Barra monthly expanding branch", flush=True)
    barra_prediction, barra_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_plus_barra"],
        WINNER_CONFIG,
        START,
        END,
    )
    print("training Alpha158+Barra+overnight-factor monthly expanding branch", flush=True)
    augmented_features = feature_sets["alpha158_plus_barra"] + [NEW_FACTOR]
    augmented_prediction, augmented_audit = fit_expanding_lgbm_predictions(
        frame,
        augmented_features,
        WINNER_CONFIG,
        START,
        END,
    )

    previous_prediction = blend_model_predictions(
        frame, base_prediction, barra_prediction, CURRENT_BARRA_WEIGHT
    )
    integrated_prediction = blend_model_predictions(
        frame, base_prediction, augmented_prediction, CURRENT_BARRA_WEIGHT
    )
    rules = baseline_scores(frame, START, END)
    previous_scores = blend_scores(
        frame, previous_prediction, rules, WINNER_CONFIG["baseline_weight"]
    )
    integrated_scores = blend_scores(
        frame, integrated_prediction, rules, WINNER_CONFIG["baseline_weight"]
    )
    overlay_scores = fixed_factor_overlay(frame, previous_scores)

    score_map = {
        "previous_lgbm": previous_scores,
        "lgbm_factor_feature": integrated_scores,
        "lgbm_factor_overlay": overlay_scores,
    }
    print("running common next-open portfolio replay", flush=True)
    results = {name: run_portfolio(history, scores) for name, scores in score_map.items()}
    periods = {name: summarize_result(result) for name, result in results.items()}
    prediction_map = {
        "previous_lgbm": previous_scores["score"],
        "lgbm_factor_feature": integrated_scores["score"],
        "lgbm_factor_overlay": overlay_scores["score"],
    }
    diagnostics = {
        name: prediction_metrics(frame, prediction, START, END)
        for name, prediction in prediction_map.items()
    }
    factor_diagnostics = {
        name: prediction_metrics(frame, frame[NEW_FACTOR], start, end)
        for name, (start, end) in period_windows().items()
    }

    curves = build_curves(results)
    OUTPUT_CURVES.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")
    render_svg(curves, OUTPUT_SVG)

    audits = {
        "alpha158_baseline": base_audit,
        "alpha158_plus_barra": barra_audit,
        "alpha158_plus_barra_plus_overnight": augmented_audit,
    }
    strict_checks = {name: maturity_checks(audit) for name, audit in audits.items()}
    all_strict = bool(strict_checks) and all(
        checks and all(checks.values()) for checks in strict_checks.values()
    )
    if not all_strict:
        raise RuntimeError("at least one monthly model used an immature training label")
    gate = {
        name: promotion_checks(periods[name], periods["previous_lgbm"])
        for name in ("lgbm_factor_feature", "lgbm_factor_overlay")
    }
    passed = [name for name, checks in gate.items() if all(checks.values())]
    if passed:
        best = max(
            passed,
            key=lambda name: periods[name]["full_2020_2026"]["performance"]["annual_return"],
        )
        decision = (
            f"{STRATEGY_LABELS[best]} passes every predefined historical gate, but it remains research-only because 2020-2026 has been repeatedly revealed."
        )
        status = "historical_gate_passed_research_only"
    else:
        best = "previous_lgbm"
        decision = (
            "Neither factor variant dominates the previous strategy on CAGR, Sharpe, drawdown, turnover and both subperiods. Keep the previous strategy pending a genuinely new forward window."
        )
        status = "factor_variants_rejected"

    gain_rows = augmented_audit.get("top_feature_importance", [])
    gain_match = next(
        (
            {**row, "top20_rank": index}
            for index, row in enumerate(gain_rows, start=1)
            if row["feature"] == NEW_FACTOR
        ),
        {"feature": NEW_FACTOR, "mean_gain": None, "top20_rank": None},
    )
    payload = _json_ready({
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "window": {"start": START, "end": END},
        "sample_status": "revealed historical diagnostic; not untouched out-of-sample",
        "strategies": {
            "previous_lgbm": "40% rules + 60% model; model = 75% Alpha158-lite + 25% Alpha158+Barra",
            "lgbm_factor_feature": f"previous architecture with {NEW_FACTOR} added inside the Barra LGBM branch",
            "lgbm_factor_overlay": f"85% previous final score + 15% positive rank({NEW_FACTOR}); weight frozen before this run",
        },
        "new_factor": {
            "name": NEW_FACTOR,
            **RETURN_FACTOR_SPECS[NEW_FACTOR],
            "direction": 1,
            "selection_origin": "2016-2017 discovery and 2018-2019 selection; 2020 veto was not used to retune the 15% weight",
        },
        "execution": {
            "signal": "T close",
            "fill": "T+1 open",
            "top_k": 5,
            "rebalance_sessions": REBALANCE_DAYS,
            "max_replacements": N_DROP,
            "one_way_cost_bps": COST_BPS,
        },
        "universe": {
            "snapshot_run_id": snapshot_id,
            "requested_codes": len(requested_codes),
            "loaded_codes": int(history["code"].nunique()),
            "warnings": warnings,
        },
        "training_audit": {
            "refit_months": base_audit["refit_months"],
            "all_strictly_mature": all_strict,
            "checks": strict_checks,
            "branches": audits,
        },
        "new_factor_model_gain": gain_match,
        "prediction_metrics": diagnostics,
        "factor_rank_ic_by_period": factor_diagnostics,
        "periods": periods,
        "historical_gate": gate,
        "selected_for_forward_research": best,
        "production_change": False,
        "decision": decision,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(BASE_DIR)),
            "report": str(OUTPUT_REPORT.relative_to(BASE_DIR)),
            "curve_csv": str(OUTPUT_CURVES.relative_to(BASE_DIR)),
            "curve_svg": str(OUTPUT_SVG.relative_to(BASE_DIR)),
        },
        "limitations": [
            "Current Top50 membership is backfilled and has survivorship and constituent bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "2020-2026 has been repeatedly revealed and cannot be used as a clean promotion holdout.",
            "A new forward paper-trading window is required before production promotion.",
        ],
    })
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_for_forward_research": best,
        "full_period": {
            name: periods[name]["full_2020_2026"] for name in STRATEGY_LABELS
        },
        "historical_gate": gate,
        "new_factor_model_gain": gain_match,
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
