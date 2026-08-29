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
from app.factor_mining import (
    FactorMiningConfig,
    MiningWindow,
    daily_rank_ic,
    discovery_feature_correlation,
    factor_diagnostics,
    select_stable_nonredundant_factors,
)
from app.lgbm_strategy import WINNER_CONFIG
from app.return_decomposition_factors import (
    SPECS,
    add_return_decomposition_factors,
)
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_new_factors_v1 import (
    build_extended_feature_frame,
    blend_model_predictions,
)
from research_qlib_lgbm_ranker_v1 import (
    baseline_scores,
    blend_scores,
    fit_expanding_lgbm_predictions,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


DISCOVERY_START = date(2016, 8, 25)
DISCOVERY_END = date(2017, 12, 31)
SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_START = date(2020, 1, 1)
DIAGNOSTIC_END = date(2020, 12, 31)
OVERLAY_WEIGHT = 0.15
CURRENT_BARRA_WEIGHT = 0.25

CONFIG = FactorMiningConfig(
    discovery=MiningWindow("discovery", DISCOVERY_START, DISCOVERY_END, "factor discovery"),
    selection=MiningWindow("selection", SELECTION_START, SELECTION_END, "candidate selection"),
    diagnostic=MiningWindow("diagnostic", DIAGNOSTIC_START, DIAGNOSTIC_END, "veto only"),
    min_cross_section=10,
    min_ic_days=126,
    min_abs_discovery_ic=0.01,
    min_direction_hit_rate=0.50,
    min_selection_retention=0.0,
    max_pair_correlation=0.80,
    max_factors=8,
)

OUTPUT_JSON = BASE_DIR / "data" / "annual_return_factor_mining_2016_2020.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "annual_return_factor_mining_2016_2020.md"
OUTPUT_CURVES = BASE_DIR / "reports" / "annual_return_factor_mining_2016_2020_curves.csv"
OUTPUT_SVG = BASE_DIR / "reports" / "annual_return_factor_mining_2016_2020.svg"


def _score_overlay(
    frame: pd.DataFrame,
    current_scores: pd.DataFrame,
    features: list[str],
    directions: dict[str, int],
) -> pd.DataFrame:
    signals = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce") * directions[feature]
        signals.append(values.groupby(frame["date"], sort=False).rank(pct=True))
    factor_score = pd.concat(signals, axis=1).mean(axis=1, skipna=True)
    output = current_scores.copy()
    output["score"] = output["score"].mul(1.0 - OVERLAY_WEIGHT).add(
        factor_score.mul(OVERLAY_WEIGHT)
    )
    output.loc[current_scores["score"].isna() | factor_score.isna(), "score"] = np.nan
    return output


def _candidate_groups(selected: list[str], diagnostics: pd.DataFrame) -> dict[str, list[str]]:
    groups = {f"single:{feature}": [feature] for feature in selected}
    for size in (2, 3):
        if len(selected) >= size:
            groups[f"top{size}_stable"] = selected[:size]
    if len(selected) >= 2:
        groups["all_stable"] = selected
    family_lookup = diagnostics.set_index("feature")["family"].to_dict()
    families: dict[str, list[str]] = {}
    for feature in selected:
        families.setdefault(str(family_lookup[feature]), []).append(feature)
    for family, features in families.items():
        if len(features) >= 2:
            groups[f"family:{family}"] = features
    unique = {}
    seen = set()
    for name, features in groups.items():
        key = tuple(features)
        if key not in seen:
            unique[name] = features
            seen.add(key)
    return unique


def _candidate_rank(row: dict) -> tuple:
    performance = row["selection"]["performance"]
    activity = row["selection"]["activity"]
    return (
        performance["annual_return"],
        performance["sharpe"],
        performance["max_drawdown"],
        -activity["annual_turnover"],
    )


def _selection_eligible(row: dict, current: dict) -> bool:
    candidate = row["selection"]
    return (
        candidate["performance"]["annual_return"]
        > current["performance"]["annual_return"]
        and candidate["performance"]["sharpe"]
        >= current["performance"]["sharpe"] * 0.90
        and candidate["performance"]["max_drawdown"]
        >= current["performance"]["max_drawdown"] - 0.03
        and candidate["activity"]["annual_turnover"]
        <= current["activity"]["annual_turnover"] * 1.10
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
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


def render_svg(curves: pd.DataFrame, path: Path) -> None:
    width, height = 1000, 580
    left, right, top, bottom = 80, 30, 70, 65
    chart_width, chart_height = width - left - right, height - top - bottom
    values = curves.to_numpy(dtype=float)
    low, high = min(0.90, float(values.min())), max(1.10, float(values.max()))
    padding = max((high - low) * 0.08, 0.05)
    low, high = max(0.0, low - padding), high + padding

    def x(index):
        return left + chart_width * index / max(len(curves) - 1, 1)

    def y(value):
        return top + chart_height * (high - value) / max(high - low, 1e-12)

    colors = {"current_lgbm": "#0B6E4F", "factor_overlay": "#C8553D"}
    labels = {"current_lgbm": "Current LGBM", "factor_overlay": "Factor overlay"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="80" y="31" font-family="Segoe UI, Arial" font-size="20" font-weight="600" fill="#1F2933">Annual-return factor overlay vs current LightGBM</text>',
        '<text x="80" y="50" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Fixed 15% factor overlay; Top5; next-open; 10-session rebalance; 12 bps one-way cost</text>',
    ]
    for tick in np.linspace(low, high, 6):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#E4E7EB"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{tick:.2f}x</text>')
    for column in curves:
        points = " ".join(f"{x(i):.2f},{y(float(v)):.2f}" for i, v in enumerate(curves[column]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.4" stroke-linejoin="round"/>')
    parts.append(f'<text x="{left}" y="{height-42}" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{pd.Timestamp(curves.index[0]).date()}</text>')
    parts.append(f'<text x="{width-right}" y="{height-42}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{pd.Timestamp(curves.index[-1]).date()}</text>')
    legend_x = left
    for column in curves:
        parts.append(f'<line x1="{legend_x}" y1="{height-16}" x2="{legend_x+24}" y2="{height-16}" stroke="{colors[column]}" stroke-width="3"/>')
        label = html.escape(f"{labels[column]} {float(curves[column].iloc[-1]):.2f}x")
        parts.append(f'<text x="{legend_x+31}" y="{height-12}" font-family="Segoe UI, Arial" font-size="11" fill="#334E68">{label}</text>')
        legend_x += 315
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(payload: dict) -> None:
    lines = [
        "# Annual-return factor mining, 2016-2020 dataset",
        "",
        f"- Status: `{payload['status']}`",
        f"- Candidate factors: {payload['candidate_factor_count']}; stable selected: {len(payload['selected_features'])}",
        f"- Overlay weight: {OVERLAY_WEIGHT:.0%}, fixed before evaluation.",
        "- Discovery: 2016-08-25 to 2017-12-31; selection: 2018-2019; 2020 is veto-only.",
        "- All periods were revealed previously and remain historical diagnostics.",
        "",
        "## Stable factors",
        "",
        "| Factor | Family | Discovery IC | Selection IC | Direction |",
        "|---|---|---:|---:|---:|",
    ]
    selected = set(payload["selected_features"])
    for row in payload["factor_diagnostics"]:
        if row["feature"] in selected:
            lines.append(
                f"| `{row['feature']}` | {row['family']} | {row['discovery_mean_rank_ic']:.4f} | "
                f"{row['selection_mean_rank_ic']:.4f} | {row['discovery_direction']:+d} |"
            )
    current_selection = payload["current_lgbm"]["selection"]
    current_diagnostic = payload["current_lgbm"]["diagnostic"]
    lines.extend(["", "## Best selection candidate", ""])
    winner = payload.get("winner")
    if winner:
        lines.extend([
            f"- Name: `{winner['name']}`",
            f"- Factors: {', '.join(f'`{item}`' for item in winner['features'])}",
            "",
            "| Window | Model | CAGR | Sharpe | Max drawdown | Annual turnover |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for window, current, candidate in (
            ("2018-2020 continuous", payload["current_lgbm"]["full_2018_2020"], winner["full_2018_2020"]),
            ("2018-2019 selection", current_selection, winner["selection"]),
            ("2020 veto diagnostic", current_diagnostic, winner["diagnostic"]),
        ):
            for label, block in (("Current LGBM", current), ("Factor overlay", candidate)):
                perf, activity = block["performance"], block["activity"]
                lines.append(
                    f"| {window} | {label} | {perf['annual_return']:.2%} | {perf['sharpe']:.3f} | "
                    f"{perf['max_drawdown']:.2%} | {activity['annual_turnover']:.2f}x |"
                )
    else:
        lines.append("No candidate increased selection CAGR while retaining the risk gates.")
    lines.extend([
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads((BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
    frame, feature_sets = build_extended_feature_frame(history)
    frame, candidate_features = add_return_decomposition_factors(frame)
    frame = frame.sort_values(["date", "code"]).reset_index(drop=True)

    print("screening candidate factor IC", flush=True)
    ic_rows = daily_rank_ic(frame, candidate_features, CONFIG.min_cross_section)
    diagnostics = factor_diagnostics(ic_rows, candidate_features, CONFIG)
    diagnostics["family"] = diagnostics["feature"].map(
        {feature: metadata["family"] for feature, metadata in SPECS.items()}
    )
    correlation = discovery_feature_correlation(frame, candidate_features, CONFIG.discovery)
    diagnostics, selected = select_stable_nonredundant_factors(diagnostics, correlation, CONFIG)
    directions = {
        str(row.feature): int(row.discovery_direction)
        for row in diagnostics.itertuples(index=False)
        if row.feature in selected
    }

    print("training current LGBM comparator", flush=True)
    base_prediction, base_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_baseline"],
        WINNER_CONFIG,
        SELECTION_START,
        DIAGNOSTIC_END,
    )
    barra_prediction, barra_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_plus_barra"],
        WINNER_CONFIG,
        SELECTION_START,
        DIAGNOSTIC_END,
    )
    current_prediction = blend_model_predictions(
        frame, base_prediction, barra_prediction, CURRENT_BARRA_WEIGHT
    )
    rules = baseline_scores(frame, SELECTION_START, DIAGNOSTIC_END)
    current_scores = blend_scores(
        frame, current_prediction, rules, WINNER_CONFIG["baseline_weight"]
    )
    current_selection_result = topk_dropout_backtest(
        history, current_scores, SELECTION_START, SELECTION_END
    )
    current_diagnostic_result = topk_dropout_backtest(
        history, current_scores, DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    current_selection = research_summary(current_selection_result, SELECTION_START, SELECTION_END)
    current_diagnostic = research_summary(current_diagnostic_result, DIAGNOSTIC_START, DIAGNOSTIC_END)

    candidate_rows = []
    score_cache = {}
    for name, features in _candidate_groups(selected, diagnostics).items():
        scores = _score_overlay(frame, current_scores, features, directions)
        score_cache[name] = scores
        result = topk_dropout_backtest(history, scores, SELECTION_START, SELECTION_END)
        summary = research_summary(result, SELECTION_START, SELECTION_END)
        candidate_rows.append({
            "name": name,
            "features": features,
            "selection": summary,
            "selection_gate": _selection_eligible({"selection": summary}, current_selection),
        })
    eligible = [row for row in candidate_rows if row["selection_gate"]]
    winner = max(eligible, key=_candidate_rank) if eligible else None
    winner_result_full = None
    if winner:
        winner_scores = score_cache[winner["name"]]
        diagnostic_result = topk_dropout_backtest(
            history, winner_scores, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        winner["diagnostic"] = research_summary(
            diagnostic_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        winner_result_full = topk_dropout_backtest(
            history, winner_scores, SELECTION_START, DIAGNOSTIC_END
        )
        diagnostic_checks = {
            "annual_return_higher": winner["diagnostic"]["performance"]["annual_return"]
            > current_diagnostic["performance"]["annual_return"],
            "sharpe_not_lower": winner["diagnostic"]["performance"]["sharpe"]
            >= current_diagnostic["performance"]["sharpe"],
            "drawdown_not_worse_by_3pp": winner["diagnostic"]["performance"]["max_drawdown"]
            >= current_diagnostic["performance"]["max_drawdown"] - 0.03,
            "turnover_not_higher_10pct": winner["diagnostic"]["activity"]["annual_turnover"]
            <= current_diagnostic["activity"]["annual_turnover"] * 1.10,
        }
        winner["diagnostic_gate"] = {
            "passed": all(diagnostic_checks.values()),
            "checks": diagnostic_checks,
        }
    passed = bool(winner and winner["diagnostic_gate"]["passed"])
    status = (
        "historical_candidate_passed_research_only"
        if passed
        else "selection_candidate_failed_2020_veto"
        if winner
        else "no_annual_return_candidate_found"
    )
    decision = (
        "A factor overlay increased annual return in both selection and the 2020 veto diagnostic, but it remains research-only because every window is already revealed."
        if passed
        else "The best factor overlay increased 2018-2019 annual return but failed the 2020 veto and must not be deployed."
        if winner
        else "No new factor overlay increased 2018-2019 annual return while retaining the predefined risk gates."
    )

    artifacts = {}
    current_full_summary = None
    if winner and winner_result_full is not None:
        current_full = topk_dropout_backtest(
            history, current_scores, SELECTION_START, DIAGNOSTIC_END
        )
        current_full_summary = research_summary(
            current_full, SELECTION_START, DIAGNOSTIC_END
        )
        winner["full_2018_2020"] = research_summary(
            winner_result_full, SELECTION_START, DIAGNOSTIC_END
        )
        curves = pd.concat({
            "current_lgbm": normalized_curve(current_full, "strategy_return"),
            "factor_overlay": normalized_curve(winner_result_full, "strategy_return"),
        }, axis=1, join="inner").dropna()
        if curves.empty or not np.allclose(curves.iloc[0].to_numpy(dtype=float), 1.0):
            raise RuntimeError("factor comparison curves do not share a normalized start")
        curves.index.name = "date"
        OUTPUT_CURVES.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")
        render_svg(curves, OUTPUT_SVG)
        artifacts = {
            "curve_csv": str(OUTPUT_CURVES.relative_to(BASE_DIR)),
            "curve_svg": str(OUTPUT_SVG.relative_to(BASE_DIR)),
        }

    candidate_rows.sort(key=_candidate_rank, reverse=True)
    payload = _json_ready({
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "objective": "find causal daily factors that increase current LGBM annual return",
        "sample_status": "2016-2020 has already been revealed; diagnostic only",
        "config": CONFIG.to_dict(),
        "overlay_weight": OVERLAY_WEIGHT,
        "candidate_factor_count": len(candidate_features),
        "candidate_factor_specs": SPECS,
        "selected_features": selected,
        "factor_diagnostics": diagnostics.to_dict(orient="records"),
        "directions": directions,
        "candidate_groups": candidate_rows,
        "winner": winner,
        "current_lgbm": {
            "architecture": "40% rule + 60% model; model is 75% Alpha158-lite + 25% Alpha158+Barra",
            "selection": current_selection,
            "diagnostic": current_diagnostic,
            "full_2018_2020": current_full_summary,
            "base_audit": base_audit,
            "barra_audit": barra_audit,
        },
        "decision": decision,
        "universe": {
            "snapshot_run_id": snapshot_id,
            "requested_codes": len(requested_codes),
            "loaded_codes_through_2020": int(history["code"].nunique()),
            "warnings": warnings,
        },
        "execution": "signal close; next-open; Top5; rebalance 10 sessions; one replacement; 12bp one-way",
        "artifacts": {
            **artifacts,
            "report": str(OUTPUT_REPORT.relative_to(BASE_DIR)),
        },
        "production_change": False,
        "limitations": [
            "Current Top50 membership is backfilled and has survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "Factor and overlay selection use the already revealed 2016-2019 history.",
            "The 2020 window is veto-only and cannot be used to change factors, directions, thresholds, or overlay weight.",
            "Any surviving factor still requires a newly frozen forward paper window before deployment.",
        ],
    })
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_features": payload["selected_features"],
        "winner": payload["winner"],
        "current_lgbm": {
            "selection": payload["current_lgbm"]["selection"],
            "diagnostic": payload["current_lgbm"]["diagnostic"],
        },
        "decision": payload["decision"],
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
