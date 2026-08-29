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
from app.double_ensemble import (
    DoubleEnsembleConfig,
    fit_expanding_double_ensemble_predictions,
)
from app.lgbm_strategy import WINNER_CONFIG
from app.master_reproduction import prediction_metrics
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_new_factors_v1 import (
    build_extended_feature_frame,
    blend_model_predictions,
    evaluate_predictions,
)
from research_qlib_lgbm_ranker_v1 import (
    fit_expanding_lgbm_predictions,
    normalized_curve,
)


DATA_START = date(2016, 8, 25)
EVALUATION_START = date(2018, 1, 1)
EVALUATION_END = date(2020, 12, 31)
CURRENT_BARRA_WEIGHT = 0.25
DOUBLE_ENSEMBLE_CONFIG = DoubleEnsembleConfig()

OUTPUT_JSON = BASE_DIR / "data" / "double_ensemble_vs_lgbm_2016_2020.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "double_ensemble_vs_lgbm_2016_2020.md"
OUTPUT_CURVES = BASE_DIR / "reports" / "double_ensemble_vs_lgbm_2016_2020_curves.csv"
OUTPUT_SVG = BASE_DIR / "reports" / "double_ensemble_vs_lgbm_2016_2020.svg"


def periods() -> dict[str, tuple[date, date]]:
    return {
        "full_2018_2020": (EVALUATION_START, EVALUATION_END),
        "2018_stress": (date(2018, 1, 1), date(2018, 12, 31)),
        "2019_recovery": (date(2019, 1, 1), date(2019, 12, 31)),
        "2020_diagnostic": (date(2020, 1, 1), date(2020, 12, 31)),
    }


def evaluate_all(history, frame, predictions) -> tuple[dict, dict]:
    summaries = {}
    results = {}
    for name, (start, end) in periods().items():
        summary, result = evaluate_predictions(history, frame, predictions, start, end)
        summaries[name] = summary
        results[name] = result
    return summaries, results


def render_svg(curves: pd.DataFrame, path: Path) -> None:
    width, height = 1000, 580
    left, right, top, bottom = 80, 30, 70, 65
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = curves.to_numpy(dtype=float)
    y_min = min(0.90, float(np.nanmin(values)))
    y_max = max(1.10, float(np.nanmax(values)))
    padding = max((y_max - y_min) * 0.08, 0.05)
    y_min, y_max = max(0.0, y_min - padding), y_max + padding

    def x_position(index: int) -> float:
        return left + chart_width * index / max(len(curves) - 1, 1)

    def y_position(value: float) -> float:
        return top + chart_height * (y_max - value) / max(y_max - y_min, 1e-12)

    colors = {
        "current_lgbm": "#0B6E4F",
        "double_ensemble": "#D1495B",
    }
    labels = {
        "current_lgbm": "Current LGBM",
        "double_ensemble": "DoubleEnsemble reproduction",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="80" y="31" font-family="Segoe UI, Arial" font-size="20" font-weight="600" fill="#1F2933">DoubleEnsemble vs current LightGBM</text>',
        '<text x="80" y="50" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">2016-2017 warm-up; 2018-2020 historical diagnostic; Top5; next-open; 12 bps one-way cost</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = y_position(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#E4E7EB" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{tick:.2f}x</text>'
        )
    for column in curves.columns:
        points = " ".join(
            f"{x_position(index):.2f},{y_position(float(value)):.2f}"
            for index, value in enumerate(curves[column])
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    start = pd.Timestamp(curves.index[0]).date().isoformat()
    end = pd.Timestamp(curves.index[-1]).date().isoformat()
    parts.extend([
        f'<text x="{left}" y="{height-42}" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{start}</text>',
        f'<text x="{width-right}" y="{height-42}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{end}</text>',
    ])
    legend_x = left
    for column in curves.columns:
        value = float(curves[column].iloc[-1])
        parts.append(
            f'<line x1="{legend_x}" y1="{height-16}" x2="{legend_x+24}" y2="{height-16}" stroke="{colors[column]}" stroke-width="3"/>'
        )
        text_value = html.escape(f"{labels[column]} {value:.2f}x")
        parts.append(
            f'<text x="{legend_x+31}" y="{height-12}" font-family="Segoe UI, Arial" font-size="11" fill="#334E68">{text_value}</text>'
        )
        legend_x += 315
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def write_report(payload: dict) -> None:
    lines = [
        "# DoubleEnsemble reproduction vs current LightGBM",
        "",
        f"- Status: `{payload['status']}`",
        f"- Available data: {payload['windows']['available_data'][0]} to {payload['windows']['available_data'][1]}",
        f"- Evaluation: {EVALUATION_START.isoformat()} to {EVALUATION_END.isoformat()}",
        "- Execution: close signal, next-open fill, Top5, 10-session rebalance, at most one replacement, 12 bps one-way cost.",
        "- The whole 2018-2020 window has already been revealed in prior research and is diagnostic, not a new blind holdout.",
        "",
        "| Period | Model | Terminal value | CAGR | Sharpe | Max drawdown | Annual turnover |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period_name in periods():
        for model_name, label in (
            ("current_lgbm", "Current LGBM"),
            ("double_ensemble", "DoubleEnsemble"),
        ):
            item = payload["periods"][period_name][model_name]
            performance = item["performance"]
            activity = item["activity"]
            lines.append(
                f"| {period_name} | {label} | {1 + performance['total_return']:.2f}x | "
                f"{format_percent(performance['annual_return'])} | {performance['sharpe']:.3f} | "
                f"{format_percent(performance['max_drawdown'])} | {activity['annual_turnover']:.2f}x |"
            )
    metrics = payload["prediction_metrics"]
    lines.extend([
        "",
        "## Prediction diagnostics",
        "",
        "| Model | Mean IC | ICIR | Mean RankIC | RankIC IR |",
        "|---|---:|---:|---:|---:|",
        f"| Current LGBM | {metrics['current_lgbm']['mean_ic']:.4f} | {metrics['current_lgbm']['ic_ir']:.4f} | {metrics['current_lgbm']['mean_rank_ic']:.4f} | {metrics['current_lgbm']['rank_ic_ir']:.4f} |",
        f"| DoubleEnsemble | {metrics['double_ensemble']['mean_ic']:.4f} | {metrics['double_ensemble']['ic_ir']:.4f} | {metrics['double_ensemble']['mean_rank_ic']:.4f} | {metrics['double_ensemble']['rank_ic_ir']:.4f} |",
        "",
        "## Reproduction scope",
        "",
        "- Paper: DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis (ICDM 2020).",
        "- Reproduced: sequential LightGBM ensemble, residual-driven sample reweighting, and dynamic feature selection.",
        "- Project adaptation: four CPU-scaled submodels, Alpha158-lite inputs, continuous date-wise return-rank regression, monthly expanding refits, and the project's 10-session next-open label.",
        "- Comparator: 40% rule protection plus 60% model rank; model rank is 75% Alpha158-lite and 25% Alpha158+Barra.",
        "- This is a structural reproduction under a common execution contract, not an exact replay of Qlib's official benchmark table.",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "Promotion checks: " + ", ".join(
            f"{name}={'pass' if passed else 'fail'}"
            for name, passed in payload["promotion_checks"].items()
        ),
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(
        (BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8")
    )
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(EVALUATION_END))].copy()
    available_start = pd.Timestamp(history["date"].min()).date()
    available_end = pd.Timestamp(history["date"].max()).date()
    frame, feature_sets = build_extended_feature_frame(history)
    frame = frame.sort_values(["date", "code"]).reset_index(drop=True)

    base_features = feature_sets["alpha158_baseline"]
    barra_features = feature_sets["alpha158_plus_barra"]
    print("training current Alpha158 LGBM", flush=True)
    base_prediction, base_audit = fit_expanding_lgbm_predictions(
        frame, base_features, WINNER_CONFIG, EVALUATION_START, EVALUATION_END
    )
    print("training current Alpha158+Barra LGBM", flush=True)
    barra_prediction, barra_audit = fit_expanding_lgbm_predictions(
        frame, barra_features, WINNER_CONFIG, EVALUATION_START, EVALUATION_END
    )
    current_prediction = blend_model_predictions(
        frame, base_prediction, barra_prediction, CURRENT_BARRA_WEIGHT
    )

    print("training DoubleEnsemble", flush=True)
    double_prediction, double_audit = fit_expanding_double_ensemble_predictions(
        frame,
        base_features,
        DOUBLE_ENSEMBLE_CONFIG,
        EVALUATION_START,
        EVALUATION_END,
    )
    if not double_audit["all_refits_strictly_mature"]:
        raise RuntimeError("DoubleEnsemble maturity audit failed")

    current_summaries, current_results = evaluate_all(
        history, frame, current_prediction
    )
    double_summaries, double_results = evaluate_all(
        history, frame, double_prediction
    )
    period_payload = {
        name: {
            "current_lgbm": current_summaries[name],
            "double_ensemble": double_summaries[name],
        }
        for name in periods()
    }
    prediction_payload = {
        "current_lgbm": prediction_metrics(
            frame, current_prediction, EVALUATION_START, EVALUATION_END
        ),
        "double_ensemble": prediction_metrics(
            frame, double_prediction, EVALUATION_START, EVALUATION_END
        ),
    }

    full_current = current_summaries["full_2018_2020"]
    full_double = double_summaries["full_2018_2020"]
    checks = {
        "annual_return_not_lower": full_double["performance"]["annual_return"]
        >= full_current["performance"]["annual_return"],
        "sharpe_not_lower": full_double["performance"]["sharpe"]
        >= full_current["performance"]["sharpe"],
        "drawdown_not_worse": full_double["performance"]["max_drawdown"]
        >= full_current["performance"]["max_drawdown"],
        "turnover_not_higher_10pct": full_double["activity"]["annual_turnover"]
        <= full_current["activity"]["annual_turnover"] * 1.10,
        "both_2018_and_2020_cagr_not_lower": (
            double_summaries["2018_stress"]["performance"]["annual_return"]
            >= current_summaries["2018_stress"]["performance"]["annual_return"]
            and double_summaries["2020_diagnostic"]["performance"]["annual_return"]
            >= current_summaries["2020_diagnostic"]["performance"]["annual_return"]
        ),
    }
    passed = all(checks.values())
    decision = (
        "DoubleEnsemble passes the revealed historical gates but remains research-only until a new forward paper window is completed."
        if passed
        else "DoubleEnsemble does not dominate the current LGBM architecture under the unchanged execution contract and must not replace it."
    )

    curves = pd.concat(
        {
            "current_lgbm": normalized_curve(
                current_results["full_2018_2020"], "strategy_return"
            ),
            "double_ensemble": normalized_curve(
                double_results["full_2018_2020"], "strategy_return"
            ),
        },
        axis=1,
        join="inner",
    ).dropna()
    if curves.empty or not np.allclose(curves.iloc[0].to_numpy(dtype=float), 1.0):
        raise RuntimeError("comparison curves do not share a normalized starting point")
    curves.index.name = "date"
    OUTPUT_CURVES.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")
    render_svg(curves, OUTPUT_SVG)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_gate_passed_research_only" if passed else "candidate_rejected",
        "paper": {
            "title": "DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis",
            "venue": "ICDM 2020",
            "official_framework": "https://github.com/microsoft/qlib",
            "reproduction_type": "CPU-scaled structural reproduction with execution-aligned labels",
        },
        "sample_status": "2018-2020 has already been revealed; diagnostic only, not untouched out-of-sample",
        "windows": {
            "requested_dataset": ["2016-01-01", "2020-12-31"],
            "available_data": [available_start.isoformat(), available_end.isoformat()],
            "warmup_and_initial_training": [available_start.isoformat(), "2017-12-31"],
            "historical_diagnostic": [EVALUATION_START.isoformat(), EVALUATION_END.isoformat()],
        },
        "universe": {
            "snapshot_run_id": snapshot_id,
            "requested_codes": len(requested_codes),
            "loaded_codes": int(history["code"].nunique()),
            "warnings": warnings,
        },
        "double_ensemble": {
            "feature_count": len(base_features),
            "config": DOUBLE_ENSEMBLE_CONFIG.to_dict(),
            "audit": double_audit,
        },
        "current_lgbm": {
            "architecture": "40% rule protection + 60% model rank; model rank is 75% Alpha158-lite + 25% Alpha158+Barra",
            "config": WINNER_CONFIG,
            "base_feature_count": len(base_features),
            "barra_feature_count": len(barra_features),
            "base_audit": base_audit,
            "barra_audit": barra_audit,
        },
        "execution": "signal close; buy next open; Top5; 10-session rebalance; one replacement; 12bp one-way cost",
        "prediction_metrics": prediction_payload,
        "periods": period_payload,
        "promotion_checks": checks,
        "decision": decision,
        "artifacts": {
            "curve_csv": str(OUTPUT_CURVES.relative_to(BASE_DIR)),
            "curve_svg": str(OUTPUT_SVG.relative_to(BASE_DIR)),
            "report": str(OUTPUT_REPORT.relative_to(BASE_DIR)),
        },
        "limitations": [
            "Current Top50 membership is backfilled and has survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "The available common history starts in August 2016, so 2016-2017 is required for feature warm-up and initial training rather than portfolio evaluation.",
            "The official Qlib DoubleEnsemble implementation and this reproduction differ in CPU scale, label horizon, and execution contract.",
            "The complete 2018-2020 window was already revealed in prior research and cannot promote a production model.",
            "A newly frozen forward paper window is required before deployment.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(payload)
    print(json.dumps({
        "status": payload["status"],
        "sample_status": payload["sample_status"],
        "periods": payload["periods"],
        "prediction_metrics": payload["prediction_metrics"],
        "promotion_checks": payload["promotion_checks"],
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
