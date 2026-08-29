from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from app.master_reproduction import (
    LightweightAdam,
    build_master_panel_from_frame,
    make_master_model,
    predict_master,
    prediction_metrics,
    train_master_epoch,
)
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_new_factors_v1 import (
    build_extended_feature_frame,
    blend_model_predictions,
    evaluate_predictions,
    fit_frozen_rank_predictions,
)
from research_qlib_ridge_topk_v1 import normalized_curve


SELECTION_TRAIN_CUTOFF = date(2019, 1, 1)
SELECTION_START = date(2019, 1, 1)
SELECTION_END = date(2019, 12, 31)
FREEZE_DATE = date(2020, 1, 2)
DIAGNOSTIC_START = date(2020, 1, 2)
DIAGNOSTIC_END = date(2026, 8, 25)
SEED = 20260829
MAX_SELECTION_EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 1e-4
D_MODEL = 64
DROPOUT = 0.5
BETA = 5.0
CURRENT_CANDIDATE_WEIGHT = 0.25

OUTPUT_JSON = BASE_DIR / "data" / "master_vs_lgbm_reproduction_v1.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "master_vs_lgbm_reproduction_v1.md"
OUTPUT_CURVES = BASE_DIR / "reports" / "master_vs_lgbm_reproduction_v1_curves.csv"
OUTPUT_SVG = BASE_DIR / "reports" / "master_vs_lgbm_reproduction_v1.svg"
MODEL_DIR = BASE_DIR / "data" / "models" / "master_reproduction_v1"


def _selection_score(metrics: dict) -> float:
    value = float(metrics.get("mean_rank_ic", np.nan))
    return value if np.isfinite(value) else -np.inf


def select_epoch(panel) -> tuple[int, list[dict]]:
    model = make_master_model(
        len(panel.feature_columns),
        len(panel.market_columns),
        seed=SEED,
        d_model=D_MODEL,
        dropout=DROPOUT,
        beta=BETA,
    )
    optimizer = LightweightAdam(model.parameters(), lr=LEARNING_RATE)
    train_days = panel.training_date_indices(SELECTION_TRAIN_CUTOFF)
    validation_days = panel.date_indices(SELECTION_START, SELECTION_END)
    history = []
    best_epoch = 1
    best_score = -np.inf
    best_state = None
    stale = 0
    for epoch in range(1, MAX_SELECTION_EPOCHS + 1):
        loss = train_master_epoch(
            model,
            panel,
            train_days,
            optimizer,
            label_cutoff=SELECTION_TRAIN_CUTOFF,
            seed=SEED + epoch,
        )
        predictions = predict_master(model, panel, validation_days)
        metrics = prediction_metrics(
            panel.frame,
            predictions,
            SELECTION_START,
            SELECTION_END,
            label_end_before=FREEZE_DATE,
        )
        score = _selection_score(metrics)
        history.append({
            "epoch": epoch,
            "training_loss": loss,
            "validation": metrics,
        })
        print(
            f"selection epoch={epoch} loss={loss:.6f} "
            f"rank_ic={metrics['mean_rank_ic']:.6f}",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("MASTER epoch selection produced no valid checkpoint")
    return best_epoch, history


def fit_final_master(panel, epochs: int):
    model = make_master_model(
        len(panel.feature_columns),
        len(panel.market_columns),
        seed=SEED,
        d_model=D_MODEL,
        dropout=DROPOUT,
        beta=BETA,
    )
    optimizer = LightweightAdam(model.parameters(), lr=LEARNING_RATE)
    train_days = panel.training_date_indices(FREEZE_DATE)
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train_master_epoch(
            model,
            panel,
            train_days,
            optimizer,
            label_cutoff=FREEZE_DATE,
            seed=SEED + 100 + epoch,
        )
        losses.append(loss)
        print(f"final epoch={epoch}/{epochs} loss={loss:.6f}", flush=True)
    return model, train_days, losses


def _subperiods() -> dict[str, tuple[date, date]]:
    return {
        "full_2020_2026": (DIAGNOSTIC_START, DIAGNOSTIC_END),
        "2020_2023": (DIAGNOSTIC_START, date(2023, 12, 31)),
        "2024_2026": (date(2024, 1, 1), DIAGNOSTIC_END),
    }


def evaluate_all_periods(history, frame, predictions) -> tuple[dict, dict]:
    summaries = {}
    results = {}
    for name, (start, end) in _subperiods().items():
        summary, result = evaluate_predictions(history, frame, predictions, start, end)
        summaries[name] = summary
        results[name] = result
    return summaries, results


def render_svg(curves: pd.DataFrame, path: Path) -> None:
    width, height = 1000, 580
    left, right, top, bottom = 82, 30, 54, 74
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = curves[["lgbm_current_architecture", "master_reproduction"]]
    y_min = min(0.9, float(values.min().min()))
    y_max = max(1.1, float(values.max().max()))
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_position(index: int) -> float:
        return left + chart_width * index / max(len(curves) - 1, 1)

    def y_position(value: float) -> float:
        return top + chart_height * (y_max - value) / (y_max - y_min)

    colors = {
        "lgbm_current_architecture": "#176B87",
        "master_reproduction": "#C2413B",
    }
    labels = {
        "lgbm_current_architecture": "Current LGBM architecture",
        "master_reproduction": "MASTER reproduction",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        '<text x="82" y="31" font-family="Segoe UI, Arial" font-size="20" font-weight="600" fill="#1F2933">MASTER vs current LightGBM architecture</text>',
        '<text x="82" y="49" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Frozen before 2020; Top5; 10-session rebalance; next-open; 12 bps one-way cost</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_position(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#D9E2E8" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{value:.2f}x</text>')
    for column in colors:
        points = " ".join(
            f"{x_position(index):.2f},{y_position(float(value)):.2f}"
            for index, value in enumerate(curves[column])
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
    start = pd.Timestamp(curves.index.min()).date().isoformat()
    end = pd.Timestamp(curves.index.max()).date().isoformat()
    parts.append(f'<text x="{left}" y="{height-43}" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{html.escape(start)}</text>')
    parts.append(f'<text x="{width-right}" y="{height-43}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{html.escape(end)}</text>')
    legend_x = left
    for column in colors:
        parts.append(f'<line x1="{legend_x}" y1="{height-19}" x2="{legend_x+24}" y2="{height-19}" stroke="{colors[column]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+31}" y="{height-15}" font-family="Segoe UI, Arial" font-size="11" fill="#334E68">{labels[column]} {float(curves[column].iloc[-1]):.2f}x</text>')
        legend_x += 285
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def write_report(payload: dict) -> None:
    full = payload["periods"]["full_2020_2026"]
    lgbm = full["lgbm_current_architecture"]["performance"]
    master = full["master_reproduction"]["performance"]
    prediction = payload["prediction_metrics"]
    lines = [
        "# MASTER structural reproduction vs current LightGBM architecture",
        "",
        f"- Status: `{payload['status']}`",
        f"- Window: {DIAGNOSTIC_START.isoformat()} to {DIAGNOSTIC_END.isoformat()}",
        "- Execution: signal at close, trade next open, Top5, rebalance every 10 sessions, 12 bps one-way cost.",
        "- Both models are frozen before 2020. This window has already been revealed and is diagnostic, not a new untouched holdout.",
        "",
        "| Model | Terminal value | CAGR | Sharpe | Max drawdown | Annual turnover |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current LGBM architecture | {1 + lgbm['total_return']:.2f}x | {_format_percent(lgbm['annual_return'])} | {lgbm['sharpe']:.3f} | {_format_percent(lgbm['max_drawdown'])} | {full['lgbm_current_architecture']['activity']['annual_turnover']:.2f}x |",
        f"| MASTER reproduction | {1 + master['total_return']:.2f}x | {_format_percent(master['annual_return'])} | {master['sharpe']:.3f} | {_format_percent(master['max_drawdown'])} | {full['master_reproduction']['activity']['annual_turnover']:.2f}x |",
        "",
        "## Subperiod stability",
        "",
        "| Period | Model | CAGR | Sharpe | Max drawdown |",
        "|---|---|---:|---:|---:|",
    ]
    for period_name in ("2020_2023", "2024_2026"):
        for model_name, label in (
            ("lgbm_current_architecture", "Current LGBM architecture"),
            ("master_reproduction", "MASTER reproduction"),
        ):
            metrics = payload["periods"][period_name][model_name]["performance"]
            lines.append(
                f"| {period_name.replace('_', '-')} | {label} | "
                f"{_format_percent(metrics['annual_return'])} | {metrics['sharpe']:.3f} | "
                f"{_format_percent(metrics['max_drawdown'])} |"
            )
    lines.extend([
        "",
        "## Prediction diagnostics",
        "",
        "| Model | Mean IC | ICIR | Mean RankIC | RankIC IR |",
        "|---|---:|---:|---:|---:|",
        f"| Current LGBM architecture | {prediction['lgbm_current_architecture']['mean_ic']:.4f} | {prediction['lgbm_current_architecture']['ic_ir']:.4f} | {prediction['lgbm_current_architecture']['mean_rank_ic']:.4f} | {prediction['lgbm_current_architecture']['rank_ic_ir']:.4f} |",
        f"| MASTER reproduction | {prediction['master_reproduction']['mean_ic']:.4f} | {prediction['master_reproduction']['ic_ir']:.4f} | {prediction['master_reproduction']['mean_rank_ic']:.4f} | {prediction['master_reproduction']['rank_ic_ir']:.4f} |",
        "",
        "## Reproduction scope",
        "",
        "- Reproduced: market-guided feature gate, intra-stock temporal attention, inter-stock same-date attention, temporal aggregation, and cross-sectional normalized return training.",
        f"- Scaled: 49 loaded stocks, {payload['model']['feature_count']} Alpha158-lite features, 63 causal market-proxy features, d_model={D_MODEL}; the paper used CSI300/CSI800, 158 stock features and d_model=256.",
        "- Market proxies: nested liquidity universes replace unavailable point-in-time CSI300/CSI500/CSI800 histories.",
        "- The official repository states that the confidential original business code is unavailable and that previously published validation data had a processing flaw. This is therefore a structural reproduction, not an exact benchmark reproduction.",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "Promotion checks: "
        + ", ".join(
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
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    source = json.loads((BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
    extended_frame, feature_sets = build_extended_feature_frame(history)
    frame = extended_frame.sort_values(["date", "code"]).reset_index(drop=True)
    panel, market_scaler = build_master_panel_from_frame(
        frame,
        feature_sets["alpha158_baseline"],
        training_end_exclusive=FREEZE_DATE,
    )
    frame = panel.frame

    best_epoch, epoch_history = select_epoch(panel)
    master_model, master_training_days, final_losses = fit_final_master(panel, best_epoch)
    prediction_days = panel.date_indices(DIAGNOSTIC_START, DIAGNOSTIC_END)
    master_predictions = predict_master(master_model, panel, prediction_days)

    alpha158_predictions, alpha158_audit = fit_frozen_rank_predictions(
        frame, feature_sets["alpha158_baseline"], FREEZE_DATE
    )
    barra_predictions, barra_audit = fit_frozen_rank_predictions(
        frame, feature_sets["alpha158_plus_barra"], FREEZE_DATE
    )
    lgbm_predictions = blend_model_predictions(
        frame,
        alpha158_predictions,
        barra_predictions,
        CURRENT_CANDIDATE_WEIGHT,
    )

    master_summaries, master_results = evaluate_all_periods(
        history, frame, master_predictions
    )
    lgbm_summaries, lgbm_results = evaluate_all_periods(
        history, frame, lgbm_predictions
    )
    periods = {
        name: {
            "lgbm_current_architecture": lgbm_summaries[name],
            "master_reproduction": master_summaries[name],
        }
        for name in _subperiods()
    }
    master_prediction_metrics = prediction_metrics(
        frame, master_predictions, DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    lgbm_prediction_metrics = prediction_metrics(
        frame, lgbm_predictions, DIAGNOSTIC_START, DIAGNOSTIC_END
    )

    full_master = master_summaries["full_2020_2026"]
    full_lgbm = lgbm_summaries["full_2020_2026"]
    checks = {
        "annual_return_not_lower": full_master["performance"]["annual_return"]
        >= full_lgbm["performance"]["annual_return"],
        "sharpe_not_lower": full_master["performance"]["sharpe"]
        >= full_lgbm["performance"]["sharpe"],
        "drawdown_not_worse": full_master["performance"]["max_drawdown"]
        >= full_lgbm["performance"]["max_drawdown"],
        "turnover_not_higher_10pct": full_master["activity"]["annual_turnover"]
        <= full_lgbm["activity"]["annual_turnover"] * 1.10,
    }
    passed = all(checks.values())
    decision = (
        "MASTER passes the historical diagnostic gates, but remains research-only until a new forward window is completed."
        if passed
        else "MASTER does not dominate the current LightGBM architecture under the unchanged execution contract and must not replace it."
    )

    curves = pd.concat(
        {
            "lgbm_current_architecture": normalized_curve(
                lgbm_results["full_2020_2026"], "strategy_return"
            ),
            "master_reproduction": normalized_curve(
                master_results["full_2020_2026"], "strategy_return"
            ),
        },
        axis=1,
        join="inner",
    ).dropna()
    curves.index.name = "date"
    OUTPUT_CURVES.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")
    render_svg(curves, OUTPUT_SVG)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(master_model.state_dict(), MODEL_DIR / "model.pt")
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_gate_passed_research_only" if passed else "candidate_rejected",
        "paper": {
            "title": "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting",
            "doi": "10.1609/aaai.v38i1.27767",
            "official_repository": "https://github.com/SJTU-DMTai/MASTER",
            "reproduction_type": "scaled structural reproduction on the project's frozen Top50 history",
        },
        "sample_status": "2020-2026 has already been revealed; diagnostic only, not untouched out-of-sample",
        "universe": {
            "snapshot_run_id": snapshot_id,
            "requested_codes": len(requested_codes),
            "loaded_codes": int(history["code"].nunique()),
            "warnings": warnings,
        },
        "model": {
            "seed": SEED,
            "lookback": panel.lookback,
            "feature_count": len(panel.feature_columns),
            "market_feature_count": len(panel.market_columns),
            "d_model": D_MODEL,
            "dropout": DROPOUT,
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "selected_epochs": best_epoch,
            "selection_epoch_history": epoch_history,
            "final_training_dates": len(master_training_days),
            "final_training_loss": final_losses,
            "market_scaler_fit_before": FREEZE_DATE.isoformat(),
            "market_scaler": market_scaler,
        },
        "lgbm_comparator": {
            "architecture": "40% rule protection + 60% model rank; model rank is 75% Alpha158-lite + 25% Alpha158+Barra",
            "freeze_date": FREEZE_DATE.isoformat(),
            "config": WINNER_CONFIG,
            "alpha158_audit": alpha158_audit,
            "barra_audit": barra_audit,
        },
        "execution": "signal close; buy next open; Top5; 10-session rebalance; one replacement; 12bp one-way cost",
        "prediction_metrics": {
            "lgbm_current_architecture": lgbm_prediction_metrics,
            "master_reproduction": master_prediction_metrics,
        },
        "periods": periods,
        "promotion_checks": checks,
        "decision": decision,
        "artifacts": {
            "curve_csv": str(OUTPUT_CURVES.relative_to(BASE_DIR)),
            "curve_svg": str(OUTPUT_SVG.relative_to(BASE_DIR)),
            "report": str(OUTPUT_REPORT.relative_to(BASE_DIR)),
            "model": str((MODEL_DIR / "model.pt").relative_to(BASE_DIR)),
        },
        "limitations": [
            "Current Top50 membership is backfilled and has survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "The paper used CSI300/CSI800 and 158 features; this run has 49 stocks and 85 Alpha158-lite features.",
            "Nested liquidity proxies replace unavailable point-in-time CSI index features.",
            "The paper used d_model=256; this CPU-scaled reproduction uses d_model=64.",
            "A new forward paper window is required before any production promotion.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_report(payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_epochs": best_epoch,
        "prediction_metrics": payload["prediction_metrics"],
        "full_period": payload["periods"]["full_2020_2026"],
        "promotion_checks": checks,
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
