from __future__ import annotations

from copy import deepcopy
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

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_multi_objective_v1 import (
    CANDIDATES as PORTFOLIO_CANDIDATES,
    multihead_portfolio_backtest,
)
from research_lgbm_new_factors_v1 import (
    blend_model_predictions,
    build_extended_feature_frame,
    fit_frozen_rank_predictions,
)
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    normalized_curve,
    research_summary,
)


SELECTION_FREEZE = date(2018, 1, 1)
SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_START = date(2020, 1, 2)
EARLY_END = date(2023, 12, 31)
DIAGNOSTIC_END = date(2026, 8, 25)

MODEL_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
RULE_WEIGHTS = (0.0, 0.20, 0.40, 0.60)
TOP_K_VALUES = (3, 5, 8, 10)
REBALANCE_VALUES = (5, 10, 15, 20)
N_DROP_VALUES = (1, 2)
TOP_SCORE_CONFIGS = 5
COST_BPS = 12.0

CURRENT_PARAMS = {
    "model_weight": 0.25,
    "rule_weight": 0.40,
    "top_k": 5,
    "rebalance_days": 10,
    "n_drop": 1,
}

OUTPUT_JSON = BASE_DIR / "data" / "lgbm_early_excess_v1.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "lgbm_early_excess_v1.md"
OUTPUT_CURVES = BASE_DIR / "reports" / "lgbm_early_excess_v1_curves.csv"
BENCHMARK_CURVES = (
    BASE_DIR / "reports" / "current_best_lgbm_vs_csi300_csi2000_2020_now.csv"
)


def portfolio_config(top_k: int) -> dict:
    config = deepcopy(PORTFOLIO_CANDIDATES["current_rank_top5"])
    config["max_k"] = int(top_k)
    config["min_k"] = int(top_k)
    return config


def run_portfolio(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    top_k: int,
    rebalance_days: int,
    n_drop: int,
    start: date,
    end: date,
) -> dict:
    prepared = scores.copy()
    prepared["expected_return"] = 0.0
    prepared["downside_probability"] = 0.0
    return multihead_portfolio_backtest(
        history,
        prepared,
        portfolio_config(top_k),
        start,
        end,
        rebalance_days=rebalance_days,
        n_drop=n_drop,
        cost_bps=COST_BPS,
    )


def summary_objective(full: dict, first: dict, second: dict) -> float:
    perf = full["performance"]
    activity = full["activity"]
    first_excess = first["performance"]["annual_return"] - first["performance"][
        "benchmark_return"
    ]
    second_excess = second["performance"]["annual_return"] - second[
        "performance"
    ]["benchmark_return"]
    worst_year_excess = min(first_excess, second_excess)
    return float(
        perf["annual_return"]
        + 0.25 * worst_year_excess
        + 0.05 * perf["sharpe"]
        + 0.05 * perf["max_drawdown"]
        - 0.002 * activity["annual_turnover"]
    )


def summarize_selection(result: dict) -> dict:
    full = research_summary(result, SELECTION_START, SELECTION_END)
    first = research_summary(result, SELECTION_START, date(2018, 12, 31))
    second = research_summary(result, date(2019, 1, 1), SELECTION_END)
    return {
        "full": full,
        "2018": first,
        "2019": second,
        "objective": summary_objective(full, first, second),
    }


def build_scores(
    frame: pd.DataFrame,
    base_prediction: pd.Series,
    barra_prediction: pd.Series,
    rule_score: pd.Series,
    model_weight: float,
    rule_weight: float,
) -> pd.DataFrame:
    model_prediction = blend_model_predictions(
        frame, base_prediction, barra_prediction, model_weight
    )
    return blend_scores(frame, model_prediction, rule_score, rule_weight)


def select_parameters(
    history: pd.DataFrame,
    frame: pd.DataFrame,
    base_prediction: pd.Series,
    barra_prediction: pd.Series,
) -> tuple[dict, list[dict], list[dict]]:
    rule_score = baseline_scores(frame, SELECTION_START, SELECTION_END)
    score_rows = []
    current_portfolio = CURRENT_PARAMS

    for model_weight in MODEL_WEIGHTS:
        for rule_weight in RULE_WEIGHTS:
            scores = build_scores(
                frame,
                base_prediction,
                barra_prediction,
                rule_score,
                model_weight,
                rule_weight,
            )
            result = run_portfolio(
                history,
                scores,
                top_k=current_portfolio["top_k"],
                rebalance_days=current_portfolio["rebalance_days"],
                n_drop=current_portfolio["n_drop"],
                start=SELECTION_START,
                end=SELECTION_END,
            )
            score_rows.append(
                {
                    "model_weight": model_weight,
                    "rule_weight": rule_weight,
                    "selection": summarize_selection(result),
                }
            )

    top_score_rows = sorted(
        score_rows, key=lambda row: row["selection"]["objective"], reverse=True
    )[:TOP_SCORE_CONFIGS]
    portfolio_rows = []
    for score_row in top_score_rows:
        scores = build_scores(
            frame,
            base_prediction,
            barra_prediction,
            rule_score,
            score_row["model_weight"],
            score_row["rule_weight"],
        )
        for top_k in TOP_K_VALUES:
            for rebalance_days in REBALANCE_VALUES:
                for n_drop in N_DROP_VALUES:
                    if n_drop >= top_k:
                        continue
                    result = run_portfolio(
                        history,
                        scores,
                        top_k=top_k,
                        rebalance_days=rebalance_days,
                        n_drop=n_drop,
                        start=SELECTION_START,
                        end=SELECTION_END,
                    )
                    portfolio_rows.append(
                        {
                            "model_weight": score_row["model_weight"],
                            "rule_weight": score_row["rule_weight"],
                            "top_k": top_k,
                            "rebalance_days": rebalance_days,
                            "n_drop": n_drop,
                            "selection": summarize_selection(result),
                        }
                    )

    winner = max(
        portfolio_rows, key=lambda row: row["selection"]["objective"]
    )
    return winner, score_rows, portfolio_rows


def strategy_periods(result: dict) -> dict:
    periods = {
        "full_2020_2026": (DIAGNOSTIC_START, DIAGNOSTIC_END),
        "early_2020_2023": (DIAGNOSTIC_START, EARLY_END),
        "late_2024_2026": (date(2024, 1, 1), DIAGNOSTIC_END),
    }
    for year in range(2020, 2027):
        periods[str(year)] = (
            max(DIAGNOSTIC_START, date(year, 1, 1)),
            min(DIAGNOSTIC_END, date(year, 12, 31)),
        )
    return {
        name: research_summary(result, start, end)
        for name, (start, end) in periods.items()
        if start <= end
    }


def curve_metrics(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        raise ValueError("curve requires at least two observations")
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / (len(clean) - 1)) - 1.0)
    max_drawdown = float(clean.div(clean.cummax()).sub(1.0).min())
    return {
        "start": clean.index[0].date().isoformat(),
        "end": clean.index[-1].date().isoformat(),
        "observations": int(len(clean)),
        "terminal_value": float(clean.iloc[-1] / clean.iloc[0]),
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
    }


def benchmark_comparison(curves: pd.DataFrame) -> dict:
    benchmarks = pd.read_csv(BENCHMARK_CURVES, parse_dates=["date"]).set_index("date")
    benchmark_columns = ["csi2000", "csi300"]
    joined = curves[["current_lgbm", "optimized_lgbm"]].join(
        benchmarks[benchmark_columns], how="inner"
    ).dropna()
    windows = {
        "early_2020_2023": (pd.Timestamp(DIAGNOSTIC_START), pd.Timestamp(EARLY_END)),
        "late_2024_2026": (pd.Timestamp("2024-01-01"), pd.Timestamp(DIAGNOSTIC_END)),
        "full_2020_2026": (pd.Timestamp(DIAGNOSTIC_START), pd.Timestamp(DIAGNOSTIC_END)),
    }
    output = {}
    for name, (start, end) in windows.items():
        window = joined.loc[start:end]
        metrics = {column: curve_metrics(window[column]) for column in joined.columns}
        for strategy in ("current_lgbm", "optimized_lgbm"):
            metrics[strategy]["annual_excess_vs_csi2000"] = (
                metrics[strategy]["annual_return"] - metrics["csi2000"]["annual_return"]
            )
            metrics[strategy]["annual_excess_vs_csi300"] = (
                metrics[strategy]["annual_return"] - metrics["csi300"]["annual_return"]
            )
        output[name] = metrics
    return output


def promotion_checks(comparison: dict) -> dict[str, bool]:
    early = comparison["early_2020_2023"]
    late = comparison["late_2024_2026"]
    full = comparison["full_2020_2026"]
    current_early = early["current_lgbm"]
    candidate_early = early["optimized_lgbm"]
    return {
        "early_cagr_improves_1pp": (
            candidate_early["annual_return"] >= current_early["annual_return"] + 0.01
        ),
        "early_excess_vs_csi2000_improves_1pp": (
            candidate_early["annual_excess_vs_csi2000"]
            >= current_early["annual_excess_vs_csi2000"] + 0.01
        ),
        "late_cagr_retains_90pct": (
            late["optimized_lgbm"]["annual_return"]
            >= late["current_lgbm"]["annual_return"] * 0.90
        ),
        "full_drawdown_not_worse_3pp": (
            full["optimized_lgbm"]["max_drawdown"]
            >= full["current_lgbm"]["max_drawdown"] - 0.03
        ),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_report(payload: dict) -> None:
    winner = payload["selected_parameters"]
    comparison = payload["benchmark_comparison"]
    early = comparison["early_2020_2023"]
    late = comparison["late_2024_2026"]
    full = comparison["full_2020_2026"]
    status = "historical candidate only" if payload["promotion_gate"]["passed"] else "rejected"
    lines = [
        "# LGBM early-excess optimization v1",
        "",
        f"- Status: {status}; production unchanged.",
        "- Parameters selected only on frozen 2018-2019 data.",
        "- 2020-2026 is diagnostic-only and was not used to choose parameters.",
        f"- Selected: Barra model weight {winner['model_weight']:.0%}, rule weight {winner['rule_weight']:.0%}, Top{winner['top_k']}, rebalance {winner['rebalance_days']} sessions, n_drop {winner['n_drop']}.",
        "- Signal at close, execution at next open, 12 bps one-way cost.",
        "",
        "## Benchmark comparison",
        "",
        "| Window | Strategy | CAGR | Max drawdown | Excess vs CSI2000 | Excess vs CSI300 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for window_name, block in (("2020-2023", early), ("2024-2026", late), ("2020-2026", full)):
        for strategy, label in (("current_lgbm", "Current 75/25"), ("optimized_lgbm", "Optimized")):
            row = block[strategy]
            lines.append(
                f"| {window_name} | {label} | {row['annual_return']:.2%} | {row['max_drawdown']:.2%} | "
                f"{row['annual_excess_vs_csi2000']:.2%} | {row['annual_excess_vs_csi300']:.2%} |"
            )
    lines.extend([
        "",
        "## Gate",
        "",
    ])
    for name, passed in payload["promotion_gate"]["checks"].items():
        lines.append(f"- {name}: {'pass' if passed else 'fail'}")
    lines.extend([
        "",
        "## Audit",
        "",
        f"- Base expanding refits: {payload['training_audit']['base_refits']}",
        f"- Barra expanding refits: {payload['training_audit']['barra_refits']}",
        f"- All labels strictly mature: {payload['training_audit']['all_strictly_mature']}",
        "- Current Top50 universe is backfilled and has constituent/survivorship bias.",
        "- Current-vintage adjusted prices are not strict point-in-time data.",
        "- A new forward paper window is still required before deployment.",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
    frame, feature_sets = build_extended_feature_frame(history)
    frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    print("fit frozen 2018 selection models", flush=True)
    frozen_base, frozen_base_audit = fit_frozen_rank_predictions(
        frame, feature_sets["alpha158_baseline"], SELECTION_FREEZE
    )
    frozen_barra, frozen_barra_audit = fit_frozen_rank_predictions(
        frame, feature_sets["alpha158_plus_barra"], SELECTION_FREEZE
    )
    winner, score_rows, portfolio_rows = select_parameters(
        history, frame, frozen_base, frozen_barra
    )
    selected = {key: winner[key] for key in CURRENT_PARAMS}
    print(f"selected parameters: {selected}", flush=True)

    print("fit expanding 2020-2026 base model", flush=True)
    expanding_base, base_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_baseline"],
        WINNER_CONFIG,
        DIAGNOSTIC_START,
        DIAGNOSTIC_END,
    )
    print("fit expanding 2020-2026 Barra model", flush=True)
    expanding_barra, barra_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_plus_barra"],
        WINNER_CONFIG,
        DIAGNOSTIC_START,
        DIAGNOSTIC_END,
    )
    diagnostic_rule = baseline_scores(frame, DIAGNOSTIC_START, DIAGNOSTIC_END)

    current_scores = build_scores(
        frame,
        expanding_base,
        expanding_barra,
        diagnostic_rule,
        CURRENT_PARAMS["model_weight"],
        CURRENT_PARAMS["rule_weight"],
    )
    optimized_scores = build_scores(
        frame,
        expanding_base,
        expanding_barra,
        diagnostic_rule,
        selected["model_weight"],
        selected["rule_weight"],
    )
    current_result = run_portfolio(
        history,
        current_scores,
        top_k=CURRENT_PARAMS["top_k"],
        rebalance_days=CURRENT_PARAMS["rebalance_days"],
        n_drop=CURRENT_PARAMS["n_drop"],
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
    )
    optimized_result = run_portfolio(
        history,
        optimized_scores,
        top_k=selected["top_k"],
        rebalance_days=selected["rebalance_days"],
        n_drop=selected["n_drop"],
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
    )

    curves = pd.concat(
        {
            "current_lgbm": normalized_curve(current_result, "strategy_return"),
            "optimized_lgbm": normalized_curve(optimized_result, "strategy_return"),
        },
        axis=1,
        join="inner",
    ).dropna()
    curves["current_drawdown"] = curves["current_lgbm"].div(
        curves["current_lgbm"].cummax()
    ).sub(1.0)
    curves["optimized_drawdown"] = curves["optimized_lgbm"].div(
        curves["optimized_lgbm"].cummax()
    ).sub(1.0)
    curves.index.name = "date"
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")

    comparison = benchmark_comparison(curves)
    checks = promotion_checks(comparison)
    maturity_rows = base_audit["maturity_audit"] + barra_audit["maturity_audit"]
    all_mature = all(
        pd.Timestamp(row["max_training_label_end_date"])
        < pd.Timestamp(row["prediction_start"])
        for row in maturity_rows
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_candidate_only" if all(checks.values()) else "rejected",
        "objective": "improve 2020-2023 excess without sacrificing 2024-2026",
        "selection_contract": {
            "freeze_date": SELECTION_FREEZE.isoformat(),
            "window": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "score_grid": {
                "model_weights": MODEL_WEIGHTS,
                "rule_weights": RULE_WEIGHTS,
            },
            "portfolio_grid": {
                "top_k": TOP_K_VALUES,
                "rebalance_days": REBALANCE_VALUES,
                "n_drop": N_DROP_VALUES,
            },
            "objective_formula": "cagr + 0.25*worst_year_excess + 0.05*sharpe + 0.05*max_drawdown - 0.002*annual_turnover",
        },
        "current_parameters": CURRENT_PARAMS,
        "selected_parameters": selected,
        "selected_selection_summary": winner["selection"],
        "top_selection_candidates": sorted(
            portfolio_rows,
            key=lambda row: row["selection"]["objective"],
            reverse=True,
        )[:20],
        "diagnostic_periods": {
            "current_lgbm": strategy_periods(current_result),
            "optimized_lgbm": strategy_periods(optimized_result),
        },
        "benchmark_comparison": comparison,
        "promotion_gate": {"passed": all(checks.values()), "checks": checks},
        "training_audit": {
            "frozen_base_strictly_mature": frozen_base_audit["strictly_mature"],
            "frozen_barra_strictly_mature": frozen_barra_audit["strictly_mature"],
            "base_refits": base_audit["refit_months"],
            "barra_refits": barra_audit["refit_months"],
            "all_strictly_mature": all_mature,
        },
        "universe_snapshot_run_id": snapshot_id,
        "production_change": False,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(BASE_DIR)),
            "report": str(OUTPUT_REPORT.relative_to(BASE_DIR)),
            "curves": str(OUTPUT_CURVES.relative_to(BASE_DIR)),
        },
        "warnings": warnings
        + [
            "2020-2026 has been repeatedly revealed and is diagnostic-only.",
            "Current Top50 membership is backfilled and has survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time data.",
            "No production or paper-trading configuration was changed.",
        ],
    }
    payload = json_ready(payload)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
