from __future__ import annotations

import argparse
import json

import pandas as pd

from evaluate_factor_batch import BATCH_BUILDERS, OUTPUT_DIR, evaluate_batch
from app.config import settings
from app.factor_mining import daily_rank_ic, expanding_factor_scores
from app.lgbm_features import build_alpha158_lite
from app.storage import Storage
from optimize_factor_strategy import load_current_history
from research_qlib_ridge_topk_v1 import baseline_scores
from run_factor_mining import _performance_pair, _score_frame
from update_factor_registry import _latest_candidates, rolling_factor_config


def backtest_batch(batch: str) -> dict:
    selection = evaluate_batch(batch)
    storage = Storage(settings.database_path)
    candidates, _ = _latest_candidates(storage)
    history, _, warnings = load_current_history(candidates)
    history["date"] = pd.to_datetime(history["date"])
    config = rolling_factor_config(history)
    research_history = history[
        history["date"].le(pd.Timestamp(config.selection.end))
    ].copy()
    frame, alpha_features = build_alpha158_lite(research_history)
    frame, batch_features = BATCH_BUILDERS[batch](frame)
    features = alpha_features + batch_features
    ic_rows = daily_rank_ic(frame, features, config.min_cross_section)
    scores, maturity_audit = expanding_factor_scores(
        frame,
        ic_rows,
        selection["selected_full_pool"],
        config.selection.start,
        config.selection.end,
        config,
    )
    factor_score_frame = _score_frame(frame, scores)
    baseline_score_frame = _score_frame(
        frame, baseline_scores(frame, config.selection.start, config.selection.end)
    )
    evaluation = _performance_pair(
        research_history,
        factor_score_frame,
        baseline_score_frame,
        config.selection.start,
        config.selection.end,
    )
    payload = {
        "batch": batch,
        "selection": config.to_dict()["selection"],
        "quarantine_prices_loaded": False,
        "selected_full_pool": selection["selected_full_pool"],
        "selected_batch_features": selection["selected_in_combined_pool"],
        "all_monthly_labels_strictly_mature": bool(maturity_audit)
        and all(row["strictly_mature"] for row in maturity_audit),
        "evaluation": evaluation,
        "warnings": warnings,
        "production_change": False,
    }
    path = OUTPUT_DIR / f"{batch}_selection_backtest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", choices=sorted(BATCH_BUILDERS))
    args = parser.parse_args()
    payload = backtest_batch(args.batch)
    candidate = payload["evaluation"]["factor_ensemble"]["performance"]
    baseline = payload["evaluation"]["current_rule_baseline"]["performance"]
    print(json.dumps({
        "batch": payload["batch"],
        "selected_batch_features": payload["selected_batch_features"],
        "annual_return": candidate["annual_return"],
        "max_drawdown": candidate["max_drawdown"],
        "sharpe": candidate["sharpe"],
        "baseline_annual_return": baseline["annual_return"],
        "baseline_max_drawdown": baseline["max_drawdown"],
        "baseline_sharpe": baseline["sharpe"],
        "quarantine_prices_loaded": False,
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
