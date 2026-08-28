from __future__ import annotations

import argparse
import json

import pandas as pd

from evaluate_factor_batch import BATCH_BUILDERS, OUTPUT_DIR
from app.config import settings
from app.factor_mining import daily_rank_ic, expanding_factor_scores
from app.lgbm_features import build_alpha158_lite
from app.storage import Storage
from optimize_factor_strategy import load_current_history
from research_qlib_ridge_topk_v1 import baseline_scores
from run_factor_mining import _performance_pair, _score_frame
from update_factor_registry import _latest_candidates, rolling_factor_config


def backtest_quarantine(batch: str) -> dict:
    frozen_path = OUTPUT_DIR / f"{batch}.json"
    if not frozen_path.exists():
        raise RuntimeError(f"缺少冻结的批次筛选结果: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("quarantine_prices_loaded") is not False:
        raise RuntimeError("批次候选不是在隔离期揭盲前冻结")

    storage = Storage(settings.database_path)
    candidates, _ = _latest_candidates(storage)
    history, _, warnings = load_current_history(candidates)
    history["date"] = pd.to_datetime(history["date"])
    config = rolling_factor_config(history)
    frame, alpha_features = build_alpha158_lite(history)
    frame, batch_features = BATCH_BUILDERS[batch](frame)
    features = alpha_features + batch_features
    ic_rows = daily_rank_ic(frame, features, config.min_cross_section)
    scores, maturity_audit = expanding_factor_scores(
        frame,
        ic_rows,
        frozen["selected_full_pool"],
        config.diagnostic.start,
        config.diagnostic.end,
        config,
    )
    factor_score_frame = _score_frame(frame, scores)
    baseline_score_frame = _score_frame(
        frame, baseline_scores(frame, config.diagnostic.start, config.diagnostic.end)
    )
    evaluation = _performance_pair(
        history,
        factor_score_frame,
        baseline_score_frame,
        config.diagnostic.start,
        config.diagnostic.end,
    )
    candidate = evaluation["factor_ensemble"]["performance"]
    baseline = evaluation["current_rule_baseline"]["performance"]
    checks = {
        "annual_return_retains_90pct": candidate["annual_return"]
        >= baseline["annual_return"] * 0.90,
        "sharpe_not_lower": candidate["sharpe"] >= baseline["sharpe"],
        "drawdown_not_worse_by_3pp": candidate["max_drawdown"]
        >= baseline["max_drawdown"] - 0.03,
    }
    payload = {
        "batch": batch,
        "frozen_selection_generated_at": frozen["generated_at"],
        "diagnostic": config.to_dict()["diagnostic"],
        "selected_full_pool": frozen["selected_full_pool"],
        "selected_batch_features": frozen["selected_in_combined_pool"],
        "all_monthly_labels_strictly_mature": bool(maturity_audit)
        and all(row["strictly_mature"] for row in maturity_audit),
        "evaluation": evaluation,
        "deployment_gate": {"passed": all(checks.values()), "checks": checks},
        "selection_effect": "deployment_blocker_only; frozen factors are never reselected here",
        "warnings": warnings,
        "production_change": False,
    }
    path = OUTPUT_DIR / f"{batch}_quarantine_backtest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", choices=sorted(BATCH_BUILDERS))
    args = parser.parse_args()
    payload = backtest_quarantine(args.batch)
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
        "deployment_gate": payload["deployment_gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
