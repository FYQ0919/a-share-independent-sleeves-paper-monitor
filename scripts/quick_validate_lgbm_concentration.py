from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from quick_validate_lgbm_topk_universe import fit_frozen_predictions
from research_lgbm_topk_universe_v1 import (
    COST_BPS,
    SELECTION_END,
    SELECTION_START,
    load_history,
    snapshot_universes,
)
from research_qlib_ridge_topk_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    research_summary,
    topk_dropout_backtest,
)


OUTPUT_PATH = BASE_DIR / "data" / "lgbm_concentration_quick_validation.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_concentration_quick_validation.md"
TOP_K_GRID = (3, 4, 5, 7, 10)


def main() -> None:
    universes, snapshot_audit = snapshot_universes()
    history, coverage = load_history(universes[50])
    history = history[history["date"].le(pd.Timestamp(SELECTION_END))].copy()
    frame, feature_columns = build_alpha158_lite(history)
    model_prediction, model_audit = fit_frozen_predictions(frame, feature_columns)
    baseline_prediction = baseline_scores(frame, SELECTION_START, SELECTION_END)
    scores = blend_scores(
        frame,
        model_prediction,
        baseline_prediction,
        WINNER_CONFIG["baseline_weight"],
    )
    rows = []
    for top_k in TOP_K_GRID:
        result = topk_dropout_backtest(
            history,
            scores,
            SELECTION_START,
            SELECTION_END,
            top_k=top_k,
            n_drop=1,
            rebalance_days=10,
            cost_bps=COST_BPS,
        )
        rows.append({
            "name": f"top{top_k}",
            "top_k": top_k,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
        })
    baseline = next(row for row in rows if row["top_k"] == 5)
    base_performance = baseline["selection"]["performance"]
    base_activity = baseline["selection"]["activity"]
    eligible = [
        row for row in rows
        if row["top_k"] != 5
        and row["selection"]["performance"]["annual_return"]
        > base_performance["annual_return"]
        and row["selection"]["performance"]["sharpe"] >= base_performance["sharpe"]
        and row["selection"]["performance"]["max_drawdown"]
        >= base_performance["max_drawdown"] - 0.02
        and row["selection"]["activity"]["annual_turnover"]
        <= base_activity["annual_turnover"] * 1.50
    ]
    winner = max(
        eligible,
        key=lambda row: row["selection"]["performance"]["annual_return"],
    ) if eligible else baseline
    rows.sort(
        key=lambda row: row["selection"]["performance"]["annual_return"],
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "quick_directional_candidate" if eligible else "quick_directional_rejected",
        "objective": "increase annual return through concentration with Sharpe floor and 2pp drawdown tolerance",
        "window": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
        "top_k_grid": list(TOP_K_GRID),
        "snapshot": snapshot_audit,
        "coverage": coverage,
        "model_audit": model_audit,
        "winner": winner,
        "gate_passed": bool(eligible),
        "candidates": rows,
        "production_change": False,
        "warnings": [
            "Fast directional validation with one pre-2018 frozen model, not monthly expanding research.",
            "The current snapshot is backfilled through history and has survivorship bias.",
            "No 2020-2026 data was opened for this concentration search.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# LGBM 集中持仓年化优先快速验证",
        "",
        f"- 状态：{'发现方向性候选，未部署' if eligible else '没有候选通过'}",
        f"- 冠军：`{winner['name']}`",
        "- 股票池：同快照Top50",
        "- 评测：2018-2019",
        "- 生产修改：否",
        "",
        "| 持股数 | 年化 | 最大回撤 | 夏普 | 年化换手 |",
        "|---:|---:|---:|---:|---:|",
        *[
            "| {top_k} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} | {turnover:.2f}x |".format(
                top_k=row["top_k"],
                annual=row["selection"]["performance"]["annual_return"],
                drawdown=row["selection"]["performance"]["max_drawdown"],
                sharpe=row["selection"]["performance"]["sharpe"],
                turnover=row["selection"]["activity"]["annual_turnover"],
            )
            for row in rows
        ],
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "winner": winner,
        "gate_passed": bool(eligible),
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
