from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import CANDIDATES, fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


START = date(2020, 1, 1)
END = date(2026, 8, 25)
CANDIDATE_NAME = "lgbm_l15_blend40"
OUTPUT_PATH = BASE_DIR / "data" / "lgbm_high_annual_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_high_annual_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_high_annual_v1_curves.csv"
CURRENT_RESULT_PATH = BASE_DIR / "reports" / "lgbm_vs_nasdaq_2020_2026.json"
CURRENT_CURVE_PATH = BASE_DIR / "reports" / "lgbm_vs_nasdaq_2020_2026.csv"


def historical_gate(candidate: dict, current: dict, candidate_activity: dict, current_activity: dict) -> dict:
    checks = {
        "annual_return_higher": candidate["annual_return"] > current["annual_return"],
        "sharpe_not_lower": candidate["sharpe"] >= current["sharpe"],
        "drawdown_not_worse_by_5pp": candidate["max_drawdown"] >= current["max_drawdown"] - 0.05,
        "turnover_not_higher_10pct": candidate_activity["annual_turnover"]
        <= current_activity["annual_turnover"] * 1.10,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    config = CANDIDATES[CANDIDATE_NAME]
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history = history[history["date"].le(pd.Timestamp(END))].copy()
    frame, features = build_alpha158_lite(history)
    baseline = baseline_scores(frame, START, END)
    prediction, audit = fit_expanding_lgbm_predictions(
        frame, features, config, START, END
    )
    maturity_violations = [
        row for row in audit["maturity_audit"]
        if row["max_training_label_end_date"] >= row["prediction_start"]
    ]
    if maturity_violations:
        raise RuntimeError(f"高年化候选使用了未成熟标签: {maturity_violations[:3]}")
    scores = blend_scores(frame, prediction, baseline, config["baseline_weight"])
    result = topk_dropout_backtest(history, scores, START, END)
    candidate_summary = research_summary(result, START, END)

    current_payload = json.loads(CURRENT_RESULT_PATH.read_text(encoding="utf-8"))
    current_summary = current_payload["strategy"]["backtest_summary"]
    gate = historical_gate(
        candidate_summary["performance"],
        current_summary["performance"],
        candidate_summary["activity"],
        current_summary["activity"],
    )
    candidate_curve = normalized_curve(result, "strategy_return").rename("high_annual_lgbm")
    current_curve = (
        pd.read_csv(CURRENT_CURVE_PATH, parse_dates=["date"])
        .set_index("date")["lgbm"]
        .rename("current_lgbm")
    )
    curves = pd.concat([candidate_curve, current_curve], axis=1, join="inner")
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "historical_gate_passed_not_deployed" if gate["passed"] else "research_only_rejected",
        "objective": "increase annual return without leverage",
        "candidate": CANDIDATE_NAME,
        "candidate_config": config,
        "selection_evidence_2018_2019": {
            "annual_return": 0.15519335447041382,
            "max_drawdown": -0.28310965741986416,
            "sharpe": 0.7580781369846901,
        },
        "window": [START.isoformat(), END.isoformat()],
        "execution_contract": "signal close; next-open execution; Top5; 10-session rebalance; 12bp one-way cost",
        "current_lgbm": current_summary,
        "high_annual_lgbm": candidate_summary,
        "historical_gate": gate,
        "causality_audit": {
            "passed": not maturity_violations,
            "monthly_refits": audit["refit_months"],
            "first": audit["maturity_audit"][0],
            "last": audit["maturity_audit"][-1],
        },
        "universe_snapshot_run_id": snapshot_id,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "production_change": False,
        "forward_paper_required": True,
        "warnings": warnings + [
            "The candidate existed before the 2020-2026 diagnostic, but this period is already revealed.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    current = current_summary["performance"]
    candidate = candidate_summary["performance"]
    REPORT_PATH.write_text(
        "\n".join([
            "# LGBM 高年化候选 v1",
            "",
            f"- 状态：{'历史门禁通过，未部署' if gate['passed'] else '历史门禁未通过'}",
            f"- 候选：`{CANDIDATE_NAME}`",
            f"- 当前LGBM：年化 {current['annual_return']:.2%}，回撤 {current['max_drawdown']:.2%}，夏普 {current['sharpe']:.3f}",
            f"- 高年化LGBM：年化 {candidate['annual_return']:.2%}，回撤 {candidate['max_drawdown']:.2%}，夏普 {candidate['sharpe']:.3f}",
            f"- 月度因果重训：{audit['refit_months']}次，成熟标签检查通过",
            "- 生产修改：否",
            "",
            "候选在2018-2019已经存在，本次仅做揭盲历史诊断。晋级仍需新的前向模拟盘。",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
