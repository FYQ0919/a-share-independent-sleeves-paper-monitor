from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.backtest_data import BaoStockHistoryProvider
from app.config import BASE_DIR, settings
from app.lgbm_strategy import WINNER_CONFIG
from app.models import Sector
from app.selection import SelectionEngine
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


DATA_START = date(2016, 8, 25)
SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_START = date(2020, 1, 1)
DIAGNOSTIC_END = date(2026, 8, 25)
COST_BPS = 12.0

SNAPSHOT_PATH = BASE_DIR / "data" / "cache" / "akshare_snapshot.json"
OUTPUT_PATH = BASE_DIR / "data" / "lgbm_topk_universe_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_topk_universe_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_topk_universe_curves.csv"

CANDIDATES = {
    "pool50_top5": {"pool_size": 50, "top_k": 5},
    "pool50_top10": {"pool_size": 50, "top_k": 10},
    "pool100_top5": {"pool_size": 100, "top_k": 5},
    "pool100_top10": {"pool_size": 100, "top_k": 10},
}


def validate_candidates() -> str:
    if list(CANDIDATES) != [
        "pool50_top5", "pool50_top10", "pool100_top5", "pool100_top10"
    ]:
        raise RuntimeError("TopK/股票池候选顺序已改变")
    expected = [(50, 5), (50, 10), (100, 5), (100, 10)]
    actual = [(row["pool_size"], row["top_k"]) for row in CANDIDATES.values()]
    if actual != expected:
        raise RuntimeError("TopK/股票池候选网格已改变")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def snapshot_universes() -> tuple[dict[int, list[dict]], dict]:
    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    stocks = pd.DataFrame(payload["stocks"])
    sectors = [Sector(**item) for item in payload.get("sectors", [])]
    selector = SelectionEngine(
        settings.min_amount,
        settings.min_market_cap,
        settings.factor_weights,
        settings.technology_reserve,
    )
    selected = selector.select(stocks, sectors, 100)
    if len(selected) < 100:
        raise RuntimeError(f"全市场快照只能生成{len(selected)}只候选，无法构造Top100")
    rows = [asdict(item) for item in selected]
    codes = [item["code"] for item in rows]
    universe_hash = hashlib.sha256(
        json.dumps(codes, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    audit = {
        "snapshot_created_at": payload["created_at"],
        "snapshot_stock_count": len(stocks),
        "selection_method": "SelectionEngine with unchanged filters, weights, and technology reserve",
        "technology_reserve": settings.technology_reserve,
        "top100_code_sha256": universe_hash,
    }
    return {50: rows[:50], 100: rows[:100]}, audit


def load_history(candidates: list[dict]) -> tuple[pd.DataFrame, dict]:
    codes = [item["code"] for item in candidates]
    names = {item["code"]: item["name"] for item in candidates}
    provider = BaoStockHistoryProvider(settings.cache_dir)
    history, warnings = provider.load(codes, names, DATA_START, DIAGNOSTIC_END)
    history["date"] = pd.to_datetime(history["date"])
    loaded_codes = sorted(history["code"].astype(str).unique().tolist())
    first_dates = history.groupby("code")["date"].min()
    return history, {
        "requested": len(codes),
        "loaded": len(loaded_codes),
        "missing_or_short": [code for code in codes if code not in loaded_codes],
        "available_by_selection_start": int(first_dates.le(pd.Timestamp(SELECTION_START)).sum()),
        "available_by_diagnostic_start": int(first_dates.le(pd.Timestamp(DIAGNOSTIC_START)).sum()),
        "warnings": warnings,
    }


def run_scores(history: pd.DataFrame, start: date, end: date) -> tuple[pd.DataFrame, dict]:
    frame, feature_columns = build_alpha158_lite(history)
    baseline = baseline_scores(frame, start, end)
    prediction, audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, start, end
    )
    scores = blend_scores(frame, prediction, baseline, WINNER_CONFIG["baseline_weight"])
    return scores, audit


def candidate_passes(row: dict, baseline: dict) -> bool:
    performance = row["selection"]["performance"]
    base_performance = baseline["selection"]["performance"]
    activity = row["selection"]["activity"]
    base_activity = baseline["selection"]["activity"]
    return bool(
        row["name"] != "pool50_top5"
        and performance["annual_return"] >= base_performance["annual_return"] * 0.95
        and abs(performance["max_drawdown"]) <= abs(base_performance["max_drawdown"]) * 0.90
        and performance["sharpe"] >= base_performance["sharpe"]
        and activity["annual_turnover"] <= base_activity["annual_turnover"] * 1.50
    )


def candidate_rank(row: dict) -> tuple:
    performance = row["selection"]["performance"]
    calmar = performance["annual_return"] / abs(performance["max_drawdown"])
    return (
        calmar,
        performance["sharpe"],
        performance["annual_return"],
        -row["selection"]["activity"]["annual_turnover"],
    )


def export_curves(winner_result: dict, baseline_result: dict) -> int:
    curves = pd.concat(
        {
            "winner": normalized_curve(winner_result, "strategy_return"),
            "pool50_top5": normalized_curve(baseline_result, "strategy_return"),
            "pool50_benchmark": normalized_curve(baseline_result, "benchmark_return"),
        },
        axis=1,
        join="inner",
    )
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")
    return len(curves)


def main() -> None:
    manifest = validate_candidates()
    universes, snapshot_audit = snapshot_universes()
    histories: dict[int, pd.DataFrame] = {}
    coverage = {}
    for pool_size in (50, 100):
        histories[pool_size], coverage[pool_size] = load_history(universes[pool_size])

    selection_scores = {}
    selection_audits = {}
    for pool_size in (50, 100):
        selection_history = histories[pool_size][
            histories[pool_size]["date"].le(pd.Timestamp(SELECTION_END))
        ].copy()
        selection_scores[pool_size], selection_audits[pool_size] = run_scores(
            selection_history, SELECTION_START, SELECTION_END
        )

    candidates = []
    selection_results = {}
    for name, config in CANDIDATES.items():
        pool_size = config["pool_size"]
        selection_history = histories[pool_size][
            histories[pool_size]["date"].le(pd.Timestamp(SELECTION_END))
        ].copy()
        result = topk_dropout_backtest(
            selection_history,
            selection_scores[pool_size],
            SELECTION_START,
            SELECTION_END,
            top_k=config["top_k"],
            n_drop=1,
            rebalance_days=10,
            cost_bps=COST_BPS,
        )
        selection_results[name] = result
        candidates.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
        })

    baseline = next(row for row in candidates if row["name"] == "pool50_top5")
    eligible = [row for row in candidates if candidate_passes(row, baseline)]
    winner = max(eligible, key=candidate_rank) if eligible else baseline
    winner["selection_gate_passed"] = bool(eligible)

    diagnostic = None
    curve_rows = 0
    checks = {"selection_candidate_exists": bool(eligible)}
    if eligible:
        needed_pools = {50, int(winner["config"]["pool_size"])}
        diagnostic_scores = {}
        diagnostic_audits = {}
        for pool_size in needed_pools:
            diagnostic_scores[pool_size], diagnostic_audits[pool_size] = run_scores(
                histories[pool_size], DIAGNOSTIC_START, DIAGNOSTIC_END
            )
        baseline_result = topk_dropout_backtest(
            histories[50],
            diagnostic_scores[50],
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
            top_k=5,
            n_drop=1,
            rebalance_days=10,
            cost_bps=COST_BPS,
        )
        winner_pool = int(winner["config"]["pool_size"])
        winner_result = topk_dropout_backtest(
            histories[winner_pool],
            diagnostic_scores[winner_pool],
            DIAGNOSTIC_START,
            DIAGNOSTIC_END,
            top_k=int(winner["config"]["top_k"]),
            n_drop=1,
            rebalance_days=10,
            cost_bps=COST_BPS,
        )
        baseline_summary = research_summary(
            baseline_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        winner_summary = research_summary(
            winner_result, DIAGNOSTIC_START, DIAGNOSTIC_END
        )
        base_performance = baseline_summary["performance"]
        winner_performance = winner_summary["performance"]
        checks.update({
            "annual_return_retained_95pct": (
                winner_performance["annual_return"] >= base_performance["annual_return"] * 0.95
            ),
            "drawdown_improved_10pct": (
                abs(winner_performance["max_drawdown"])
                <= abs(base_performance["max_drawdown"]) * 0.90
            ),
            "sharpe_not_lower": winner_performance["sharpe"] >= base_performance["sharpe"],
            "turnover_not_higher_50pct": (
                winner_summary["activity"]["annual_turnover"]
                <= baseline_summary["activity"]["annual_turnover"] * 1.50
            ),
        })
        curve_rows = export_curves(winner_result, baseline_result)
        diagnostic = {
            "pool50_top5": baseline_summary,
            "winner": winner_summary,
            "winner_name": winner["name"],
            "model_audit": {
                str(pool_size): {
                    "refit_months": audit["refit_months"],
                    "first": audit["maturity_audit"][0],
                    "last": audit["maturity_audit"][-1],
                }
                for pool_size, audit in diagnostic_audits.items()
            },
        }

    passed = bool(eligible) and all(checks.values())
    candidates.sort(key=candidate_rank, reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "historical_gate_passed_not_deployed" if passed else "research_only_not_promoted",
        "strategy": "expanding LGBM factorial comparison of Top5/Top10 and pool50/pool100",
        "windows": {
            "candidate_selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_deployment_blocker": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        },
        "execution_contract": (
            "signal at close; execute next open; rebalance every 10 sessions; "
            "drop at most one holding; 12bp one-way cost"
        ),
        "candidate_manifest_sha256": manifest,
        "snapshot": snapshot_audit,
        "coverage": {str(key): value for key, value in coverage.items()},
        "winner": winner,
        "selection_model_audit": {
            str(pool_size): {
                "refit_months": audit["refit_months"],
                "first": audit["maturity_audit"][0],
                "last": audit["maturity_audit"][-1],
            }
            for pool_size, audit in selection_audits.items()
        },
        "diagnostic": diagnostic,
        "promotion_gate": {"passed": passed, "checks": checks},
        "candidates": candidates,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)) if curve_rows else None,
        "production_change": False,
        "forward_paper_required_sessions": 126,
        "warnings": [
            "This factorial experiment uses the 2026-08-27 full-market snapshot, not the older production Top50 snapshot.",
            "Both universes are current-snapshot constituents backfilled through history and have constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "The 2020-2026 window has been repeatedly revealed and is only a deployment blocker.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# LGBM Top5/Top10 与 50/100 股票池研究 v1",
        "",
        f"- 状态：{'历史门禁通过，未部署' if passed else '未通过晋级门禁'}",
        f"- 快照：{snapshot_audit['snapshot_created_at']}",
        f"- 选择冠军：`{winner['name']}`",
        f"- Top50实际加载：{coverage[50]['loaded']}/50",
        f"- Top100实际加载：{coverage[100]['loaded']}/100",
        "- 生产修改：否",
        "",
        "## 2018-2019冻结候选",
        "",
        "| 候选 | 年化 | 最大回撤 | 夏普 | 年化换手 |",
        "|---|---:|---:|---:|---:|",
        *[
            "| {name} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} | {turnover:.2f}x |".format(
                name=row["name"],
                annual=row["selection"]["performance"]["annual_return"],
                drawdown=row["selection"]["performance"]["max_drawdown"],
                sharpe=row["selection"]["performance"]["sharpe"],
                turnover=row["selection"]["activity"]["annual_turnover"],
            )
            for row in candidates
        ],
    ]
    if diagnostic:
        base = diagnostic["pool50_top5"]["performance"]
        selected = diagnostic["winner"]["performance"]
        report.extend([
            "",
            "## 2020-2026已揭示区间阻断验证",
            "",
            f"- 年化：{base['annual_return']:.2%} -> {selected['annual_return']:.2%}",
            f"- 最大回撤：{base['max_drawdown']:.2%} -> {selected['max_drawdown']:.2%}",
            f"- 夏普：{base['sharpe']:.3f} -> {selected['sharpe']:.3f}",
        ])
    report.extend([
        "",
        "该实验只改变持股数和候选池规模，模型参数、调仓周期、替换数、成交时点与成本保持不变。",
    ])
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "coverage": payload["coverage"],
        "winner": winner,
        "diagnostic": diagnostic,
        "promotion_gate": payload["promotion_gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
