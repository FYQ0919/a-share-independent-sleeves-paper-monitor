from __future__ import annotations

"""Forward validation of the frozen LightGBM ranker on the 2024-2026 window.

This script intentionally does not change the production strategy. It uses the
same frozen Top50 snapshot and the same next-open/12bp execution assumptions as
the existing research code, then reports model and portfolio-mechanics effects
separately.
"""

from datetime import date, datetime
import json

import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR
from app.lgbm_strategy import BASELINE_WEIGHTS, WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    fit_expanding_lgbm_predictions,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


START = date(2024, 1, 1)
END = date(2026, 8, 25)
SOURCE_PATH = BASE_DIR / "data" / "factor_optimization.json"
OUTPUT_PATH = BASE_DIR / "data" / "lgbm_validation_2024_2026.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_validation_2024_2026.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_validation_2024_2026_curve.csv"


def run_engine_rule(history: pd.DataFrame, codes: list[str], risk_overlay: str) -> dict:
    engine = BacktestEngine()
    engine._factor_weights_override = BASELINE_WEIGHTS
    config = BacktestConfig(
        mode="live",
        start_date=START,
        end_date=END,
        codes=codes,
        top_n=5,
        rebalance_days=10,
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
        rebalance_policy="adaptive",
        min_hold_days=5,
        rank_buffer=20,
        score_gap=15,
        max_replacements=1,
        entry_rank=3,
        max_adaptive_per_cycle=1,
        risk_overlay=risk_overlay,
        drawdown_limit=0.25 if risk_overlay != "none" else 0.0,
        drawdown_cooldown_days=10,
    )
    return engine.run(history, config, [])


def main() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history = history[history["date"] <= pd.Timestamp(END)].copy()
    codes = sorted(history["code"].unique().tolist())

    frame, feature_columns = build_alpha158_lite(history)
    baseline_prediction = baseline_scores(frame, START, END)
    prediction, audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, START, END
    )
    lgbm_scores = blend_scores(
        frame, prediction, baseline_prediction, WINNER_CONFIG["baseline_weight"]
    )
    rule_scores = blend_scores(
        frame,
        baseline_prediction,
        baseline_prediction,
        1.0,
    )

    # These two runs share exactly the same Top5/drop-one/10-session/cost
    # portfolio simulator. Only the cross-sectional scores differ.
    lgbm_result = topk_dropout_backtest(history, lgbm_scores, START, END)
    rule_topk_result = topk_dropout_backtest(history, rule_scores, START, END)

    lgbm_curve = normalized_curve(lgbm_result, "strategy_return").rename("lgbm")
    lgbm_curve.index.name = "date"
    nasdaq_path = BASE_DIR / "reports" / "test_strategy_vs_nasdaq.csv"
    nasdaq = pd.read_csv(nasdaq_path, parse_dates=["date"]).set_index("date")["nasdaq"]
    nasdaq = nasdaq[nasdaq.index <= pd.Timestamp(END)]
    common_start = lgbm_curve.index.min()
    nasdaq_base = nasdaq[nasdaq.index <= common_start].iloc[-1]
    nasdaq = nasdaq.div(float(nasdaq_base)).rename("nasdaq")
    curve = pd.concat([lgbm_curve, nasdaq], axis=1).sort_index().ffill().dropna().reset_index()
    curve = curve[curve["date"] >= common_start].copy()
    curve["date"] = curve["date"].dt.strftime("%Y-%m-%d")
    curve.to_csv(CURVE_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")

    # These runs reproduce the website's operational adaptive strategy and
    # separate the effect of the risk overlay from the model comparison.
    rule_none_result = run_engine_rule(history, codes, "none")
    rule_default_result = run_engine_rule(history, codes, "trend_volatility")

    summaries = {
        "lgbm_topk_dropout": research_summary(lgbm_result, START, END),
        "rule_topk_dropout": research_summary(rule_topk_result, START, END),
        "rule_adaptive_no_risk": {
            "performance": rule_none_result["metrics"],
            "config": rule_none_result["config"],
        },
        "rule_adaptive_default_risk": {
            "performance": rule_default_result["metrics"],
            "config": rule_default_result["config"],
        },
    }

    lgbm_perf = summaries["lgbm_topk_dropout"]["performance"]
    rule_topk_perf = summaries["rule_topk_dropout"]["performance"]
    checks = {
        "lgbm_beats_same_mechanics_annual_return": lgbm_perf["annual_return"] > rule_topk_perf["annual_return"],
        "lgbm_beats_same_mechanics_sharpe": lgbm_perf["sharpe"] > rule_topk_perf["sharpe"],
        "lgbm_drawdown_not_worse_by_3pp": lgbm_perf["max_drawdown"] >= rule_topk_perf["max_drawdown"] - 0.03,
    }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "window": {"start": START.isoformat(), "end": END.isoformat()},
        "universe_snapshot_run_id": snapshot_id,
        "requested_codes": len(requested_codes),
        "loaded_codes": len(codes),
        "model": {
            "backend": "lightgbm_native_lambdarank",
            "config": WINNER_CONFIG,
            "feature_count": len(feature_columns),
            "refit_months": audit["refit_months"],
            "maturity_audit": audit["maturity_audit"],
        },
        "execution_contract": "收盘生成信号；下一交易日开盘执行；Top5；每10个交易日主调仓；单边成本12bp",
        "comparisons": summaries,
        "checks": checks,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "warnings": warnings + [
            "这是冻结Top50回填上的研究型前向验证，不是严格无存续偏差的可实现收益",
            "LightGBM与规则TopK对照使用同一topk_dropout模拟器；网站自适应结果另列",
            "未因本结果修改生产默认策略",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def pct(value: float) -> str:
        return f"{value:.2%}"

    lines = [
        "# LightGBM 2024-2026 前向验证",
        "",
        f"- 区间：{START.isoformat()} 至 {END.isoformat()}",
        f"- 冻结股票池：{snapshot_id}（请求50只，加载{len(codes)}只）",
        f"- LightGBM：按月扩展训练，实际重训 {audit['refit_months']} 个月",
        "- 执行：收盘信号、下一交易日开盘、Top5、10日主调仓、单边12bp",
        "",
        "| 对照 | 年化收益 | 夏普 | 最大回撤 | 换手指标 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in [
        ("lgbm_topk_dropout", "LightGBM（同机制）"),
        ("rule_topk_dropout", "规则评分（同机制）"),
        ("rule_adaptive_no_risk", "规则自适应（无风控）"),
        ("rule_adaptive_default_risk", "规则自适应（默认风控）"),
    ]:
        perf = summaries[name]["performance"]
        activity = summaries[name].get("activity", {})
        turnover = activity.get("annual_turnover", perf.get("average_turnover", 0.0))
        lines.append(
            f"| {label} | {pct(perf['annual_return'])} | {perf['sharpe']:.3f} | {pct(perf['max_drawdown'])} | {turnover:.2f}x |"
        )
    lines.extend([
        "",
        f"- LightGBM相对同机制规则：年化差 {pct(lgbm_perf['annual_return'] - rule_topk_perf['annual_return'])}，夏普差 {lgbm_perf['sharpe'] - rule_topk_perf['sharpe']:.3f}。",
        f"- 同机制门禁：{'通过' if all(checks.values()) else '未通过'}。",
        "- 结论：该结果只用于解释验证集差异，不自动替换生产策略。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "curve": str(CURVE_PATH),
        "comparisons": summaries,
        "checks": checks,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
