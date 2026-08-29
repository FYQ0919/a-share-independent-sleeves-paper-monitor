from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_equal_weight_strategy_blend_2020_2026 import html_fragment


SOURCE_PATH = ROOT / "reports" / "semiconductor_trend_csi300_hedge_v1_curves.csv"
PREVIOUS_PATH = ROOT / "reports" / "equal_weight_strategy_blend_2020_2026.csv"
OUTPUT_CSV = ROOT / "reports" / "high_trend_75_hedge_blend_2020_2026.csv"
OUTPUT_JSON = ROOT / "data" / "high_trend_75_hedge_blend_2020_2026.json"
OUTPUT_PNG = ROOT / "reports" / "high_trend_75_hedge_blend_2020_2026.png"
OUTPUT_HTML = (
    Path(tempfile.gettempdir())
    / "codex-visualizations"
    / "high-trend-75-hedge-blend-2020-2026.html"
)

TREND_WEIGHT = 0.75
LGBM_WEIGHT = 0.25
RISK_FOCUSED_WEIGHT = 0.90
WEIGHT_GRID = tuple(round(value, 2) for value in np.arange(0.55, 0.95, 0.05))


def performance(curve: pd.Series) -> dict:
    returns = curve.pct_change().fillna(0.0)
    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / len(curve)) - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    return {
        "terminal_value": float(curve.iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(curve.div(curve.cummax()).sub(1.0).min()),
        "observations": int(len(curve)),
    }


def blended_curve(trend: pd.Series, lgbm: pd.Series, trend_weight: float) -> pd.Series:
    return trend.mul(trend_weight).add(lgbm.mul(1.0 - trend_weight))


def mapped_html(frame: pd.DataFrame, summaries: dict[str, dict]) -> str:
    sampled = frame.iloc[::5].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]])
    mapped = pd.DataFrame(index=sampled.index)
    mapped["equal_weight_blend"] = sampled["high_trend_blend"]
    mapped["strongest_candidate"] = sampled["previous_50_50_blend"]
    mapped["formal_lgbm"] = sampled["pure_trend_hedge75"]
    mapped["equal_weight_blend_drawdown"] = sampled["high_trend_blend_drawdown"]
    mapped["strongest_candidate_drawdown"] = sampled["previous_50_50_blend_drawdown"]
    mapped["formal_lgbm_drawdown"] = sampled["pure_trend_hedge75_drawdown"]
    records = []
    for timestamp, row in mapped.iterrows():
        record = {"date": timestamp.date().isoformat()}
        for name in ("equal_weight_blend", "strongest_candidate", "formal_lgbm"):
            record[name] = round(float(row[name]), 6)
            record[f"{name}_drawdown"] = round(float(row[f"{name}_drawdown"]), 6)
        records.append(record)
    html = html_fragment(
        records,
        {
            "equal_weight_blend": summaries["high_trend_blend"],
            "strongest_candidate": summaries["previous_50_50_blend"],
            "formal_lgbm": summaries["pure_trend_hedge75"],
        },
    )
    replacements = {
        "equal-weight-strategy-blend": "high-trend-75-hedge-blend",
        "2020-2026 两策略 50/50 等资金组合": "2020-2026 趋势专家75% / LGBM25% 组合",
        "初始资金各 50%，子策略独立复利，不额外强制日间再平衡": "趋势专家资金75%，正式 LGBM 资金25%；趋势侧 CSI300 最大对冲75%",
        "50/50 等资金组合": "趋势75% / LGBM25% 新组合",
        "趋势专家 + CSI300 50% 对冲": "上一版 50/50 组合",
        "之前正式 LGBM": "纯趋势专家（CSI300 75% 对冲）",
        "初始资金各 50%": "初始资金按 75% / 25% 分配",
    }
    for source, target in replacements.items():
        html = html.replace(source, target)
    return html


def main() -> None:
    source = (
        pd.read_csv(SOURCE_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc["2020-01-02":"2026-08-25"]
    )
    trend = source["trend_hedge_75"].div(float(source["trend_hedge_75"].iloc[0]))
    lgbm = source["current_lgbm"].div(float(source["current_lgbm"].iloc[0]))
    previous = (
        pd.read_csv(PREVIOUS_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()["equal_weight_blend"]
        .reindex(source.index)
    )

    frame = pd.DataFrame(
        {
            "high_trend_blend": blended_curve(trend, lgbm, TREND_WEIGHT),
            "previous_50_50_blend": previous,
            "risk_focused_90_10": blended_curve(trend, lgbm, RISK_FOCUSED_WEIGHT),
            "pure_trend_hedge75": trend,
            "formal_lgbm": lgbm,
        }
    ).dropna()
    for name in list(frame.columns):
        frame[f"{name}_drawdown"] = frame[name].div(frame[name].cummax()).sub(1.0)
    frame.index.name = "date"
    frame.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")

    summaries = {
        name: performance(frame[name])
        for name in (
            "high_trend_blend",
            "previous_50_50_blend",
            "risk_focused_90_10",
            "pure_trend_hedge75",
            "formal_lgbm",
        )
    }
    scan = []
    for weight in WEIGHT_GRID:
        curve = blended_curve(trend, lgbm, weight)
        scan.append({"trend_weight": weight, "lgbm_weight": 1.0 - weight, **performance(curve)})
    best_sharpe = max(scan, key=lambda row: row["sharpe"])
    best_drawdown = max(scan, key=lambda row: row["max_drawdown"])
    if abs(best_sharpe["trend_weight"] - TREND_WEIGHT) > 1e-12:
        raise RuntimeError("The fixed 75/25 allocation no longer has the best scanned Sharpe")

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(mapped_html(frame, summaries), encoding="utf-8")
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_diagnostic_only_not_deployed",
        "window": ["2020-01-02", "2026-08-25"],
        "selected": {
            "trend_expert_weight": TREND_WEIGHT,
            "formal_lgbm_weight": LGBM_WEIGHT,
            "trend_expert_csi300_maximum_hedge": 0.75,
            "sleeve_rebalance": "none; each sleeve compounds independently",
            "selection_reason": "highest Sharpe in the fixed 55%-90% trend-weight scan",
        },
        "metrics": summaries,
        "weight_scan": scan,
        "best_scanned_sharpe": best_sharpe,
        "best_scanned_drawdown": best_drawdown,
        "selected_minus_previous": {
            metric: summaries["high_trend_blend"][metric] - summaries["previous_50_50_blend"][metric]
            for metric in ("annual_return", "sharpe", "max_drawdown")
        },
        "production_change": False,
        "artifacts": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "png": str(OUTPUT_PNG.relative_to(ROOT)),
            "html": str(OUTPUT_HTML),
        },
        "limitations": [
            "All curves are repeatedly revealed historical diagnostics, not fresh out-of-sample evidence.",
            "The underlying strategy curves already include their modeled stock and hedge costs.",
            "No extra cross-sleeve transfer cost is modeled because sleeve weights are not rebalanced after inception.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "csv": str(OUTPUT_CSV),
        "json": str(OUTPUT_JSON),
        "html": str(OUTPUT_HTML),
        "metrics": summaries,
        "best_scanned_sharpe": best_sharpe,
        "best_scanned_drawdown": best_drawdown,
        "selected_minus_previous": payload["selected_minus_previous"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
