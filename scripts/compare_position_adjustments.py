from __future__ import annotations

from datetime import date
import json

from app.backtest_engine import BacktestConfig, BacktestEngine
from analyze_strategy import BASE_DIR, load_history, period_summary, summarize_result


OUTPUT_PATH = BASE_DIR / "data" / "position_adjustment_comparison.json"
CASES = [
    ("自然漂移", "none", 0.03),
    ("每日恢复等权", "daily_equal", 0.03),
    ("第5日恢复等权", "midcycle_equal", 0.03),
    ("偏离2%再平衡", "band_equal", 0.02),
    ("偏离3%再平衡", "band_equal", 0.03),
    ("偏离5%再平衡", "band_equal", 0.05),
]


def main():
    history = load_history()
    codes = sorted(history["code"].unique().tolist())
    engine = BacktestEngine()
    rows = []
    for label, mode, band in CASES:
        config = BacktestConfig(
            "live", date(2024, 1, 1), date(2026, 8, 25), codes,
            5, 10, 1_000_000, 12, "contrarian", mode, band,
        )
        result = engine.run(history, config, [])
        rows.append({
            "label": label,
            "mode": mode,
            "weight_band": band,
            "full": summarize_result(result, label),
            "training": period_summary(result, date(2024, 1, 1), date(2025, 6, 30)),
            "validation": period_summary(result, date(2025, 7, 1), date(2026, 8, 25)),
            "main_rebalances": len(result["rebalances"]),
            "interim_adjustments": result["data"]["interim_adjustment_count"],
        })
    payload = {
        "contract": "Top5名单每10交易日更新；持有期内只调整原有5只股票的权重",
        "cost_bps": 12,
        "cases": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for row in rows:
        print(
            row["label"],
            f"total={row['full']['total_return']:.2%}",
            f"excess={row['full']['excess_return']:.2%}",
            f"drawdown={row['full']['max_drawdown']:.2%}",
            f"cost={row['full']['total_cost']:.2%}",
            f"validation_excess={row['validation']['excess_return']:.2%}",
            f"interim={row['interim_adjustments']}",
        )


if __name__ == "__main__":
    main()
