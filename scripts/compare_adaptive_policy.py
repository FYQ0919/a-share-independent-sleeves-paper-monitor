from __future__ import annotations

from dataclasses import replace
from datetime import date
import json

from app.backtest_engine import BacktestConfig, BacktestEngine
from analyze_strategy import BASE_DIR, load_history, period_summary, summarize_result


OUTPUT_PATH = BASE_DIR / "data" / "adaptive_policy_comparison.json"


def main():
    history = load_history()
    codes = sorted(history["code"].unique().tolist())
    engine = BacktestEngine()
    base = BacktestConfig(
        mode="live",
        start_date=date(2024, 1, 1),
        end_date=date(2026, 8, 25),
        codes=codes,
        top_n=5,
        rebalance_days=10,
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
    )
    cases = [("固定10日", base)]
    for min_hold_days in (3, 5, 7):
        for rank_buffer in (7, 8, 10):
            for score_gap in (10.0, 15.0, 20.0):
                label = f"自适应 H{min_hold_days} B{rank_buffer} G{score_gap:.0f}"
                cases.append((label, replace(
                    base,
                    rebalance_policy="adaptive",
                    min_hold_days=min_hold_days,
                    rank_buffer=rank_buffer,
                    score_gap=score_gap,
                    max_replacements=1,
                    entry_rank=3,
                    max_adaptive_per_cycle=1,
                )))

    rows = []
    for label, config in cases:
        result = engine.run(history, config, [])
        rows.append({
            "label": label,
            "config": {
                "rebalance_policy": config.rebalance_policy,
                "min_hold_days": config.min_hold_days,
                "rank_buffer": config.rank_buffer,
                "score_gap": config.score_gap,
                "max_replacements": config.max_replacements,
                "entry_rank": config.entry_rank,
                "max_adaptive_per_cycle": config.max_adaptive_per_cycle,
            },
            "full": summarize_result(result, label),
            "training": period_summary(result, date(2024, 1, 1), date(2025, 6, 30)),
            "validation": period_summary(result, date(2025, 7, 1), date(2026, 8, 25)),
            "scheduled_rebalances": result["data"]["scheduled_rebalances"],
            "adaptive_rebalances": result["data"]["adaptive_rebalances"],
        })
    baseline = rows[0]
    adaptive = rows[1:]
    adaptive.sort(
        key=lambda row: (
            min(row["training"]["excess_return"], row["validation"]["excess_return"]),
            row["validation"]["information_ratio"],
            row["full"]["total_return"],
        ),
        reverse=True,
    )
    payload = {
        "contract": "每日收盘评估；最少持有期、排名缓冲和分差满足时最多提前替换1只；第10日仍执行主调仓",
        "baseline": baseline,
        "adaptive_cases": adaptive,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for row in [baseline] + adaptive[:15]:
        print(
            row["label"],
            f"train_excess={row['training']['excess_return']:.2%}",
            f"valid_excess={row['validation']['excess_return']:.2%}",
            f"full={row['full']['total_return']:.2%}",
            f"drawdown={row['full']['max_drawdown']:.2%}",
            f"IR={row['full']['information_ratio']:.2f}",
            f"adaptive={row['adaptive_rebalances']}",
        )


if __name__ == "__main__":
    main()
