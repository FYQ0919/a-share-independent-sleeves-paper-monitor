from __future__ import annotations

from datetime import date
import json

from app.backtest_engine import BacktestConfig, BacktestEngine
from analyze_strategy import BASE_DIR, load_history, period_summary, summarize_result


OUTPUT_PATH = BASE_DIR / "data" / "strategy_comparison.json"
PERIODS = [
    ("2024上半年", date(2024, 1, 1), date(2024, 6, 30)),
    ("2024下半年", date(2024, 7, 1), date(2024, 12, 31)),
    ("2025上半年", date(2025, 1, 1), date(2025, 6, 30)),
    ("2025下半年", date(2025, 7, 1), date(2025, 12, 31)),
    ("2026年至今", date(2026, 1, 1), date(2026, 8, 25)),
]


def main():
    history = load_history()
    codes = sorted(history["code"].unique().tolist())
    engine = BacktestEngine()
    configs = {
        "baseline": BacktestConfig(
            "live", date(2024, 1, 1), date(2026, 8, 25), codes, 5, 20, 1_000_000, 12, "multi_factor"
        ),
        "optimized": BacktestConfig(
            "live", date(2024, 1, 1), date(2026, 8, 25), codes, 5, 10, 1_000_000, 12, "contrarian"
        ),
    }
    results = {name: engine.run(history, config, []) for name, config in configs.items()}
    payload = {
        "universe": codes,
        "full_period": {
            name: summarize_result(result, result["data"]["strategy"])
            for name, result in results.items()
        },
        "rolling_periods": [
            {
                "label": label,
                "start": start.isoformat(),
                "end": end.isoformat(),
                **{name: period_summary(result, start, end) for name, result in results.items()},
            }
            for label, start, end in PERIODS
        ],
        "optimized_factor_weights": results["optimized"]["data"]["factor_weights"],
        "limitations": [
            "股票池来自当前候选，存在成分、存续和退市偏差",
            "因子和参数经过本样本探索，半年分段不是全新未触碰测试集",
            "未模拟涨跌停无法成交、滑点、分红、容量和部分成交",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for row in payload["rolling_periods"]:
        print(
            row["label"],
            f"baseline_excess={row['baseline']['excess_return']:.2%}",
            f"optimized_excess={row['optimized']['excess_return']:.2%}",
            f"optimized_drawdown={row['optimized']['max_drawdown']:.2%}",
        )


if __name__ == "__main__":
    main()
