from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_current_best_lgbm_csi300_2020_2026 import load_strategy, metrics


OFFICIAL_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
BENCHMARK_CACHE = ROOT / "data" / "cache" / "index" / "csi2000_2020_2026.csv"
OUTPUT_JSON = ROOT / "data" / "current_best_lgbm_vs_csi2000_2020_2026.json"
OUTPUT_CSV = ROOT / "reports" / "current_best_lgbm_vs_csi2000_2020_2026.csv"
OUTPUT_REPORT = ROOT / "reports" / "current_best_lgbm_vs_csi2000_2020_2026.md"


def fetch_csi2000(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    params = {
        "indexCode": "932000",
        "startDate": start.date().isoformat(),
        "endDate": end.date().isoformat(),
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.csindex.com.cn/",
    }
    try:
        with httpx.Client(timeout=30, trust_env=False, headers=headers) as client:
            response = client.get(OFFICIAL_URL, params=params)
            response.raise_for_status()
        payload = response.json()
        if str(payload.get("code")) != "200":
            raise RuntimeError(f"CSI official API error: {payload.get('msg', 'unknown')}")
        frame = pd.DataFrame(payload.get("data") or [])
        if frame.empty or "tradeDate" not in frame or "close" not in frame:
            raise RuntimeError("CSI official API returned no CSI2000 history")
        compact_dates = frame["tradeDate"].astype(str).str.replace("-", "", regex=False)
        frame["date"] = pd.to_datetime(compact_dates, format="%Y%m%d", errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame[["date", "close"]].dropna().drop_duplicates("date").sort_values("date")
        BENCHMARK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(BENCHMARK_CACHE, index=False, encoding="utf-8-sig", float_format="%.6f")
        source = {
            "mode": "live_csindex_official_api",
            "source": f"{OFFICIAL_URL}?indexCode=932000",
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
    except Exception as exc:
        if not BENCHMARK_CACHE.exists():
            raise
        frame = pd.read_csv(BENCHMARK_CACHE, parse_dates=["date"])
        source = {
            "mode": "cached_fallback",
            "source": str(BENCHMARK_CACHE.relative_to(ROOT)),
            "fallback_reason": str(exc),
            "cache_modified_at": datetime.fromtimestamp(
                BENCHMARK_CACHE.stat().st_mtime
            ).astimezone().isoformat(),
        }
    values = frame.set_index("date")["close"].sort_index().astype(float)
    values = values.loc[(values.index >= start) & (values.index <= end)]
    if len(values) < 2:
        raise RuntimeError("CSI2000 has insufficient observations in the strategy window")
    return values, source


def main() -> None:
    strategy = load_strategy()
    benchmark, source = fetch_csi2000(strategy.index.min(), strategy.index.max())
    common_dates = strategy.index.intersection(benchmark.index).sort_values()
    strategy = strategy.reindex(common_dates).astype(float)
    benchmark = benchmark.reindex(common_dates).astype(float)
    strategy = strategy.div(float(strategy.iloc[0]))
    benchmark = benchmark.div(float(benchmark.iloc[0]))

    curve = pd.DataFrame(
        {"current_best_lgbm": strategy, "csi2000": benchmark}, index=common_dates
    )
    curve["relative_wealth"] = curve["current_best_lgbm"].div(curve["csi2000"])
    curve.index.name = "date"
    curve.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")

    strategy_metrics = metrics(strategy)
    benchmark_metrics = metrics(benchmark)
    daily = curve[["current_best_lgbm", "csi2000"]].pct_change().dropna()
    active = daily["current_best_lgbm"].sub(daily["csi2000"])
    tracking_error = float(active.std(ddof=0) * np.sqrt(252.0))
    benchmark_variance = float(daily["csi2000"].var(ddof=0))
    beta = (
        float(daily["current_best_lgbm"].cov(daily["csi2000"]) / benchmark_variance)
        if benchmark_variance > 0
        else 0.0
    )
    excess = {
        "annualized_return_gap": float(
            strategy_metrics["annual_return"] - benchmark_metrics["annual_return"]
        ),
        "cumulative_return_arithmetic_gap": float(
            strategy_metrics["total_return"] - benchmark_metrics["total_return"]
        ),
        "relative_wealth": float(strategy.iloc[-1] / benchmark.iloc[-1]),
        "relative_cumulative_return": float(strategy.iloc[-1] / benchmark.iloc[-1] - 1.0),
        "tracking_error": tracking_error,
        "information_ratio": (
            float(active.mean() * 252.0 / tracking_error) if tracking_error > 0 else 0.0
        ),
        "beta": beta,
        "annualized_jensen_alpha_zero_rate": float(
            (daily["current_best_lgbm"].mean() - beta * daily["csi2000"].mean()) * 252.0
        ),
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic_not_promoted",
        "window": {
            "requested_start": strategy.index[0].date().isoformat(),
            "requested_end": strategy.index[-1].date().isoformat(),
            "common_start": common_dates[0].date().isoformat(),
            "common_end": common_dates[-1].date().isoformat(),
            "common_trading_days": int(len(common_dates)),
        },
        "strategy": {
            "name": "Current best 75/25 LGBM",
            "execution": "T-close signal; T+1 open; Top5; 10-session rebalance; max one replacement; 12bp one-way cost",
            "metrics": strategy_metrics,
        },
        "benchmark": {
            "name": "中证2000",
            "symbol": "932000",
            "return_type": "price return; excludes dividends, fees and taxes",
            "source_status": source,
            "metrics": benchmark_metrics,
        },
        "excess": excess,
        "benchmark_role": (
            "Small/micro-cap style opportunity-cost benchmark; not the primary style-matched "
            "benchmark for the current large-cap-biased Top50 universe."
        ),
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
        "limitations": [
            "The official CSI endpoint returned no 2026 observations, so the comparison stops at 2025-12-31 instead of extending the index with an ETF proxy.",
            "The current Top50 universe is backfilled and has survivorship and constituent bias.",
            "Current-vintage adjusted A-share prices are not strict point-in-time prices.",
            "The 2020-2026 period has been repeatedly revealed and is not an untouched holdout.",
            "CSI2000 is a price index and is materially smaller-cap than the strategy universe.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Current best LGBM vs CSI2000, 2020-2025 common window",
        "",
        f"- Window: {payload['window']['common_start']} to {payload['window']['common_end']}",
        "- Benchmark: official CSI2000 price index (932000).",
        "",
        "| Metric | Current best LGBM | CSI2000 | Excess |",
        "|---|---:|---:|---:|",
        f"| Terminal value | {strategy_metrics['terminal_value']:.2f}x | {benchmark_metrics['terminal_value']:.2f}x | Relative wealth {excess['relative_wealth']:.2f}x |",
        f"| Total return | {strategy_metrics['total_return']:.2%} | {benchmark_metrics['total_return']:.2%} | {excess['cumulative_return_arithmetic_gap']:.2%} |",
        f"| CAGR | {strategy_metrics['annual_return']:.2%} | {benchmark_metrics['annual_return']:.2%} | {excess['annualized_return_gap']:.2%} |",
        f"| Max drawdown | {strategy_metrics['max_drawdown']:.2%} | {benchmark_metrics['max_drawdown']:.2%} | - |",
        "",
        f"- Relative cumulative return: {excess['relative_cumulative_return']:.2%}",
        f"- Information ratio: {excess['information_ratio']:.3f}",
        f"- Beta versus CSI2000: {excess['beta']:.3f}",
        f"- Annualized Jensen alpha (0% rate): {excess['annualized_jensen_alpha_zero_rate']:.2%}",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
