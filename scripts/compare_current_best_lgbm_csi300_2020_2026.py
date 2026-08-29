from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_SOURCE = ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
STRATEGY_AUDIT = ROOT / "data" / "overnight_factor_lgbm_2020_2026.json"
BENCHMARK_CACHE = ROOT / "data" / "cache" / "index" / "csi300_2020_2026.csv"
OUTPUT_JSON = ROOT / "data" / "current_best_lgbm_vs_csi300_2020_2026.json"
OUTPUT_CSV = ROOT / "reports" / "current_best_lgbm_vs_csi300_2020_2026.csv"
OUTPUT_REPORT = ROOT / "reports" / "current_best_lgbm_vs_csi300_2020_2026.md"
TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def metrics(values: pd.Series) -> dict[str, float | str | int]:
    clean = values.dropna().astype(float)
    returns = clean.pct_change().dropna()
    elapsed_years = max((clean.index[-1] - clean.index[0]).days / 365.2425, 1 / 365.2425)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    return {
        "start_date": clean.index[0].date().isoformat(),
        "end_date": clean.index[-1].date().isoformat(),
        "observations": int(len(clean)),
        "terminal_value": float(clean.iloc[-1] / clean.iloc[0]),
        "total_return": total_return,
        "annual_return": float((1.0 + total_return) ** (1.0 / elapsed_years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": (
            float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0
        ),
        "max_drawdown": float(clean.div(clean.cummax()).sub(1.0).min()),
    }


def load_strategy() -> pd.Series:
    audit = json.loads(STRATEGY_AUDIT.read_text(encoding="utf-8"))
    if audit.get("selected_for_forward_research") != "previous_lgbm":
        raise RuntimeError("latest factor research no longer selects previous_lgbm")
    frame = pd.read_csv(STRATEGY_SOURCE, parse_dates=["date"])
    values = frame.set_index("date")["previous_lgbm"].sort_index().astype(float)
    if values.empty or not np.isclose(float(values.iloc[0]), 1.0):
        raise RuntimeError("current best strategy curve lacks a normalized start")
    return values


def fetch_csi300(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    params = {
        "param": (
            f"sh000300,day,{(start - pd.Timedelta(days=5)).date()},"
            f"{end.date()},2000,qfq"
        )
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    try:
        with httpx.Client(timeout=30, trust_env=False, headers=headers) as client:
            response = client.get(TENCENT_URL, params=params)
            response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError(f"Tencent CSI300 response error: {payload.get('msg', 'unknown')}")
        node = data.get("sh000300", {})
        rows = node.get("qfqday") or node.get("day") or []
        if len(rows) < 2:
            raise RuntimeError("Tencent CSI300 response has insufficient rows")
        frame = pd.DataFrame(
            rows,
            columns=["date", "open", "close", "high", "low", "volume", *(["extra"] if len(rows[0]) == 7 else [])],
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame[["date", "close"]].dropna().drop_duplicates("date").sort_values("date")
        BENCHMARK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(BENCHMARK_CACHE, index=False, encoding="utf-8-sig", float_format="%.6f")
        status = {
            "mode": "live_tencent_api",
            "source": f"{TENCENT_URL}?param=sh000300,day,...",
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
    except Exception as exc:
        if not BENCHMARK_CACHE.exists():
            raise
        frame = pd.read_csv(BENCHMARK_CACHE, parse_dates=["date"])
        status = {
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
        raise RuntimeError("CSI300 has insufficient observations in the strategy window")
    return values, status


def main() -> None:
    strategy = load_strategy()
    benchmark, source_status = fetch_csi300(strategy.index.min(), strategy.index.max())
    common_dates = strategy.index.intersection(benchmark.index).sort_values()
    if len(common_dates) < 2:
        raise RuntimeError("strategy and CSI300 have insufficient common trading dates")
    strategy = strategy.reindex(common_dates).astype(float)
    benchmark = benchmark.reindex(common_dates).astype(float)
    strategy = strategy.div(float(strategy.iloc[0]))
    benchmark = benchmark.div(float(benchmark.iloc[0]))

    curve = pd.DataFrame(
        {"current_best_lgbm": strategy, "csi300": benchmark}, index=common_dates
    )
    curve["relative_wealth"] = curve["current_best_lgbm"].div(curve["csi300"])
    curve.index.name = "date"
    curve.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")

    strategy_metrics = metrics(strategy)
    benchmark_metrics = metrics(benchmark)
    daily = curve[["current_best_lgbm", "csi300"]].pct_change().dropna()
    active = daily["current_best_lgbm"].sub(daily["csi300"])
    tracking_error = float(active.std(ddof=0) * np.sqrt(252.0))
    information_ratio = (
        float(active.mean() * 252.0 / tracking_error) if tracking_error > 0 else 0.0
    )
    benchmark_variance = float(daily["csi300"].var(ddof=0))
    beta = (
        float(daily["current_best_lgbm"].cov(daily["csi300"]) / benchmark_variance)
        if benchmark_variance > 0
        else 0.0
    )
    annualized_alpha = float(
        (daily["current_best_lgbm"].mean() - beta * daily["csi300"].mean()) * 252.0
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
        "information_ratio": information_ratio,
        "beta": beta,
        "annualized_jensen_alpha_zero_rate": annualized_alpha,
    }

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic_not_promoted",
        "window": {
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
            "name": "沪深300",
            "symbol": "000300",
            "return_type": "price return; excludes dividends, fees and taxes",
            "source_status": source_status,
            "metrics": benchmark_metrics,
        },
        "excess": excess,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
        "limitations": [
            "The current Top50 universe is backfilled and has survivorship and constituent bias.",
            "Current-vintage adjusted A-share prices are not strict point-in-time prices.",
            "The 2020-2026 period has been repeatedly revealed and is not an untouched holdout.",
            "CSI300 is a price index; a total-return index would be a stricter investor-return benchmark.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Current best LGBM vs CSI300, 2020-2026",
        "",
        f"- Window: {payload['window']['common_start']} to {payload['window']['common_end']}",
        "- Main A-share alpha benchmark: CSI300 price index (000300).",
        "",
        "| Metric | Current best LGBM | CSI300 | Excess |",
        "|---|---:|---:|---:|",
        f"| Terminal value | {strategy_metrics['terminal_value']:.2f}x | {benchmark_metrics['terminal_value']:.2f}x | Relative wealth {excess['relative_wealth']:.2f}x |",
        f"| Total return | {strategy_metrics['total_return']:.2%} | {benchmark_metrics['total_return']:.2%} | {excess['cumulative_return_arithmetic_gap']:.2%} |",
        f"| CAGR | {strategy_metrics['annual_return']:.2%} | {benchmark_metrics['annual_return']:.2%} | {excess['annualized_return_gap']:.2%} |",
        f"| Max drawdown | {strategy_metrics['max_drawdown']:.2%} | {benchmark_metrics['max_drawdown']:.2%} | - |",
        "",
        f"- Relative cumulative return: {excess['relative_cumulative_return']:.2%}",
        f"- Information ratio: {excess['information_ratio']:.3f}",
        f"- Beta versus CSI300: {excess['beta']:.3f}",
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
