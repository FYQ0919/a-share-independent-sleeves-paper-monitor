from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR, settings
from app.storage import Storage
from optimize_factor_strategy import (
    END,
    MODELS,
    START,
    VALIDATION_END,
    VALIDATION_START,
    load_current_history,
)


OUTPUT_DIR = BASE_DIR / "reports"
PERIODS = {
    "decade": {
        "start": START,
        "end": END,
        "nasdaq_from": "2016-08-24",
        "nasdaq_to": END.isoformat(),
        "output_stem": "decade_strategy_vs_nasdaq",
        "sample_status": (
            "Full research window combining training, validation, and a previously revealed "
            "holdout; not an untouched out-of-sample test"
        ),
    },
    "validation": {
        "start": VALIDATION_START,
        "end": VALIDATION_END,
        "nasdaq_from": "2020-12-31",
        "nasdaq_to": "2024-01-02",
        "output_stem": "validation_strategy_vs_nasdaq",
        "sample_status": "Validation period used in model selection; not untouched out-of-sample",
    },
    "test": {
        "start": date(2024, 1, 1),
        "end": END,
        "nasdaq_from": "2023-12-29",
        "nasdaq_to": END.isoformat(),
        "output_stem": "test_strategy_vs_nasdaq",
        "sample_status": "Untouched model-selection test period",
    },
}


def frozen_research_candidates(generated_at: str) -> tuple[list[dict], str]:
    cutoff = pd.Timestamp(generated_at)
    eligible = [
        run
        for run in Storage(settings.database_path).history(200)
        if len(run.get("candidates", [])) >= 50
        and pd.Timestamp(run["created_at"]) <= cutoff
    ]
    if not eligible:
        raise RuntimeError("找不到因子选择时使用的冻结Top50快照")
    snapshot = max(eligible, key=lambda run: pd.Timestamp(run["created_at"]))
    return snapshot["candidates"][:50], snapshot["run_id"]


def run_selected_strategy() -> tuple[
    pd.DataFrame,
    str,
    dict[str, float],
    list[str],
    str,
]:
    optimization_path = BASE_DIR / "data" / "factor_optimization.json"
    optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
    model_name = optimization["selected"]["name"]
    weights = MODELS[model_name]
    candidates, snapshot_id = frozen_research_candidates(optimization["generated_at"])

    history, _, warnings = load_current_history(candidates)
    loaded_codes = sorted(history["code"].unique().tolist())
    engine = BacktestEngine()
    feature_frame = engine._features(history)
    engine._features = lambda _: feature_frame
    engine._factor_weights_override = weights
    result = engine.run(
        history,
        BacktestConfig(
            mode="live",
            start_date=START,
            end_date=END,
            codes=loaded_codes,
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
        ),
        warnings,
    )
    curves = pd.DataFrame(result["curves"])
    curves["date"] = pd.to_datetime(curves["date"])
    return curves[["date", "strategy"]], model_name, weights, warnings, snapshot_id


def fetch_nasdaq_composite(url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
    rows = response.json()["data"]["tradesTable"]["rows"]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["nasdaq_close"] = pd.to_numeric(
        frame["close"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    return (
        frame[["date", "nasdaq_close"]]
        .dropna()
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def normalized_period(
    frame: pd.DataFrame,
    value_column: str,
    period_start: date,
    period_end: date,
) -> pd.DataFrame:
    exact = frame[frame["date"] == pd.Timestamp(period_start)]
    before = frame[frame["date"] < pd.Timestamp(period_start)]
    period = frame[
        (frame["date"] >= pd.Timestamp(period_start))
        & (frame["date"] <= pd.Timestamp(period_end))
    ].copy()
    if period.empty or (exact.empty and before.empty):
        raise RuntimeError(f"{value_column} does not cover the validation window")
    base_row = exact.iloc[-1] if not exact.empty else before.iloc[-1]
    base_value = float(base_row[value_column])
    baseline = pd.DataFrame(
        {"date": [base_row["date"]], value_column: [base_value]}
    )
    normalized = pd.concat([baseline, period], ignore_index=True).drop_duplicates(
        "date", keep="first"
    )
    normalized[value_column] = normalized[value_column].astype(float).div(base_value)
    return normalized


def curve_metrics(values: pd.Series, dates: pd.Series) -> dict[str, float | str]:
    returns = values.pct_change().dropna()
    drawdown = values.div(values.cummax()).sub(1)
    elapsed_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    volatility = float(returns.std(ddof=0) * np.sqrt(252))
    annualized_mean = float(returns.mean() * 252)
    return {
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "start_value": float(values.iloc[0]),
        "end_value": float(values.iloc[-1]),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "annual_return": float((values.iloc[-1] / values.iloc[0]) ** (1 / elapsed_years) - 1),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": volatility,
        "sharpe_zero_rate": annualized_mean / volatility if volatility > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=sorted(PERIODS), default="validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    period = PERIODS[args.period]
    period_start = period["start"]
    period_end = period["end"]
    output_stem = period["output_stem"]
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    json_path = OUTPUT_DIR / f"{output_stem}.json"
    nasdaq_url = (
        "https://api.nasdaq.com/api/quote/COMP/historical"
        f"?assetclass=index&fromdate={period['nasdaq_from']}"
        f"&todate={period['nasdaq_to']}&limit=4000"
    )
    strategy, model_name, weights, warnings, snapshot_id = run_selected_strategy()
    nasdaq = fetch_nasdaq_composite(nasdaq_url)
    strategy = normalized_period(strategy, "strategy", period_start, period_end)
    nasdaq = normalized_period(nasdaq, "nasdaq_close", period_start, period_end).rename(
        columns={"nasdaq_close": "nasdaq"}
    )

    calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                min(strategy["date"].min(), nasdaq["date"].min()),
                pd.Timestamp(period_end),
                freq="D",
            )
        }
    )
    comparison = (
        calendar.merge(strategy, on="date", how="left")
        .merge(nasdaq, on="date", how="left")
        .set_index("date")
        .ffill()
        .dropna()
        .reset_index()
    )
    comparison["date"] = comparison["date"].dt.strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.8f")

    strategy_metrics = curve_metrics(strategy["strategy"], strategy["date"])
    nasdaq_metrics = curve_metrics(nasdaq["nasdaq"], nasdaq["date"])
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "period": args.period,
        "sample_status": period["sample_status"],
        "window": {
            "start": period_start.isoformat(),
            "end": period_end.isoformat(),
        },
        "normalization": f"Use the last available observation before {period_start.isoformat()} as 1.0",
        "strategy": {
            "model": model_name,
            "weights": weights,
            "universe_snapshot_run_id": snapshot_id,
            "execution": "close signal, next-open execution, 12 bps one-way cost",
            "macro_policy": "observation only; automatic execution disabled",
            "metrics": strategy_metrics,
        },
        "nasdaq": {
            "name": "Nasdaq Composite",
            "symbol": "COMP",
            "return_type": "price return, excluding dividends, fees, taxes, and FX",
            "source": nasdaq_url,
            "metrics": nasdaq_metrics,
        },
        "comparison": {
            "strategy_minus_nasdaq_total_return": (
                strategy_metrics["total_return"] - nasdaq_metrics["total_return"]
            )
        },
        "warnings": warnings
        + [
            period["sample_status"],
            f"The strategy universe is frozen to research snapshot {snapshot_id}.",
            "The strategy uses the current Top50 universe backfilled through history and is subject to constituent and survivorship bias.",
        ],
        "files": {"curve_csv": str(csv_path)},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"json": str(json_path), **payload["comparison"], "strategy": strategy_metrics, "nasdaq": nasdaq_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
