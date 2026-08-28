from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
LGBM_CURVE_PATH = BASE_DIR / "reports" / "qlib_lgbm_ranker_2020_curves.csv"
OUTPUT_STEM = BASE_DIR / "reports" / "lgbm_vs_nasdaq_2020"
NASDAQ_URL = (
    "https://api.nasdaq.com/api/quote/COMP/historical"
    "?assetclass=index&fromdate=2019-12-31&todate=2020-12-31&limit=1000"
)


def fetch_nasdaq() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
        response = client.get(NASDAQ_URL)
        response.raise_for_status()
    rows = response.json()["data"]["tradesTable"]["rows"]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["nasdaq_close"] = pd.to_numeric(
        frame["close"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    return (
        frame[["date", "nasdaq_close"]]
        .dropna()
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def metrics(values: pd.Series, dates: pd.Series) -> dict[str, float | str]:
    returns = values.pct_change().dropna()
    drawdown = values.div(values.cummax()).sub(1)
    elapsed_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    volatility = float(returns.std(ddof=0) * np.sqrt(252))
    annualized_mean = float(returns.mean() * 252)
    return {
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "annual_return": float((values.iloc[-1] / values.iloc[0]) ** (1 / elapsed_years) - 1),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": volatility,
        "sharpe_zero_rate": annualized_mean / volatility if volatility else 0.0,
    }


def main() -> None:
    lgbm = pd.read_csv(LGBM_CURVE_PATH, parse_dates=["date"])[
        ["date", "lgbm_ranker_top5"]
    ].copy()
    start = lgbm["date"].min()
    end = lgbm["date"].max()
    lgbm["lgbm"] = lgbm["lgbm_ranker_top5"].div(lgbm["lgbm_ranker_top5"].iloc[0])

    nasdaq = fetch_nasdaq()
    nasdaq = nasdaq[(nasdaq["date"] >= start) & (nasdaq["date"] <= end)].copy()
    if nasdaq.empty:
        raise RuntimeError("纳斯达克综合指数数据未覆盖 LGBM 诊断窗口")
    nasdaq["nasdaq"] = nasdaq["nasdaq_close"].div(nasdaq["nasdaq_close"].iloc[0])

    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    comparison = (
        calendar.merge(lgbm[["date", "lgbm"]], on="date", how="left")
        .merge(nasdaq[["date", "nasdaq"]], on="date", how="left")
        .set_index("date")
        .ffill()
        .dropna()
        .reset_index()
    )
    comparison.to_csv(
        OUTPUT_STEM.with_suffix(".csv"), index=False, encoding="utf-8-sig", float_format="%.8f"
    )

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "sample_status": "2020 revealed out-of-selection diagnostic; not a fresh untouched test",
        "lgbm": {
            "name": "LGBM LambdaRank Top5",
            "execution": "close signal; next-open execution; 12 bps one-way cost",
            "metrics": metrics(lgbm["lgbm"], lgbm["date"]),
        },
        "nasdaq": {
            "name": "Nasdaq Composite",
            "symbol": "COMP",
            "return_type": "price return; excludes dividends, fees, taxes, and FX",
            "source": NASDAQ_URL,
            "metrics": metrics(nasdaq["nasdaq"], nasdaq["date"]),
        },
        "limitations": [
            "LGBM uses a frozen current-Top50 universe backfilled through history.",
            "The universe has constituent and survivorship bias.",
            "Current-vintage adjusted prices and valuation availability are not strict point-in-time data.",
            "China and US trading holidays differ; the display curve forward-fills each market independently.",
        ],
    }
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
