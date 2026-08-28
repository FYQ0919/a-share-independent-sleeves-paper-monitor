from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

import validate_lgbm_paper_hybrid as hybrid
from app.config import BASE_DIR as APP_BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    fit_expanding_lgbm_predictions,
)


START = date(2016, 8, 25)
END = date(2026, 8, 25)
NASDAQ_URL = (
    "https://api.nasdaq.com/api/quote/NDX/historical"
    "?assetclass=index&fromdate={start}&todate={end}&limit=4000"
)
OUTPUT_JSON = APP_BASE_DIR / "reports" / "best_lgbm_vs_nasdaq100_10y.json"
OUTPUT_CSV = APP_BASE_DIR / "reports" / "best_lgbm_vs_nasdaq100_10y.csv"


def fetch_ndx() -> tuple[pd.DataFrame, str]:
    url = NASDAQ_URL.format(start=(pd.Timestamp(START) - pd.Timedelta(days=2)).date(), end=END)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("tradesTable", {}).get("rows", [])
    if not rows:
        raise RuntimeError("NASDAQ NDX historical response has no rows")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["ndx_close"] = pd.to_numeric(
        frame["close"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    frame = frame[["date", "ndx_close"]].dropna().drop_duplicates("date").sort_values("date")
    return frame, url


def curve_metrics(curve: pd.DataFrame, value_column: str) -> dict:
    values = curve[value_column].astype(float)
    returns = values.pct_change().dropna()
    elapsed_years = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days / 365.25, 1 / 365.25)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    return {
        "start_date": curve["date"].iloc[0].date().isoformat(),
        "end_date": curve["date"].iloc[-1].date().isoformat(),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1.0),
        "annual_return": float((values.iloc[-1] / values.iloc[0]) ** (1 / elapsed_years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": float(returns.mean() * 252.0 / volatility) if volatility else 0.0,
        "max_drawdown": float(values.div(values.cummax()).sub(1.0).min()),
    }


def yearly_returns(curve: pd.DataFrame, value_column: str) -> dict:
    indexed = curve.set_index("date")[value_column]
    return {
        str(year): float(group.iloc[-1] / group.iloc[0] - 1.0)
        for year, group in indexed.groupby(indexed.index.year)
    }


def main() -> None:
    # The hybrid simulator reads these module constants for its requested
    # window; setting them here keeps its execution accounting unchanged.
    hybrid.START = START
    hybrid.END = END
    source = json.loads((APP_BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history = hybrid._history_frame(history)
    history = history[history["date"] <= pd.Timestamp(END)].copy()
    frame, feature_columns = build_alpha158_lite(history)
    baseline = baseline_scores(frame, START, END)
    prediction, audit = fit_expanding_lgbm_predictions(frame, feature_columns, WINNER_CONFIG, START, END)
    scores = blend_scores(frame, prediction, baseline, WINNER_CONFIG["baseline_weight"])
    risk_features = hybrid.paper_risk_features(history)
    result = hybrid.simulate_hybrid(history, scores, risk_features, "inverse_vol", False)
    strategy_periods = pd.DataFrame(result["return_periods"])
    strategy_periods["date"] = pd.to_datetime(strategy_periods["period_end"])
    strategy_curve = (1.0 + strategy_periods["strategy_return"].astype(float)).cumprod()
    strategy = pd.DataFrame({"date": strategy_periods["date"], "strategy": strategy_curve})
    ndx, ndx_url = fetch_ndx()
    ndx = ndx[(ndx["date"] >= strategy["date"].min()) & (ndx["date"] <= strategy["date"].max())].copy()
    if ndx.empty:
        raise RuntimeError("NDX does not overlap the strategy curve")
    ndx["ndx"] = ndx["ndx_close"] / ndx["ndx_close"].iloc[0]
    strategy["strategy"] /= strategy["strategy"].iloc[0]
    common = pd.DataFrame({"date": pd.date_range(strategy["date"].min(), min(strategy["date"].max(), ndx["date"].max()), freq="D")})
    common = common.merge(strategy[["date", "strategy"]], on="date", how="left")
    common = common.merge(ndx[["date", "ndx"]], on="date", how="left").sort_values("date").ffill().dropna()
    common = common[common["date"] >= max(strategy["date"].min(), ndx["date"].min())].reset_index(drop=True)
    common[["strategy", "ndx"]] = common[["strategy", "ndx"]].div(common[["strategy", "ndx"]].iloc[0])
    common.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.10f")
    strategy_common = common[["date", "strategy"]]
    ndx_common = common[["date", "ndx"]].rename(columns={"ndx": "strategy"})
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "window": {"requested_start": START.isoformat(), "requested_end": END.isoformat(), "common_start": common["date"].iloc[0].date().isoformat(), "common_end": common["date"].iloc[-1].date().isoformat()},
        "strategy": {
            "name": "Current best candidate: LightGBM Top5 + 60-session inverse-volatility weights",
            "model_version": "lgbm_ranker_top5_v1",
            "config": WINNER_CONFIG,
            "feature_count": len(feature_columns),
            "model_refits": audit["refit_months"],
            "execution": "signal close; next-open; Top5; 10-session rebalance; 12bp one-way cost",
            "metrics": curve_metrics(strategy_common, "strategy"),
            "yearly_returns": yearly_returns(strategy_common, "strategy"),
            "backtest_metrics": result["metrics"],
        },
        "nasdaq100": {
            "name": "Nasdaq-100",
            "symbol": "NDX",
            "return_type": "price return; excludes dividends, fees, taxes and FX",
            "source": ndx_url,
            "metrics": curve_metrics(ndx_common, "strategy"),
            "yearly_returns": yearly_returns(ndx_common, "strategy"),
        },
        "universe_snapshot_run_id": snapshot_id,
        "requested_codes": len(requested_codes),
        "loaded_codes": int(history["code"].nunique()),
        "curve_csv": str(OUTPUT_CSV.relative_to(APP_BASE_DIR)),
        "warnings": warnings + [
            "策略使用当前Top50历史回填，存在成分与存续偏差；不是严格point-in-time收益",
            "NDX为价格指数，不含分红；中美假日不同，曲线按日向前填充仅用于展示",
            "这是近十年回溯式walk-forward研究，不是冻结后的全新盲测",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
