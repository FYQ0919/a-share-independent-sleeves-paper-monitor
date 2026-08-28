from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


START = date(2020, 1, 1)
RESEARCH_RESULT = BASE_DIR / "data" / "qlib_lgbm_ranker_research_v1.json"
OUTPUT_STEM = BASE_DIR / "reports" / "lgbm_vs_nasdaq_2020_2026"
NASDAQ_URL = (
    "https://api.nasdaq.com/api/quote/COMP/historical"
    "?assetclass=index&fromdate=2019-12-31&todate={end}&limit=4000"
)


def fetch_nasdaq(end: date) -> pd.DataFrame:
    url = NASDAQ_URL.format(end=end.isoformat())
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


def curve_metrics(values: pd.Series, dates: pd.Series) -> dict[str, float | str]:
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


def yearly_returns(curve: pd.DataFrame, value_column: str) -> dict[str, float]:
    indexed = curve.set_index("date")[value_column]
    output = {}
    for year, values in indexed.groupby(indexed.index.year):
        output[str(year)] = float(values.iloc[-1] / values.iloc[0] - 1)
    return output


def main() -> None:
    research = json.loads(RESEARCH_RESULT.read_text(encoding="utf-8"))
    winner = research["selection_winner"]
    if winner["name"] != "lgbm_l7_blend40":
        raise RuntimeError("冻结的LGBM研究冠军与预期不一致")

    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    end = pd.Timestamp(history["date"].max()).date()
    history = history[history["date"] <= pd.Timestamp(end)].copy()
    frame, feature_columns = build_alpha158_lite(history)

    prediction, audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, winner["config"], START, end
    )
    if not audit["maturity_audit"]:
        raise RuntimeError("没有生成LGBM月度滚动模型")
    violations = [
        row for row in audit["maturity_audit"]
        if row["max_training_label_end_date"] >= row["prediction_start"]
    ]
    if violations:
        raise RuntimeError(f"发现未成熟标签进入训练集: {violations[:3]}")

    baseline = baseline_scores(frame, START, end)
    scores = blend_scores(frame, prediction, baseline, winner["config"]["baseline_weight"])
    result = topk_dropout_backtest(history, scores, START, end)
    lgbm_curve = normalized_curve(result, "strategy_return").rename("lgbm").reset_index()
    lgbm_curve.columns = ["date", "lgbm"]
    lgbm_curve["date"] = pd.to_datetime(lgbm_curve["date"])

    nasdaq = fetch_nasdaq(end)
    nasdaq = nasdaq[
        (nasdaq["date"] >= lgbm_curve["date"].min())
        & (nasdaq["date"] <= lgbm_curve["date"].max())
    ].copy()
    if nasdaq.empty:
        raise RuntimeError("纳斯达克综合指数未覆盖LGBM回测区间")
    nasdaq["nasdaq"] = nasdaq["nasdaq_close"].div(nasdaq["nasdaq_close"].iloc[0])

    display_end = min(lgbm_curve["date"].max(), nasdaq["date"].max())
    lgbm_curve = lgbm_curve[lgbm_curve["date"] <= display_end].copy()
    nasdaq = nasdaq[nasdaq["date"] <= display_end].copy()
    calendar = pd.DataFrame({
        "date": pd.date_range(
            max(lgbm_curve["date"].min(), nasdaq["date"].min()), display_end, freq="D"
        )
    })
    comparison = (
        calendar.merge(lgbm_curve, on="date", how="left")
        .merge(nasdaq[["date", "nasdaq"]], on="date", how="left")
        .set_index("date")
        .ffill()
        .dropna()
        .reset_index()
    )
    comparison.to_csv(
        OUTPUT_STEM.with_suffix(".csv"), index=False, encoding="utf-8-sig", float_format="%.10f"
    )

    lgbm_metrics = curve_metrics(lgbm_curve["lgbm"], lgbm_curve["date"])
    nasdaq_metrics = curve_metrics(nasdaq["nasdaq"], nasdaq["date"])
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "retrospective_walk_forward_research_only",
        "window": {
            "requested_start": START.isoformat(),
            "common_curve_start": comparison["date"].iloc[0].date().isoformat(),
            "common_curve_end": comparison["date"].iloc[-1].date().isoformat(),
        },
        "strategy": {
            "name": "LGBM LambdaRank Top5",
            "winner": winner["name"],
            "config": winner["config"],
            "feature_count": len(feature_columns),
            "model_refits": audit["refit_months"],
            "execution": "signal at close; execute next open; Top5; 10-session evaluation; 12bp one-way cost",
            "metrics": lgbm_metrics,
            "yearly_returns": yearly_returns(lgbm_curve, "lgbm"),
            "backtest_summary": research_summary(result, START, end),
        },
        "nasdaq": {
            "name": "Nasdaq Composite",
            "symbol": "COMP",
            "return_type": "price return; excludes dividends, fees, taxes, and FX",
            "source": NASDAQ_URL.format(end=end.isoformat()),
            "metrics": nasdaq_metrics,
            "yearly_returns": yearly_returns(nasdaq[["date", "nasdaq"]], "nasdaq"),
        },
        "causality_audit": {
            "passed": True,
            "rule": "every training label_end_date must be strictly earlier than prediction month start",
            "monthly_refits_checked": len(audit["maturity_audit"]),
            "first_refit": audit["maturity_audit"][0],
            "last_refit": audit["maturity_audit"][-1],
        },
        "universe_snapshot_run_id": snapshot_id,
        "production_change": False,
        "warnings": warnings + [
            "This is a retrospective walk-forward evaluation, not a fresh untouched out-of-sample test.",
            "The current Top50 universe is backfilled through history and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "Valuation fields do not have certified per-field available_date timestamps.",
            "China and US holidays differ; the display curve forward-fills each market independently.",
        ],
    }
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "json": str(OUTPUT_STEM.with_suffix(".json")),
        "csv": str(OUTPUT_STEM.with_suffix(".csv")),
        "window": payload["window"],
        "lgbm": lgbm_metrics,
        "nasdaq": nasdaq_metrics,
        "model_refits": audit["refit_months"],
        "causality_audit": payload["causality_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
