from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "cache" / "backtest"
OUTPUT_PATH = BASE_DIR / "data" / "strategy_diagnostics.json"
START = date(2024, 1, 1)
SPLIT = date(2025, 7, 1)
END = date(2026, 8, 25)


WEIGHT_SETS = {
    "current": {"momentum": 0.35, "trend": 0.15, "liquidity": 0.15, "value": 0.15, "risk": 0.20},
    "equal_factor": {"momentum": 0.20, "trend": 0.20, "liquidity": 0.20, "value": 0.20, "risk": 0.20},
    "momentum": {"momentum": 0.65, "trend": 0.20, "liquidity": 0.05, "value": 0.05, "risk": 0.05},
    "value_risk": {"momentum": 0.05, "trend": 0.05, "liquidity": 0.05, "value": 0.40, "risk": 0.45},
    "risk_value": {"momentum": 0.00, "trend": 0.00, "liquidity": 0.05, "value": 0.30, "risk": 0.65},
    "risk_only": {"momentum": 0.00, "trend": 0.00, "liquidity": 0.00, "value": 0.00, "risk": 1.00},
    "train_ic": {"momentum": 0.00, "trend": 0.33, "liquidity": 0.00, "value": 0.15, "risk": 0.52},
    "risk_defensive": {"momentum": 0.05, "trend": 0.10, "liquidity": 0.00, "value": 0.15, "risk": 0.70},
    "reversal_value": {"momentum": -0.15, "trend": -0.05, "liquidity": 0.05, "value": 0.45, "risk": 0.70},
    "contrarian": {"medium_reversal": 0.40, "inefficiency": 0.30, "liquidity_cooling": 0.30},
    "contrarian_risk": {"medium_reversal": 0.35, "inefficiency": 0.25, "liquidity_cooling": 0.25, "risk": 0.15},
    "contrarian_value": {"medium_reversal": 0.35, "inefficiency": 0.25, "liquidity_cooling": 0.20, "risk": 0.10, "value": 0.10},
}


def load_history() -> pd.DataFrame:
    frames = []
    for path in sorted(CACHE_DIR.glob("*_qfq.csv")):
        frame = pd.read_csv(path, dtype={"code": str})
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)
    if not frames:
        raise RuntimeError("没有找到历史缓存")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["code", "date"])
        .drop_duplicates(["code", "date"], keep="last")
        .reset_index(drop=True)
    )


def run_case(history: pd.DataFrame, weights: dict[str, float], config: BacktestConfig):
    engine = BacktestEngine()
    engine._factor_weights_override = weights
    return engine.run(history, config, [])


def summarize_result(result: dict, name: str):
    metrics = result["metrics"]
    config = result["config"]
    return {
        "name": name,
        "top_n": config["top_n"],
        "rebalance_days": config["rebalance_days"],
        "total_return": metrics["total_return"],
        "benchmark_return": metrics["benchmark_return"],
        "excess_return": metrics["excess_return"],
        "annual_return": metrics["annual_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "information_ratio": metrics["information_ratio"],
        "average_turnover": metrics["average_turnover"],
        "total_cost": metrics["total_cost"],
    }


def period_summary(result: dict, start: date, end: date):
    if result.get("return_periods"):
        periods = pd.DataFrame(result["return_periods"])
        periods["period_start"] = pd.to_datetime(periods["period_start"])
        periods["period_end"] = pd.to_datetime(periods["period_end"])
        sample = periods[
            (periods["period_end"] >= pd.Timestamp(start))
            & (periods["period_end"] <= pd.Timestamp(end))
        ].copy()
        if sample.empty:
            raise ValueError("诊断分段内没有已完结的收益区间")
        strategy_returns = sample["strategy_return"].astype(float)
        benchmark_returns = sample["benchmark_return"].astype(float)
        strategy_values = (1 + strategy_returns).cumprod()
        benchmark_values = (1 + benchmark_returns).cumprod()
        total_return = float(strategy_values.iloc[-1] - 1)
        benchmark_return = float(benchmark_values.iloc[-1] - 1)
        volatility = float(strategy_returns.std(ddof=0) * np.sqrt(252))
        annualized_mean = float(strategy_returns.mean() * 252)
        max_drawdown = float(strategy_values.div(strategy_values.cummax()).sub(1).min())
        excess_daily = strategy_returns.reset_index(drop=True) - benchmark_returns.reset_index(drop=True)
        information_ratio = (
            float(excess_daily.mean() / excess_daily.std(ddof=0) * np.sqrt(252))
            if excess_daily.std(ddof=0) > 0
            else 0.0
        )
        return {
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "excess_return": total_return - benchmark_return,
            "annual_return": (1 + total_return) ** (252 / len(sample)) - 1,
            "max_drawdown": max_drawdown,
            "sharpe": annualized_mean / volatility if volatility > 0 else 0.0,
            "information_ratio": information_ratio,
            "return_periods": int(len(sample)),
            "first_period_end": sample.iloc[0]["period_end"].date().isoformat(),
            "last_period_end": sample.iloc[-1]["period_end"].date().isoformat(),
        }

    curves = pd.DataFrame(result["curves"])
    curves["date"] = pd.to_datetime(curves["date"])
    before = curves[curves["date"] < pd.Timestamp(start)]
    period = curves[(curves["date"] >= pd.Timestamp(start)) & (curves["date"] <= pd.Timestamp(end))]
    if period.empty:
        raise ValueError("诊断分段内没有净值数据")
    strategy_base = float(before.iloc[-1]["strategy"]) if not before.empty else 1.0
    benchmark_base = float(before.iloc[-1]["benchmark"]) if not before.empty else 1.0
    strategy_values = pd.concat([pd.Series([strategy_base]), period["strategy"].reset_index(drop=True)], ignore_index=True)
    benchmark_values = pd.concat([pd.Series([benchmark_base]), period["benchmark"].reset_index(drop=True)], ignore_index=True)
    strategy_returns = strategy_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()
    total_return = float(strategy_values.iloc[-1] / strategy_base - 1)
    benchmark_return = float(benchmark_values.iloc[-1] / benchmark_base - 1)
    volatility = float(strategy_returns.std(ddof=0) * np.sqrt(252))
    annualized_mean = float(strategy_returns.mean() * 252)
    normalized = strategy_values / strategy_base
    max_drawdown = float(normalized.div(normalized.cummax()).sub(1).min())
    excess_daily = strategy_returns - benchmark_returns
    information_ratio = (
        float(excess_daily.mean() / excess_daily.std(ddof=0) * np.sqrt(252))
        if excess_daily.std(ddof=0) > 0
        else 0.0
    )
    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "annual_return": (1 + total_return) ** (252 / len(period)) - 1,
        "max_drawdown": max_drawdown,
        "sharpe": annualized_mean / volatility if volatility > 0 else 0.0,
        "information_ratio": information_ratio,
    }


def factor_ic(history: pd.DataFrame, rebalance_days: int = 20):
    engine = BacktestEngine()
    frame = engine._features(history)
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))]
    open_prices = frame.pivot(index="date", columns="code", values="open").reindex(dates)
    future_returns = open_prices.shift(-rebalance_days).div(open_prices).sub(1)
    rows = []
    for signal_date in dates[::rebalance_days]:
        cross = frame[frame["date"] == signal_date].copy()
        cross = cross[(cross["tradestatus"] == 1) & (cross["is_st"] == 0)]
        if len(cross) < 5:
            continue
        scored = engine._score(cross).set_index("code")
        forward = future_returns.loc[signal_date]
        for factor in engine.FACTOR_WEIGHTS:
            joined = pd.concat([scored[f"{factor}_score"], forward], axis=1).dropna()
            if len(joined) >= 5:
                ranked = joined.rank(pct=True)
                rows.append({
                    "date": signal_date,
                    "factor": factor,
                    "ic": ranked.iloc[:, 0].corr(ranked.iloc[:, 1]),
                })
    values = pd.DataFrame(rows)
    output = []
    for factor, group in values.groupby("factor"):
        train = group[group["date"] < pd.Timestamp(SPLIT)]["ic"].dropna()
        valid = group[group["date"] >= pd.Timestamp(SPLIT)]["ic"].dropna()
        output.append({
            "factor": factor,
            "train_mean_ic": float(train.mean()),
            "train_positive_rate": float((train > 0).mean()),
            "validation_mean_ic": float(valid.mean()),
            "validation_positive_rate": float((valid > 0).mean()),
            "observations": int(len(group)),
        })
    return output


def candidate_factor_ic(history: pd.DataFrame, rebalance_days: int = 20):
    engine = BacktestEngine()
    frame = engine._features(history)
    frame["reversal_5"] = -frame["ret_5"]
    frame["momentum_skip_5"] = (1 + frame["ret_60"]).div(1 + frame["ret_5"]).sub(1)
    frame["low_volatility_60"] = -frame["volatility_60"]
    frame["downside_quality_60"] = -frame["downside_volatility_60"]
    frame["trend_efficiency_60"] = frame["ret_60"].div(frame["absolute_return_60"].replace(0, np.nan))
    frame["volume_confirmation"] = frame["ret_20"].mul(
        np.log(frame["amount_20"].div(frame["amount_60"]).clip(lower=0.05))
    )
    frame["liquidity_change"] = frame["amount_20"].div(frame["amount_60"]).sub(1)
    frame["intraday_strength_20_signal"] = frame["intraday_strength_20"]
    frame["drawdown_position_120"] = frame["drawdown_120"]
    signals = [
        "reversal_5", "momentum_skip_5", "low_volatility_60", "downside_quality_60",
        "trend_efficiency_60", "volume_confirmation", "liquidity_change",
        "intraday_strength_20_signal", "drawdown_position_120",
    ]
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))]
    open_prices = frame.pivot(index="date", columns="code", values="open").reindex(dates)
    future_returns = open_prices.shift(-rebalance_days).div(open_prices).sub(1)
    rows = []
    for signal_date in dates[::rebalance_days]:
        cross = frame[frame["date"] == signal_date].set_index("code")
        forward = future_returns.loc[signal_date]
        for signal in signals:
            joined = pd.concat([cross[signal], forward], axis=1).dropna()
            if len(joined) < 5:
                continue
            ranked = joined.rank(pct=True)
            rows.append({"date": signal_date, "factor": signal, "ic": ranked.iloc[:, 0].corr(ranked.iloc[:, 1])})
    values = pd.DataFrame(rows)
    output = []
    for factor, group in values.groupby("factor"):
        train = group[group["date"] < pd.Timestamp(SPLIT)]["ic"].dropna()
        valid = group[group["date"] >= pd.Timestamp(SPLIT)]["ic"].dropna()
        output.append({
            "factor": factor,
            "train_mean_ic": float(train.mean()),
            "train_positive_rate": float((train > 0).mean()),
            "validation_mean_ic": float(valid.mean()),
            "validation_positive_rate": float((valid > 0).mean()),
            "observations": int(len(group)),
        })
    return sorted(output, key=lambda row: min(row["train_mean_ic"], row["validation_mean_ic"]), reverse=True)


def stock_returns(history: pd.DataFrame):
    frame = history[(history["date"] >= pd.Timestamp(START)) & (history["date"] <= pd.Timestamp(END))]
    rows = []
    for code, group in frame.sort_values("date").groupby("code"):
        opens = group["open"].dropna()
        if len(opens) < 2:
            continue
        rows.append({
            "code": code,
            "name": str(group["name"].iloc[-1]),
            "open_to_open_return": float(opens.iloc[-1] / opens.iloc[0] - 1),
        })
    return sorted(rows, key=lambda row: row["open_to_open_return"], reverse=True)


def main():
    history = load_history()
    codes = sorted(history["code"].unique().tolist())
    base = BacktestConfig("live", START, END, codes, 5, 20, 1_000_000, 12)
    comparisons = []
    for name, weights in WEIGHT_SETS.items():
        for top_n in (3, 5, 8, 10):
            for rebalance_days in (10, 20, 40, 60):
                config = replace(base, top_n=top_n, rebalance_days=rebalance_days)
                result = run_case(history, weights, config)
                comparisons.append({
                    "name": name,
                    "weights": weights,
                    "top_n": top_n,
                    "rebalance_days": rebalance_days,
                    "training": period_summary(result, START, date(2025, 6, 30)),
                    "validation": period_summary(result, SPLIT, END),
                    "full": summarize_result(result, name),
                })

    comparisons.sort(
        key=lambda item: (
            item["validation"]["excess_return"],
            item["training"]["excess_return"],
            item["validation"]["sharpe"],
        ),
        reverse=True,
    )
    payload = {
        "period": {"start": START.isoformat(), "split": SPLIT.isoformat(), "end": END.isoformat()},
        "symbols": len(codes),
        "stock_returns": stock_returns(history),
        "factor_ic": factor_ic(history),
        "candidate_factor_ic": candidate_factor_ic(history),
        "comparisons": comparisons,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print("Top validation cases:")
    for item in comparisons[:15]:
        print(
            item["name"],
            f"Top{item['top_n']}",
            f"R{item['rebalance_days']}",
            f"train_excess={item['training']['excess_return']:.2%}",
            f"valid_excess={item['validation']['excess_return']:.2%}",
            f"full_excess={item['full']['excess_return']:.2%}",
            f"valid_sharpe={item['validation']['sharpe']:.2f}",
        )


if __name__ == "__main__":
    main()
