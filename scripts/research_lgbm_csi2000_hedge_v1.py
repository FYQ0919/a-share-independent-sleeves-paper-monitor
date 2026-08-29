from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CURVES = ROOT / "reports" / "current_best_lgbm_vs_csi300_csi2000_2020_now.csv"
OUTPUT_CURVES = ROOT / "reports" / "lgbm_csi2000_hedge_v1_curves.csv"
OUTPUT_JSON = ROOT / "data" / "lgbm_csi2000_hedge_v1.json"
OUTPUT_REPORT = ROOT / "reports" / "lgbm_csi2000_hedge_v1.md"

LOOKBACK = 120
THRESHOLD = 1.0
HEDGE_RATIO = 0.75
HEDGE_CHANGE_COST_BPS = 2.0
SIGNAL_LAG = 1


def curve_metrics(values: pd.Series) -> dict[str, float | int | str]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("at least two curve observations are required")
    returns = clean.pct_change().dropna()
    elapsed_years = max((clean.index[-1] - clean.index[0]).days / 365.2425, 1 / 365.2425)
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    return {
        "start_date": clean.index[0].date().isoformat(),
        "end_date": clean.index[-1].date().isoformat(),
        "observations": int(len(clean)),
        "terminal_value": float(clean.iloc[-1] / clean.iloc[0]),
        "total_return": total_return,
        "annual_return": float((1.0 + total_return) ** (1.0 / elapsed_years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(clean.div(clean.cummax()).sub(1.0).min()),
    }


def apply_csi2000_hedge(curves: pd.DataFrame) -> pd.DataFrame:
    required = {"current_best_lgbm", "csi2000", "csi300"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = curves.loc[:, sorted(required)].astype(float).copy()
    benchmark = frame["csi2000"]
    moving_average = benchmark.rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    close_signal = benchmark.div(moving_average).lt(THRESHOLD).astype(float)

    # The return recorded at T is close[T] / close[T-1] - 1. Using signal[T-1]
    # ensures every hedge return depends only on information available by T-1.
    hedge_position = close_signal.shift(SIGNAL_LAG).fillna(0.0).mul(HEDGE_RATIO)
    strategy_return = frame["current_best_lgbm"].pct_change().fillna(0.0)
    csi2000_return = benchmark.pct_change().fillna(0.0)
    hedge_pnl = hedge_position.mul(csi2000_return).mul(-1.0)
    hedge_cost = hedge_position.diff().abs().fillna(hedge_position.abs()).mul(
        HEDGE_CHANGE_COST_BPS / 10_000.0
    )
    hedged_return = strategy_return.add(hedge_pnl).sub(hedge_cost)
    hedged_curve = (1.0 + hedged_return).cumprod()
    hedged_curve.iloc[0] = 1.0

    output = frame[["current_best_lgbm", "csi2000", "csi300"]].copy()
    output["lgbm_csi2000_hedged"] = hedged_curve
    output["csi2000_hedge_ratio"] = hedge_position
    output["csi2000_close_signal"] = close_signal
    output["csi2000_hedge_pnl"] = hedge_pnl
    output["csi2000_hedge_cost"] = hedge_cost
    return output


def main() -> None:
    source = pd.read_csv(INPUT_CURVES, parse_dates=["date"]).set_index("date").sort_index()
    curves = apply_csi2000_hedge(source)
    curves.index.name = "date"
    OUTPUT_CURVES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")

    base_metrics = curve_metrics(curves["current_best_lgbm"])
    hedge_metrics = curve_metrics(curves["lgbm_csi2000_hedged"])
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic_not_promoted",
        "strategy": "current 75/25 LGBM plus causal CSI2000 short-return hedge proxy",
        "window": {
            "start": curves.index[0].date().isoformat(),
            "end": curves.index[-1].date().isoformat(),
            "trading_days": int(len(curves)),
        },
        "parameters": {
            "lookback": LOOKBACK,
            "threshold": THRESHOLD,
            "hedge_ratio": HEDGE_RATIO,
            "signal_lag_sessions": SIGNAL_LAG,
            "hedge_change_cost_bps": HEDGE_CHANGE_COST_BPS,
            "selection": "Fixed from the existing CSI300 hedge rule; not retuned on 2020-2026 CSI2000 results.",
        },
        "metrics": {
            "base_lgbm": base_metrics,
            "csi2000_hedged_lgbm": hedge_metrics,
            "cagr_change": float(hedge_metrics["annual_return"] - base_metrics["annual_return"]),
            "max_drawdown_change": float(hedge_metrics["max_drawdown"] - base_metrics["max_drawdown"]),
            "average_hedge_ratio": float(curves["csi2000_hedge_ratio"].mean()),
            "position_changes": int(curves["csi2000_hedge_ratio"].diff().abs().gt(0).sum()),
        },
        "causality_audit": {
            "signal": "CSI2000 close divided by trailing MA120, using observations through T close only",
            "return_alignment": "signal[T-1] is applied to CSI2000 close[T-1]-to-close[T] return",
            "future_data_used": False,
        },
        "production_change": False,
        "artifacts": {
            "curves": str(OUTPUT_CURVES.relative_to(ROOT)),
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
        "limitations": [
            "China has no standard CSI2000 stock-index futures contract; the hedge is a short CSI2000 price-return research proxy.",
            "Historical CSI2000 open prices were unavailable, so hedge PnL uses close-to-close returns and is not an executable next-open replay.",
            "Borrow availability, borrow fees, financing, slippage, tracking error and margin are not modeled; only 2 bp per hedge-position change is charged.",
            "CSI2000 after 2025-12-31 uses an ETF-calibrated proxy.",
            "The 2020-2026 period is repeatedly revealed and is not an untouched out-of-sample holdout.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# LGBM with CSI2000 hedge proxy, 2020 to latest",
        "",
        f"- Status: {payload['status']}; production and paper trading unchanged.",
        f"- Fixed rule: MA{LOOKBACK}, threshold {THRESHOLD:.2f}, hedge ratio {HEDGE_RATIO:.0%}, one-session lag.",
        "",
        "| Curve | Terminal value | CAGR | Max drawdown | Sharpe |",
        "|---|---:|---:|---:|---:|",
        f"| Base 75/25 LGBM | {base_metrics['terminal_value']:.2f}x | {base_metrics['annual_return']:.2%} | {base_metrics['max_drawdown']:.2%} | {base_metrics['sharpe_zero_rate']:.3f} |",
        f"| LGBM + CSI2000 hedge proxy | {hedge_metrics['terminal_value']:.2f}x | {hedge_metrics['annual_return']:.2%} | {hedge_metrics['max_drawdown']:.2%} | {hedge_metrics['sharpe_zero_rate']:.3f} |",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
