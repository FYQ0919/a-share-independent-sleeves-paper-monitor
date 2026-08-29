from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTION_CURVE = ROOT / "reports" / "double_ensemble_vs_lgbm_2016_2020_curves.csv"
DIAGNOSTIC_CURVE = ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
INDEX_CACHE = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
OUTPUT_JSON = ROOT / "data" / "lgbm_csi300_hedge_v1.json"
OUTPUT_REPORT = ROOT / "reports" / "lgbm_csi300_hedge_v1.md"
OUTPUT_CURVES = ROOT / "reports" / "lgbm_csi300_hedge_v1_curves.csv"

TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
LOOKBACKS = (60, 120, 200)
THRESHOLDS = (0.98, 1.00, 1.02)
HEDGE_RATIOS = (0.25, 0.50, 0.75)
FUTURES_COST_BPS = 2.0
SIGNAL_TO_RETURN_LAG = 2


def fetch_csi300_ohlc() -> tuple[pd.DataFrame, dict]:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    try:
        with httpx.Client(timeout=30, trust_env=False, headers=headers) as client:
            rows = []
            for start, end in (
                ("2017-01-01", "2021-12-31"),
                ("2022-01-01", "2026-08-25"),
            ):
                params = {"param": f"sh000300,day,{start},{end},2000,qfq"}
                response = client.get(TENCENT_URL, params=params)
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", {})
                if not isinstance(data, dict):
                    raise RuntimeError(
                        f"CSI300 response error for {start} to {end}: "
                        f"{payload.get('msg', 'unknown')}"
                    )
                node = data.get("sh000300", {})
                part = node.get("qfqday") or node.get("day") or []
                rows.extend(part)
        if len(rows) < 1500:
            raise RuntimeError("CSI300 response has insufficient rows")
        frame = pd.DataFrame(rows).iloc[:, :6]
        frame.columns = ["date", "open", "close", "high", "low", "volume"]
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ("open", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame[["date", "open", "close"]]
            .dropna()
            .drop_duplicates("date")
            .sort_values("date")
        )
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(INDEX_CACHE, index=False, encoding="utf-8-sig", float_format="%.6f")
        source = {
            "mode": "live_tencent_api",
            "source": f"{TENCENT_URL}?param=sh000300,day,...",
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
    except Exception as exc:
        if not INDEX_CACHE.exists():
            raise
        frame = pd.read_csv(INDEX_CACHE, parse_dates=["date"])
        source = {
            "mode": "cached_fallback",
            "source": str(INDEX_CACHE.relative_to(ROOT)),
            "reason": str(exc),
        }
    return frame.set_index("date").sort_index(), source


def performance(returns: pd.Series) -> dict:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / max(len(clean), 1)) - 1.0)
    volatility = float(clean.std(ddof=0) * np.sqrt(252.0))
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": float(equity.div(equity.cummax()).sub(1.0).min()),
        "sharpe": float(clean.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "terminal_value": float(equity.iloc[-1]),
        "observations": int(len(clean)),
    }


def hedge_returns(
    strategy_returns: pd.Series,
    index: pd.DataFrame,
    *,
    lookback: int,
    threshold: float,
    hedge_ratio: float,
) -> tuple[pd.Series, pd.Series]:
    close = index["close"].reindex(strategy_returns.index).ffill()
    index_open_return = index["open"].pct_change().reindex(strategy_returns.index)
    trend = close.div(close.rolling(lookback, min_periods=lookback).mean())
    signal = trend.lt(threshold).astype(float)
    # A T-close signal is first tradable at T+1 open. The open-to-open PnL is
    # recognized on T+2, hence the two-index shift against curve pct_change().
    hedge_position = signal.shift(SIGNAL_TO_RETURN_LAG).fillna(0.0).mul(hedge_ratio)
    hedge_pnl = hedge_position.mul(index_open_return.fillna(0.0)).mul(-1.0)
    hedge_cost = hedge_position.diff().abs().fillna(hedge_position.abs()).mul(
        FUTURES_COST_BPS / 10_000.0
    )
    return strategy_returns.add(hedge_pnl).sub(hedge_cost), hedge_position


def objective(metrics: dict) -> float:
    return float(
        metrics["annual_return"]
        + 0.06 * metrics["sharpe"]
        + 0.08 * metrics["max_drawdown"]
    )


def load_curve(path: Path, column: str, start: str, end: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    values = pd.to_numeric(frame[column], errors="coerce").dropna().loc[start:end]
    if len(values) < 2:
        raise RuntimeError(f"insufficient curve data: {path.name}/{column}")
    return values


def main() -> None:
    index, source = fetch_csi300_ohlc()
    selection_curve = load_curve(
        SELECTION_CURVE, "current_lgbm", "2018-01-01", "2019-12-31"
    )
    selection_returns = selection_curve.pct_change().fillna(0.0)
    selection_rows = []
    for lookback in LOOKBACKS:
        for threshold in THRESHOLDS:
            for hedge_ratio in HEDGE_RATIOS:
                returns, position = hedge_returns(
                    selection_returns,
                    index,
                    lookback=lookback,
                    threshold=threshold,
                    hedge_ratio=hedge_ratio,
                )
                metrics = performance(returns)
                selection_rows.append(
                    {
                        "lookback": lookback,
                        "threshold": threshold,
                        "hedge_ratio": hedge_ratio,
                        "metrics": metrics,
                        "objective": objective(metrics),
                        "average_hedge": float(position.mean()),
                    }
                )
    winner = max(selection_rows, key=lambda row: row["objective"])

    current_curve = load_curve(
        DIAGNOSTIC_CURVE, "previous_lgbm", "2020-01-02", "2026-08-25"
    )
    current_returns = current_curve.pct_change().fillna(0.0)
    hedged_returns, hedge_position = hedge_returns(
        current_returns,
        index,
        lookback=winner["lookback"],
        threshold=winner["threshold"],
        hedge_ratio=winner["hedge_ratio"],
    )
    hedged_curve = (1.0 + hedged_returns).cumprod()
    hedged_curve.iloc[0] = 1.0
    current_curve = current_curve.div(float(current_curve.iloc[0]))
    curves = pd.DataFrame(
        {
            "current_lgbm": current_curve,
            "lgbm_csi300_hedged": hedged_curve,
            "csi300_hedge_ratio": hedge_position,
        }
    ).dropna()
    curves["current_drawdown"] = curves["current_lgbm"].div(
        curves["current_lgbm"].cummax()
    ).sub(1.0)
    curves["hedged_drawdown"] = curves["lgbm_csi300_hedged"].div(
        curves["lgbm_csi300_hedged"].cummax()
    ).sub(1.0)
    curves.index.name = "date"
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")

    windows = {
        "early_2020_2023": ("2020-01-02", "2023-12-31"),
        "late_2024_2026": ("2024-01-01", "2026-08-25"),
        "full_2020_2026": ("2020-01-02", "2026-08-25"),
    }
    diagnostics = {}
    for name, (start, end) in windows.items():
        current = curves.loc[start:end, "current_lgbm"].pct_change().fillna(0.0)
        hedged = curves.loc[start:end, "lgbm_csi300_hedged"].pct_change().fillna(0.0)
        diagnostics[name] = {
            "current_lgbm": performance(current),
            "lgbm_csi300_hedged": performance(hedged),
            "average_hedge": float(curves.loc[start:end, "csi300_hedge_ratio"].mean()),
        }

    early = diagnostics["early_2020_2023"]
    late = diagnostics["late_2024_2026"]
    full = diagnostics["full_2020_2026"]
    checks = {
        "early_cagr_improves_1pp": (
            early["lgbm_csi300_hedged"]["annual_return"]
            >= early["current_lgbm"]["annual_return"] + 0.01
        ),
        "late_cagr_retains_90pct": (
            late["lgbm_csi300_hedged"]["annual_return"]
            >= late["current_lgbm"]["annual_return"] * 0.90
        ),
        "full_drawdown_not_worse": (
            full["lgbm_csi300_hedged"]["max_drawdown"]
            >= full["current_lgbm"]["max_drawdown"]
        ),
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_candidate_only" if all(checks.values()) else "rejected",
        "strategy": "current 75/25 LGBM plus causal CSI300 futures hedge",
        "selection_window": ["2018-01-01", "2019-12-31"],
        "selected_parameters": {
            key: winner[key] for key in ("lookback", "threshold", "hedge_ratio")
        },
        "selected_selection_metrics": winner["metrics"],
        "selection_candidates": sorted(
            selection_rows, key=lambda row: row["objective"], reverse=True
        ),
        "diagnostics": diagnostics,
        "promotion_gate": {"passed": all(checks.values()), "checks": checks},
        "execution_audit": {
            "signal": "CSI300 close / trailing close MA, using data through signal close",
            "fill": "next open",
            "pnl": "open-to-next-open",
            "signal_to_return_lag": SIGNAL_TO_RETURN_LAG,
            "futures_cost_bps": FUTURES_COST_BPS,
        },
        "source": source,
        "production_change": False,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
            "curves": str(OUTPUT_CURVES.relative_to(ROOT)),
        },
        "limitations": [
            "This overlay requires CSI300 index futures or equivalent short exposure and is not a pure long-only stock strategy.",
            "The hedge uses index open-to-open returns as a futures proxy and does not model basis, margin, roll, or financing.",
            "The current Top50 universe is backfilled and has survivorship bias.",
            "2020-2026 has been repeatedly revealed and is diagnostic-only.",
            "No production or paper-trading configuration was changed.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# LGBM CSI300 hedge overlay v1",
        "",
        f"- Status: {payload['status']}; production unchanged.",
        f"- Selected on 2018-2019 only: MA{winner['lookback']}, threshold {winner['threshold']:.2f}, max hedge {winner['hedge_ratio']:.0%}.",
        f"- Signal-close to next-open execution; futures proxy cost {FUTURES_COST_BPS:.1f} bps per position change.",
        "",
        "| Window | Strategy | CAGR | Max drawdown | Sharpe | Average hedge |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, label in (("early_2020_2023", "2020-2023"), ("late_2024_2026", "2024-2026"), ("full_2020_2026", "2020-2026")):
        block = diagnostics[name]
        for key, strategy_label in (("current_lgbm", "Current LGBM"), ("lgbm_csi300_hedged", "Hedged LGBM")):
            metrics = block[key]
            lines.append(
                f"| {label} | {strategy_label} | {metrics['annual_return']:.2%} | "
                f"{metrics['max_drawdown']:.2%} | {metrics['sharpe']:.3f} | "
                f"{block['average_hedge']:.1%} |"
            )
    lines.extend(["", "## Gate", ""])
    lines.extend(
        f"- {name}: {'pass' if passed else 'fail'}" for name, passed in checks.items()
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
