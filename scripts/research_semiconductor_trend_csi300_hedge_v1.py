from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_lgbm_csi300_hedge_v1 import hedge_returns, performance


TREND_CURVE_PATH = ROOT / "reports" / "lgbm_semiconductor_trend_v1_curves.csv"
INDEX_PATH = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
OUTPUT_JSON = ROOT / "data" / "semiconductor_trend_csi300_hedge_v1.json"
OUTPUT_REPORT = ROOT / "reports" / "semiconductor_trend_csi300_hedge_v1.md"
OUTPUT_CURVES = ROOT / "reports" / "semiconductor_trend_csi300_hedge_v1_curves.csv"

LOOKBACK = 120
THRESHOLD = 1.0
RATIOS = (0.0, 0.25, 0.40, 0.50, 0.60, 0.75)
WINDOWS = {
    "2020_2023": ("2020-01-02", "2023-12-31"),
    "2024_2025": ("2024-01-01", "2025-12-31"),
    "2026_ytd": ("2026-01-01", "2026-08-25"),
    "2026_june": ("2026-06-01", "2026-07-01"),
    "full": ("2020-01-02", "2026-08-25"),
}


def evaluate(
    strategy_returns: pd.Series,
    index: pd.DataFrame,
    ratio: float,
) -> tuple[pd.Series, pd.Series, dict]:
    managed, position = hedge_returns(
        strategy_returns,
        index,
        lookback=LOOKBACK,
        threshold=THRESHOLD,
        hedge_ratio=ratio,
    )
    windows = {}
    for name, (start, end) in WINDOWS.items():
        metrics = performance(managed.loc[start:end])
        windows[name] = {
            **metrics,
            "average_actual_hedge": float(position.loc[start:end].mean()),
            "position_changes": int(position.loc[start:end].diff().abs().gt(0).sum()),
        }
    return managed, position, windows


def main() -> None:
    strategy_curves = (
        pd.read_csv(TREND_CURVE_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc["2020-01-02":"2026-08-25"]
    )
    index = (
        pd.read_csv(INDEX_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    if strategy_curves.index[-1] != pd.Timestamp("2026-08-25"):
        raise RuntimeError("Trend curve endpoint changed")
    if index.index[-1] < strategy_curves.index[-1]:
        raise RuntimeError("CSI300 index cache is stale relative to the trend curve")

    trend_returns = strategy_curves["trend_expert_nocap"].pct_change().fillna(0.0)
    baseline_returns = strategy_curves["current_lgbm"].pct_change().fillna(0.0)

    curves = pd.DataFrame(index=strategy_curves.index)
    curves["trend_expert"] = (1.0 + trend_returns).cumprod()
    curves["current_lgbm"] = (1.0 + baseline_returns).cumprod()
    rows = []
    trend_managed: dict[float, pd.Series] = {}
    trend_positions: dict[float, pd.Series] = {}
    for ratio in RATIOS:
        managed, position, windows = evaluate(trend_returns, index, ratio)
        trend_managed[ratio] = managed
        trend_positions[ratio] = position
        curves[f"trend_hedge_{int(ratio * 100):02d}"] = (1.0 + managed).cumprod()
        rows.append({
            "maximum_hedge_ratio": ratio,
            "windows": windows,
        })

    baseline_50, baseline_position, baseline_windows = evaluate(
        baseline_returns, index, 0.50
    )
    curves["current_lgbm_hedge_50"] = (1.0 + baseline_50).cumprod()
    curves["hedge_position_50"] = baseline_position
    curves.index.name = "date"
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")

    by_ratio = {row["maximum_hedge_ratio"]: row for row in rows}
    unhedged = by_ratio[0.0]["windows"]
    hedge_50 = by_ratio[0.50]["windows"]
    changes = {}
    for window in WINDOWS:
        changes[window] = {
            "annual_return_pp": (
                hedge_50[window]["annual_return"] - unhedged[window]["annual_return"]
            ),
            "sharpe": hedge_50[window]["sharpe"] - unhedged[window]["sharpe"],
            "max_drawdown_pp": (
                hedge_50[window]["max_drawdown"] - unhedged[window]["max_drawdown"]
            ),
        }

    full_unhedged = unhedged["full"]
    full_50 = hedge_50["full"]
    gate = {
        "max_drawdown_improves_at_least_3pp": (
            full_50["max_drawdown"] >= full_unhedged["max_drawdown"] + 0.03
        ),
        "annual_return_retains_95pct": (
            full_50["annual_return"] >= full_unhedged["annual_return"] * 0.95
        ),
        "sharpe_not_lower": full_50["sharpe"] >= full_unhedged["sharpe"],
        "early_drawdown_not_worse": (
            hedge_50["2020_2023"]["max_drawdown"]
            >= unhedged["2020_2023"]["max_drawdown"]
        ),
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_diagnostic_only_not_deployed",
        "strategy": "semiconductor trend expert plus causal CSI300 futures hedge",
        "parameters": {
            "lookback": LOOKBACK,
            "threshold": THRESHOLD,
            "evaluated_maximum_hedge_ratios": list(RATIOS),
            "reference_ratio": 0.50,
            "signal_at": "close",
            "execution_at": "next_open",
            "pnl_window": "open_to_next_open",
            "position_change_cost_bps": 2.0,
        },
        "ratio_diagnostics": rows,
        "reference_50pct_changes": changes,
        "current_lgbm_50pct_reference": baseline_windows,
        "gate": {"passed": all(gate.values()), "checks": gate},
        "production_change": False,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
            "curves": str(OUTPUT_CURVES.relative_to(ROOT)),
        },
        "limitations": [
            "The CSI300 open-to-open return is a futures proxy without basis, margin, roll, financing or collateral yield.",
            "The trend strategy uses a frozen current Top50 backfilled through history and has constituent and survivorship bias.",
            "The 2020-2026 interval is repeatedly revealed and cannot select a production hedge ratio.",
            "The trend hypothesis was prompted by the already revealed June 2026 equipment rally.",
            "No production or paper-trading configuration was changed.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Semiconductor trend expert plus CSI300 hedge v1",
        "",
        "- Status: historical diagnostic only; production unchanged.",
        "- Hedge rule: CSI300 close below MA120, next-open execution, 2bp per position change.",
        "",
        "| Window | Max hedge | CAGR | Sharpe | Max drawdown | Average actual hedge |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        for row in rows:
            ratio = row["maximum_hedge_ratio"]
            metrics = row["windows"][window]
            lines.append(
                f"| {window} | {ratio:.0%} | {metrics['annual_return']:.2%} | "
                f"{metrics['sharpe']:.3f} | {metrics['max_drawdown']:.2%} | "
                f"{metrics['average_actual_hedge']:.1%} |"
            )
    lines.extend([
        "",
        "## Reference 50% overlay versus unhedged",
        "",
        "| Window | CAGR change | Sharpe change | Drawdown improvement |",
        "|---|---:|---:|---:|",
    ])
    for window, values in changes.items():
        lines.append(
            f"| {window} | {values['annual_return_pp']:+.2%} | "
            f"{values['sharpe']:+.3f} | {values['max_drawdown_pp']:+.2%} |"
        )
    lines.extend([
        "",
        f"Gate: {'PASS' if payload['gate']['passed'] else 'FAIL'}; historical result cannot authorize deployment.",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_JSON),
        "report": str(OUTPUT_REPORT),
        "curves": str(OUTPUT_CURVES),
        "full_ratio_diagnostics": [
            {
                "ratio": row["maximum_hedge_ratio"],
                **row["windows"]["full"],
            }
            for row in rows
        ],
        "reference_50pct_changes": changes,
        "gate": payload["gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
