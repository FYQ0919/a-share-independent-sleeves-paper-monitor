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


INDEX_PATH = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
STRATEGY_PATH = ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
OUTPUT_JSON = ROOT / "data" / "lgbm_csi300_weak_hedge_v2.json"
OUTPUT_REPORT = ROOT / "reports" / "lgbm_csi300_weak_hedge_v2.md"
OUTPUT_CURVES = ROOT / "reports" / "lgbm_csi300_weak_hedge_v2_curves.csv"

LOOKBACK = 120
THRESHOLD = 1.0
OLD_RATIO = 0.75
SELECTED_RATIO = 0.50
RATIOS = (0.0, 0.25, 0.40, 0.50, 0.60, 0.75)
WINDOWS = {
    "early_2020_2023": ("2020-01-02", "2023-12-31"),
    "late_2024_2026": ("2024-01-01", "2026-08-25"),
    "full_2020_2026": ("2020-01-02", "2026-08-25"),
}


def evaluate_ratios(strategy_returns: pd.Series, index: pd.DataFrame) -> list[dict]:
    rows = []
    for ratio in RATIOS:
        managed, position = hedge_returns(
            strategy_returns,
            index,
            lookback=LOOKBACK,
            threshold=THRESHOLD,
            hedge_ratio=ratio,
        )
        windows = {
            name: performance(managed.loc[start:end])
            for name, (start, end) in WINDOWS.items()
        }
        rows.append(
            {
                "hedge_ratio": ratio,
                "average_hedge_ratio": float(position.mean()),
                "windows": windows,
            }
        )
    return rows


def main() -> None:
    index = pd.read_csv(INDEX_PATH, parse_dates=["date"]).set_index("date").sort_index()
    curve = (
        pd.read_csv(STRATEGY_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()["previous_lgbm"]
        .loc["2020-01-02":"2026-08-25"]
    )
    strategy_returns = curve.pct_change().fillna(0.0)
    candidates = evaluate_ratios(strategy_returns, index)
    by_ratio = {row["hedge_ratio"]: row for row in candidates}
    selected = by_ratio[SELECTED_RATIO]
    old = by_ratio[OLD_RATIO]

    weak_returns, weak_position = hedge_returns(
        strategy_returns,
        index,
        lookback=LOOKBACK,
        threshold=THRESHOLD,
        hedge_ratio=SELECTED_RATIO,
    )
    old_returns, old_position = hedge_returns(
        strategy_returns,
        index,
        lookback=LOOKBACK,
        threshold=THRESHOLD,
        hedge_ratio=OLD_RATIO,
    )
    curves = pd.DataFrame(
        {
            "base_lgbm": (1.0 + strategy_returns).cumprod(),
            "weak_50_hedge": (1.0 + weak_returns).cumprod(),
            "old_75_hedge": (1.0 + old_returns).cumprod(),
            "weak_50_position": weak_position,
            "old_75_position": old_position,
        }
    )
    curves.index.name = "date"
    curves.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_diagnostic_preference_override",
        "strategy": "75/25 LGBM plus weakened causal CSI300 hedge",
        "frozen_parameters": {
            "lookback": LOOKBACK,
            "threshold": THRESHOLD,
            "maximum_hedge_ratio": SELECTED_RATIO,
            "previous_maximum_hedge_ratio": OLD_RATIO,
            "signal_at": "close",
            "execution_at": "next_open",
            "pnl_window": "open_to_next_open",
            "change_cost_bps": 2.0,
        },
        "selection_note": (
            "50% is a deployment preference chosen after inspecting the revealed "
            "2020-2026 diagnostics. It is not a fresh out-of-sample selection."
        ),
        "candidate_diagnostics": candidates,
        "selected_diagnostics": selected,
        "previous_diagnostics": old,
        "selected_minus_previous": {
            name: {
                "annual_return": (
                    selected["windows"][name]["annual_return"]
                    - old["windows"][name]["annual_return"]
                ),
                "sharpe": (
                    selected["windows"][name]["sharpe"]
                    - old["windows"][name]["sharpe"]
                ),
                "max_drawdown": (
                    selected["windows"][name]["max_drawdown"]
                    - old["windows"][name]["max_drawdown"]
                ),
            }
            for name in WINDOWS
        },
        "production_change": False,
        "paper_change": True,
        "limitations": [
            "The 2020-2026 period is repeatedly revealed and is not an untouched holdout.",
            "The CSI300 open-to-open return is a futures proxy without basis, margin, roll or financing.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "A fresh forward-paper window is required to confirm the annual-return improvement.",
        ],
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
            "curves": str(OUTPUT_CURVES.relative_to(ROOT)),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Weakened CSI300 hedge v2",
        "",
        "- Frozen maximum hedge: 50% (previously 75%).",
        "- MA120 and threshold 1.00 are unchanged; signal and execution remain causal.",
        "- Selection status: revealed diagnostic preference override, not fresh OOS.",
        "",
        "| Window | 50% CAGR | 75% CAGR | CAGR change | 50% Sharpe | 75% Sharpe | 50% MDD | 75% MDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "early_2020_2023": "2020-2023",
        "late_2024_2026": "2024-2026",
        "full_2020_2026": "2020-2026",
    }
    for name, label in labels.items():
        new = selected["windows"][name]
        previous = old["windows"][name]
        lines.append(
            f"| {label} | {new['annual_return']:.2%} | {previous['annual_return']:.2%} | "
            f"{new['annual_return'] - previous['annual_return']:+.2%} | "
            f"{new['sharpe']:.3f} | {previous['sharpe']:.3f} | "
            f"{new['max_drawdown']:.2%} | {previous['max_drawdown']:.2%} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
