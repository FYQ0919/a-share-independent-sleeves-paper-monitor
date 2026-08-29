from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_strategy_ndx_image import Canvas
from scripts.research_lgbm_csi300_hedge_v1 import performance


CURVE_PATH = ROOT / "reports" / "lgbm_csi300_weak_hedge_v2_curves.csv"
INDEX_PATH = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
OUTPUT_CSV = ROOT / "reports" / "lgbm_sharpe_2025_2026.csv"
OUTPUT_JSON = ROOT / "data" / "lgbm_sharpe_2025_2026.json"
OUTPUT_REPORT = ROOT / "reports" / "lgbm_sharpe_2025_2026.md"
OUTPUT_IMAGE = ROOT / "reports" / "lgbm_sharpe_2025_2026.png"

REQUESTED_START = pd.Timestamp("2025-01-01")
WIDTH = 1400
HEIGHT = 900
MARGIN = {"left": 112, "right": 250, "top": 190, "bottom": 160}


LABELS = {
    "lgbm_csi300_hedged50": "LGBM + CSI300 dynamic hedge (max 50%)",
    "base_lgbm": "Base LGBM stock sleeve",
    "csi300": "CSI300 price index",
}


def load_curves() -> pd.DataFrame:
    strategy = (
        pd.read_csv(CURVE_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    index = (
        pd.read_csv(INDEX_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    common_dates = strategy.index.intersection(index.index)
    common_dates = common_dates[common_dates >= REQUESTED_START]
    frame = pd.DataFrame(
        {
            "lgbm_csi300_hedged50": strategy["weak_50_hedge"].reindex(common_dates),
            "base_lgbm": strategy["base_lgbm"].reindex(common_dates),
            "csi300": pd.to_numeric(index["close"], errors="coerce").reindex(common_dates),
            "hedge_ratio": strategy["weak_50_position"].reindex(common_dates),
        },
        index=common_dates,
    ).dropna()
    if len(frame) < 2:
        raise RuntimeError("2025-2026 has insufficient common strategy/index data")
    for column in LABELS:
        frame[column] = frame[column].div(float(frame[column].iloc[0]))
    frame.index.name = "date"
    return frame


def calculate_metrics(frame: pd.DataFrame) -> tuple[dict, dict]:
    full = {
        column: {"label": label, **performance(frame[column].pct_change().fillna(0.0))}
        for column, label in LABELS.items()
    }
    calendar = {}
    for year in sorted(frame.index.year.unique()):
        subset = frame.loc[str(year)]
        calendar[str(year)] = {
            column: performance(subset[column].pct_change().fillna(0.0))
            for column in LABELS
        }
    return full, calendar


def _nice_bounds(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 0.1)
    step = max(0.1, math.ceil((span / 5.0) * 10.0) / 10.0)
    return max(0.0, math.floor((low - step * 0.35) / step) * step), math.ceil(
        (high + step * 0.35) / step
    ) * step


def render_chart(frame: pd.DataFrame, full: dict, calendar: dict) -> None:
    canvas = Canvas(WIDTH, HEIGHT, (247, 248, 246))
    ink = (31, 40, 36)
    muted = (99, 111, 105)
    grid = (214, 221, 217)
    green = (16, 122, 86)
    blue = (47, 96, 158)
    orange = (211, 101, 46)
    left, right = MARGIN["left"], WIDTH - MARGIN["right"]
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    start, end = frame.index[0], frame.index[-1]
    y_min, y_max = _nice_bounds(frame[list(LABELS)].stack())

    def x(value: pd.Timestamp) -> float:
        return left + (value - start).days / max((end - start).days, 1) * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    canvas.text(left, 38, "LGBM SHARPE AND GROWTH | 2025-2026", ink, 4)
    canvas.text(
        left,
        87,
        f"{start:%Y-%m-%d} TO {end:%Y-%m-%d} | START NAV 1.00 | {len(frame)} TRADING DAYS",
        muted,
        2,
    )
    legend = [
        ("lgbm_csi300_hedged50", "HEDGED LGBM", green, left),
        ("base_lgbm", "BASE LGBM", blue, left + 420),
        ("csi300", "CSI300", orange, left + 780),
    ]
    for column, label, color, offset in legend:
        canvas.line(offset, 130, offset + 42, 130, color, 4)
        canvas.text(offset + 54, 123, f"{label} SHARPE {full[column]['sharpe']:.2f}", ink, 2)

    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        py = y(value)
        canvas.line(left, py, right, py, grid)
        canvas.text(28, round(py) - 7, f"{value:.2f}X", muted, 2)
    if y_min <= 1.0 <= y_max:
        canvas.dashed_line(left, y(1.0), right, y(1.0), muted)

    ticks = pd.date_range(start.normalize(), end.normalize(), freq="QS")
    if start not in ticks:
        ticks = ticks.insert(0, start)
    for date in ticks:
        if date > end:
            continue
        px = round(x(date))
        canvas.line(px, bottom, px, bottom + 7, grid)
        canvas.text(max(left, px - 36), bottom + 20, f"{date:%y-%m}", muted, 2)

    series = [
        ("lgbm_csi300_hedged50", green, 4),
        ("base_lgbm", blue, 3),
        ("csi300", orange, 3),
    ]
    for column, color, width in series:
        points = [(x(date), y(value)) for date, value in frame[column].items()]
        for first, second in zip(points, points[1:]):
            canvas.line(*first, *second, color, width)

    endpoints = []
    for column, color, _ in series:
        value = float(frame[column].iloc[-1])
        canvas.circle(round(x(end)), round(y(value)), 6, color)
        endpoints.append([y(value), column, color, value])
    endpoints.sort(key=lambda item: item[0])
    for index in range(1, len(endpoints)):
        endpoints[index][0] = max(endpoints[index][0], endpoints[index - 1][0] + 58)
    overflow = endpoints[-1][0] - (bottom - 34)
    if overflow > 0:
        for item in endpoints:
            item[0] -= overflow
    short_labels = {
        "lgbm_csi300_hedged50": "HEDGED",
        "base_lgbm": "BASE",
        "csi300": "CSI300",
    }
    for label_y, column, color, value in endpoints:
        canvas.text(right + 18, round(label_y) - 18, f"{short_labels[column]} {value:.2f}X", color, 2)
        canvas.text(
            right + 18,
            round(label_y) + 4,
            f"RETURN {full[column]['total_return']:+.1%}",
            muted,
            2,
        )

    canvas.text(
        left,
        HEIGHT - 108,
        f"FULL SHARPE | HEDGED {full['lgbm_csi300_hedged50']['sharpe']:.3f} | BASE {full['base_lgbm']['sharpe']:.3f} | CSI300 {full['csi300']['sharpe']:.3f}",
        ink,
        2,
    )
    canvas.text(
        left,
        HEIGHT - 80,
        f"2025 SHARPE | HEDGED {calendar['2025']['lgbm_csi300_hedged50']['sharpe']:.3f} | BASE {calendar['2025']['base_lgbm']['sharpe']:.3f} | CSI300 {calendar['2025']['csi300']['sharpe']:.3f}",
        ink,
        2,
    )
    canvas.text(
        left,
        HEIGHT - 52,
        f"2026 YTD SHARPE | HEDGED {calendar['2026']['lgbm_csi300_hedged50']['sharpe']:.3f} | BASE {calendar['2026']['base_lgbm']['sharpe']:.3f} | CSI300 {calendar['2026']['csi300']['sharpe']:.3f}",
        ink,
        2,
    )
    canvas.text(
        left,
        HEIGHT - 24,
        "DAILY RETURN | ZERO RISK-FREE RATE | SQRT 252 ANNUALIZATION | CSI300 PRICE INDEX",
        muted,
        2,
    )
    canvas.save(OUTPUT_IMAGE)


def main() -> None:
    frame = load_curves()
    full, calendar = calculate_metrics(frame)
    frame.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")
    render_chart(frame, full, calendar)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic",
        "window": [frame.index[0].date().isoformat(), frame.index[-1].date().isoformat()],
        "observations": len(frame),
        "sharpe_definition": {
            "frequency": "daily",
            "risk_free_rate": 0.0,
            "annualization": "sqrt(252)",
            "standard_deviation": "population (ddof=0)",
        },
        "full_period": full,
        "calendar_periods": calendar,
        "average_actual_hedge_ratio": float(frame["hedge_ratio"].mean()),
        "strategy_contract": {
            "stock": "frozen 75/25 Alpha158-Barra LGBM, Top5, fixed 10-session rebalance, T+1 open, 12 bps one-way",
            "hedge": "CSI300 MA120 threshold 1.0, max 50%, next-open, 2 bps position-change cost",
        },
        "limitations": [
            "The period is repeatedly revealed historical data, not a fresh out-of-sample test.",
            "The Top50 universe is backfilled and has constituent and survivorship bias.",
            "The CSI300 benchmark is a price index and excludes dividends.",
            "The index hedge is a futures proxy without basis, margin, roll, or financing.",
        ],
        "artifacts": {
            "image": str(OUTPUT_IMAGE.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# LGBM Sharpe analysis, 2025-2026",
        "",
        f"- Window: {frame.index[0]:%Y-%m-%d} to {frame.index[-1]:%Y-%m-%d} ({len(frame)} common trading days).",
        "- Sharpe: daily mean / daily population standard deviation * sqrt(252), zero risk-free rate.",
        "",
        "| Period | Series | Return | CAGR | Sharpe | Max drawdown |",
        "|---|---|---:|---:|---:|---:|",
    ]
    blocks = [("Full", full)] + [(year, calendar[year]) for year in sorted(calendar)]
    for period, metrics in blocks:
        for column in LABELS:
            item = metrics[column]
            lines.append(
                f"| {period} | {LABELS[column]} | {item['total_return']:.2%} | "
                f"{item['annual_return']:.2%} | {item['sharpe']:.3f} | "
                f"{item['max_drawdown']:.2%} |"
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
