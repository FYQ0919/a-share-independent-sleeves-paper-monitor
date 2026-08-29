from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CURVES = ROOT / "reports" / "drl_adaptive_allocator_v1_curves.csv"
OUTPUT_HTML = ROOT / "reports" / "drl_adaptive_allocator_v1_chart.html"
OUTPUT_PNG = ROOT / "reports" / "drl_adaptive_allocator_v1_chart.png"
START = "2022-12-30"

WIDTH = 1400
HEIGHT = 1100
LEFT = 105
RIGHT = 55

SERIES = {
    "drl_adaptive": ("DRL adaptive", "#167a72", 4.0),
    "fixed_75_25_same_contract": ("Fixed 75/25, same execution", "#d97706", 3.0),
    "previous_75_25_independent_sleeves": (
        "Previous 75/25 independent sleeves",
        "#3568b8",
        2.5,
    ),
    "formal_lgbm": ("Formal LGBM", "#7a4fa3", 2.5),
}


def performance(curve: pd.Series) -> dict:
    returns = curve.pct_change(fill_method=None).dropna()
    total = float(curve.iloc[-1] - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    return {
        "terminal": float(curve.iloc[-1]),
        "cagr": float((1.0 + total) ** (252.0 / len(returns)) - 1.0),
        "sharpe": float(returns.mean() * 252.0 / volatility),
        "drawdown": float(curve.div(curve.cummax()).sub(1.0).min()),
    }


def scale(value: float, source_min: float, source_max: float, low: float, high: float) -> float:
    if source_max <= source_min:
        return (low + high) / 2.0
    ratio = (value - source_min) / (source_max - source_min)
    return low + ratio * (high - low)


def path_points(
    dates: pd.DatetimeIndex,
    values: pd.Series,
    y_min: float,
    y_max: float,
    top: float,
    bottom: float,
) -> str:
    start_ns = float(dates[0].value)
    end_ns = float(dates[-1].value)
    points = []
    for date, value in zip(dates, values):
        x = scale(float(date.value), start_ns, end_ns, LEFT, WIDTH - RIGHT)
        y = scale(float(value), y_min, y_max, bottom, top)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def axes(
    dates: pd.DatetimeIndex,
    y_ticks: list[float],
    y_min: float,
    y_max: float,
    top: float,
    bottom: float,
    formatter,
) -> str:
    start_ns = float(dates[0].value)
    end_ns = float(dates[-1].value)
    pieces = [
        f'<rect x="{LEFT}" y="{top}" width="{WIDTH - LEFT - RIGHT}" '
        f'height="{bottom - top}" fill="#ffffff" stroke="#c7cbd1"/>',
    ]
    for tick in y_ticks:
        y = scale(tick, y_min, y_max, bottom, top)
        pieces.append(
            f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.2f}" y2="{y:.2f}" '
            'stroke="#e2e5e9" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{LEFT - 15}" y="{y + 5:.2f}" text-anchor="end" '
            f'class="axis">{escape(formatter(tick))}</text>'
        )
    for year in range(dates[0].year, dates[-1].year + 1):
        date = pd.Timestamp(f"{year}-01-01")
        if date < dates[0] or date > dates[-1]:
            continue
        x = scale(float(date.value), start_ns, end_ns, LEFT, WIDTH - RIGHT)
        pieces.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{bottom}" '
            'stroke="#eceef1" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{x:.2f}" y="{bottom + 27}" text-anchor="middle" '
            f'class="axis">{year}</text>'
        )
    return "".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CURVES)
    parser.add_argument("--output", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--title", default="DRL adaptive allocation vs fixed baselines")
    parser.add_argument(
        "--subtitle",
        default="Walk-forward adaptive period: 2023-01-01 to 2026-08-25 | net of allocation and hedge-change costs",
    )
    args = parser.parse_args()
    frame = (
        pd.read_csv(args.input, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[START:]
    )
    curves = frame[list(SERIES)].div(frame[list(SERIES)].iloc[0])
    drawdowns = curves.div(curves.cummax()).sub(1.0)
    metrics = {name: performance(curves[name]) for name in SERIES}
    sampled = curves.iloc[::2]
    sampled_drawdowns = drawdowns.reindex(sampled.index)

    net_top, net_bottom = 205.0, 575.0
    dd_top, dd_bottom = 650.0, 855.0
    weight_top, weight_bottom = 925.0, 1050.0
    net_min = min(0.95, float(sampled.min().min()) * 0.98)
    net_max = float(sampled.max().max()) * 1.04
    net_ticks = np.linspace(net_min, net_max, 6).tolist()
    dd_min = min(-0.05, float(sampled_drawdowns.min().min()) * 1.08)
    dd_ticks = np.linspace(dd_min, 0.0, 5).tolist()

    metric_rows = []
    for name, (label, color, line_width) in SERIES.items():
        metric = metrics[name]
        metric_rows.append(
            f'<tr><td><span class="swatch" style="background:{color}"></span>{escape(label)}</td>'
            f'<td>{metric["terminal"]:.2f}x</td><td>{metric["cagr"]:.2%}</td>'
            f'<td>{metric["sharpe"]:.3f}</td><td>{metric["drawdown"]:.2%}</td></tr>'
        )

    net_lines = []
    dd_lines = []
    for name, (_, color, line_width) in SERIES.items():
        net_lines.append(
            f'<polyline points="{path_points(sampled.index, sampled[name], net_min, net_max, net_top, net_bottom)}" '
            f'fill="none" stroke="{color}" stroke-width="{line_width}" stroke-linejoin="round"/>'
        )
        dd_lines.append(
            f'<polyline points="{path_points(sampled.index, sampled_drawdowns[name], dd_min, 0.0, dd_top, dd_bottom)}" '
            f'fill="none" stroke="{color}" stroke-width="{line_width}" stroke-linejoin="round"/>'
        )

    weight_dates = frame.index[::2]
    trend_points = path_points(
        weight_dates,
        frame.loc[weight_dates, "trend_weight"],
        0.0,
        0.90,
        weight_top,
        weight_bottom,
    )
    hedge_points = path_points(
        weight_dates,
        frame.loc[weight_dates, "hedge_ratio"],
        0.0,
        0.90,
        weight_top,
        weight_bottom,
    )
    weight_axes = axes(
        weight_dates,
        [0.0, 0.3, 0.6, 0.9],
        0.0,
        0.90,
        weight_top,
        weight_bottom,
        lambda value: f"{value:.0%}",
    )

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(args.title)}</title>
<style>
body {{ margin:0; background:#ffffff; color:#20242a; font-family:"Segoe UI",Arial,sans-serif; }}
.page {{ width:{WIDTH}px; height:{HEIGHT}px; box-sizing:border-box; padding:25px 42px; }}
h1 {{ margin:0; font-size:28px; font-weight:600; letter-spacing:0; }}
.subtitle {{ margin-top:7px; color:#60666f; font-size:16px; }}
table {{ position:absolute; top:78px; right:48px; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:3px 9px; text-align:right; border-bottom:1px solid #e0e3e7; }}
th:first-child,td:first-child {{ text-align:left; }}
.swatch {{ display:inline-block; width:12px; height:3px; margin:0 7px 4px 0; }}
svg {{ position:absolute; left:0; top:0; width:{WIDTH}px; height:{HEIGHT}px; }}
.axis {{ fill:#555d67; font-size:13px; }}
.legend {{ fill:#30353b; font-size:14px; }}
.panel-title {{ fill:#20242a; font-size:16px; font-weight:600; }}
.axis-title {{ fill:#555d67; font-size:13px; }}
.note {{ position:absolute; left:105px; bottom:9px; color:#666d76; font-size:13px; }}
</style>
</head>
<body><div class="page">
<h1>{escape(args.title)}</h1>
<div class="subtitle">{escape(args.subtitle)}</div>
<table><thead><tr><th>Strategy</th><th>End</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table>
<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="DRL adaptive allocation performance, drawdown and weights">
<text x="{LEFT}" y="195" class="panel-title">Normalized net value</text>
{axes(sampled.index, net_ticks, net_min, net_max, net_top, net_bottom, lambda value: f'{value:.1f}x')}
{''.join(net_lines)}
<text x="22" y="390" class="axis-title" transform="rotate(-90 22 390)">Net value (x)</text>
<text x="{LEFT}" y="635" class="panel-title">Drawdown</text>
{axes(sampled.index, dd_ticks, dd_min, 0.0, dd_top, dd_bottom, lambda value: f'{value:.0%}')}
{''.join(dd_lines)}
<text x="22" y="760" class="axis-title" transform="rotate(-90 22 760)">Drawdown (%)</text>
<text x="{LEFT}" y="910" class="panel-title">Executed allocation: trend weight and maximum CSI300 hedge ratio</text>
{weight_axes}
<polyline points="{trend_points}" fill="none" stroke="#167a72" stroke-width="3"/>
<polyline points="{hedge_points}" fill="none" stroke="#c2413b" stroke-width="2.5"/>
<line x1="1030" x2="1063" y1="910" y2="910" stroke="#167a72" stroke-width="3"/><text x="1070" y="915" class="legend">Trend weight</text>
<line x1="1180" x2="1213" y1="910" y2="910" stroke="#c2413b" stroke-width="2.5"/><text x="1220" y="915" class="legend">Max hedge</text>
</svg>
<div class="note">Historical diagnostic only. The period was repeatedly inspected; no production configuration was changed.</div>
</div></body></html>'''
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
