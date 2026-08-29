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
from scripts.research_lgbm_csi300_hedge_v1 import hedge_returns, performance


STRATEGY_PATH = ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
INDEX_PATH = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
OUTPUT_CSV = ROOT / "reports" / "lgbm_weak_hedge_202606_now.csv"
OUTPUT_JSON = ROOT / "data" / "lgbm_weak_hedge_202606_now.json"
OUTPUT_REPORT = ROOT / "reports" / "lgbm_weak_hedge_202606_now.md"
OUTPUT_IMAGE = ROOT / "reports" / "lgbm_weak_hedge_202606_now.png"

START_DATE = pd.Timestamp("2026-06-01")
LOOKBACK = 120
THRESHOLD = 1.0
MAX_HEDGE_RATIO = 0.50
WIDTH = 1400
HEIGHT = 900
MARGIN = {"left": 112, "right": 250, "top": 190, "bottom": 155}


def load_backtest() -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.Timestamp]:
    strategy = (
        pd.read_csv(STRATEGY_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()["previous_lgbm"]
    )
    strategy = pd.to_numeric(strategy, errors="coerce").dropna()
    index = (
        pd.read_csv(INDEX_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    index[["open", "close"]] = index[["open", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    index = index.dropna(subset=["open", "close"])

    # Build the causal hedge on the complete history so MA120 and the two-session
    # signal-to-PnL lag are already established before the requested window.
    full_returns = strategy.pct_change().fillna(0.0)
    hedged_returns, hedge_position = hedge_returns(
        full_returns,
        index,
        lookback=LOOKBACK,
        threshold=THRESHOLD,
        hedge_ratio=MAX_HEDGE_RATIO,
    )
    hedged_nav = (1.0 + hedged_returns).cumprod()

    common_end = min(strategy.index.max(), index.index.max())
    available_dates = strategy.index.intersection(index.index)
    available_dates = available_dates[
        (available_dates >= START_DATE) & (available_dates <= common_end)
    ]
    if len(available_dates) < 2:
        raise RuntimeError("2026-06-01 onward has insufficient common observations")
    actual_start = available_dates.min()
    actual_end = available_dates.max()

    frame = pd.DataFrame(
        {
            "base_lgbm": strategy.reindex(available_dates),
            "lgbm_csi300_hedged50": hedged_nav.reindex(available_dates),
            "csi300": index["close"].reindex(available_dates),
            "hedge_ratio": hedge_position.reindex(available_dates),
        },
        index=available_dates,
    ).dropna()
    for column in ("base_lgbm", "lgbm_csi300_hedged50", "csi300"):
        frame[column] = frame[column].div(float(frame[column].iloc[0]))
    frame.index.name = "date"

    position_changes = hedge_position.diff().abs().gt(1e-12)
    return frame, position_changes.reindex(frame.index).fillna(False), actual_start, actual_end


def metrics_for(frame: pd.DataFrame) -> dict[str, dict]:
    labels = {
        "lgbm_csi300_hedged50": "LGBM + CSI300 dynamic hedge (max 50%)",
        "base_lgbm": "Base LGBM stock sleeve",
        "csi300": "CSI300 price index",
    }
    result = {}
    for column, label in labels.items():
        returns = frame[column].pct_change().fillna(0.0)
        result[column] = {"label": label, **performance(returns)}
    return result


def _nice_bounds(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 0.02)
    step = max(0.01, math.ceil((span / 5.0) * 100.0) / 100.0)
    lower = math.floor((low - step * 0.55) / step) * step
    upper = math.ceil((high + step * 0.55) / step) * step
    return max(0.0, lower), upper


def render_image(frame: pd.DataFrame, metrics: dict[str, dict]) -> None:
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
    nav_columns = ["lgbm_csi300_hedged50", "base_lgbm", "csi300"]
    y_min, y_max = _nice_bounds(frame[nav_columns].stack())

    def x(date: pd.Timestamp) -> float:
        total = max((end - start).days, 1)
        return left + (date - start).days / total * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    canvas.text(left, 38, "LGBM BACKTEST | JUN 2026 TO LATEST AVAILABLE", ink, 4)
    canvas.text(
        left,
        87,
        f"{start:%Y-%m-%d} TO {end:%Y-%m-%d} | START NAV 1.00 | {len(frame)} TRADING DAYS",
        muted,
        2,
    )

    legend = [
        ("lgbm_csi300_hedged50", "HEDGED LGBM", green, left),
        ("base_lgbm", "BASE LGBM", blue, left + 390),
        ("csi300", "CSI300", orange, left + 730),
    ]
    for column, label, color, offset in legend:
        canvas.line(offset, 130, offset + 42, 130, color, 4)
        total_return = metrics[column]["total_return"]
        canvas.text(offset + 54, 123, f"{label} {total_return:+.2%}", ink, 2)

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        py = y(value)
        canvas.line(left, py, right, py, grid)
        canvas.text(28, round(py) - 7, f"{value:.2f}X", muted, 2)
    canvas.dashed_line(left, y(1.0), right, y(1.0), muted)

    month_ticks = pd.date_range(start.normalize(), end.normalize(), freq="MS")
    if start not in month_ticks:
        month_ticks = month_ticks.insert(0, start)
    for date in month_ticks:
        if date > end:
            continue
        px = round(x(date))
        canvas.line(px, bottom, px, bottom + 7, grid)
        canvas.text(max(left, px - 36), bottom + 20, f"{date:%m-%d}", muted, 2)

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
    minimum_gap = 52
    for i in range(1, len(endpoints)):
        endpoints[i][0] = max(endpoints[i][0], endpoints[i - 1][0] + minimum_gap)
    overflow = endpoints[-1][0] - (bottom - 32)
    if overflow > 0:
        for item in endpoints:
            item[0] -= overflow
    endpoint_labels = {
        "lgbm_csi300_hedged50": "HEDGED",
        "base_lgbm": "BASE",
        "csi300": "CSI300",
    }
    for label_y, column, color, value in endpoints:
        canvas.text(right + 18, round(label_y) - 18, f"{endpoint_labels[column]} {value:.3f}X", color, 2)
        canvas.text(
            right + 18,
            round(label_y) + 4,
            f"MDD {metrics[column]['max_drawdown']:.2%}",
            muted,
            2,
        )

    hedged = metrics["lgbm_csi300_hedged50"]
    base = metrics["base_lgbm"]
    benchmark = metrics["csi300"]
    canvas.text(
        left,
        HEIGHT - 102,
        f"HEDGED: ANN {hedged['annual_return']:+.2%} | SHARPE {hedged['sharpe']:.2f} | MDD {hedged['max_drawdown']:.2%}",
        ink,
        2,
    )
    canvas.text(
        left,
        HEIGHT - 76,
        f"BASE: ANN {base['annual_return']:+.2%} | SHARPE {base['sharpe']:.2f} | MDD {base['max_drawdown']:.2%}",
        ink,
        2,
    )
    canvas.text(
        left,
        HEIGHT - 50,
        f"CSI300: ANN {benchmark['annual_return']:+.2%} | SHARPE {benchmark['sharpe']:.2f} | MDD {benchmark['max_drawdown']:.2%}",
        ink,
        2,
    )
    canvas.text(
        left,
        HEIGHT - 24,
        "ANN IS MECHANICAL 252-DAY ANNUALIZATION | HEDGE: MA120 | NEXT-OPEN | 2BP CHANGE COST",
        muted,
        2,
    )
    canvas.save(OUTPUT_IMAGE)


def main() -> None:
    frame, position_changes, actual_start, actual_end = load_backtest()
    metrics = metrics_for(frame)
    frame.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")
    render_image(frame, metrics)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic",
        "requested_start": START_DATE.date().isoformat(),
        "actual_window": [actual_start.date().isoformat(), actual_end.date().isoformat()],
        "latest_available_note": (
            "The requested through-now window ends at the latest common strategy "
            "and CSI300 observation, not the current calendar date."
        ),
        "strategy": {
            "stock_sleeve": "frozen 75/25 Alpha158-Barra LGBM, Top5, fixed 10-day rebalance",
            "stock_execution": "T-close signal, T+1 open execution, 12 bps one-way cost",
            "hedge": {
                "benchmark": "CSI300",
                "lookback": LOOKBACK,
                "threshold": THRESHOLD,
                "maximum_ratio": MAX_HEDGE_RATIO,
                "execution": "T-close signal, T+1 open; open-to-next-open PnL proxy",
                "position_change_cost_bps": 2.0,
            },
        },
        "metrics": metrics,
        "average_actual_hedge_ratio": float(frame["hedge_ratio"].mean()),
        "hedge_position_change_days": int(position_changes.sum()),
        "limitations": [
            "This is a revealed historical diagnostic, not a fresh out-of-sample result.",
            "The Top50 universe is backfilled and is exposed to constituent and survivorship bias.",
            "CSI300 close prices are a price-index benchmark and exclude dividends.",
            "The index hedge is a futures proxy without basis, margin, roll, financing, or slippage beyond the stated position-change cost.",
            "Annualized return is a mechanical 252-trading-day extrapolation from a short window.",
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
        "# LGBM 2026-06 to latest available backtest",
        "",
        f"- Window: {actual_start:%Y-%m-%d} to {actual_end:%Y-%m-%d} ({len(frame)} trading days).",
        "- All curves start at NAV 1.00 on the first common trading day.",
        "- Annual return is a mechanical 252-day annualization of this short window.",
        "",
        "| Series | Total return | Annualized | Sharpe | Max drawdown | Terminal NAV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for column in ("lgbm_csi300_hedged50", "base_lgbm", "csi300"):
        item = metrics[column]
        lines.append(
            f"| {item['label']} | {item['total_return']:.2%} | {item['annual_return']:.2%} | "
            f"{item['sharpe']:.3f} | {item['max_drawdown']:.2%} | {item['terminal_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Average actual hedge ratio: {payload['average_actual_hedge_ratio']:.1%}.",
            f"- Hedge position change days: {payload['hedge_position_change_days']}.",
            "",
            "## Limitations",
            "",
            *(f"- {item}" for item in payload["limitations"]),
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
