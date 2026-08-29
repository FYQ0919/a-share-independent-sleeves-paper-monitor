from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_strategy_ndx_image import Canvas


WIDTH = 1500
HEIGHT = 860
LEFT = 105
RIGHT = 1240
TOP = 150
BOTTOM = 740


def main() -> None:
    input_path = ROOT / "reports" / "lgbm_high_sharpe_overlay_v1_curves.csv"
    output_path = ROOT / "reports" / "lgbm_high_sharpe_overlay_v1.png"
    frame = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    if len(frame) < 2:
        raise RuntimeError("high-Sharpe curve file has insufficient rows")

    canvas = Canvas(WIDTH, HEIGHT, (247, 248, 246))
    base_color = (23, 107, 135)
    managed_color = (33, 128, 91)
    ink = (32, 40, 36)
    muted = (102, 113, 107)
    grid = (217, 223, 219)
    start, end = frame["date"].iloc[0], frame["date"].iloc[-1]
    y_min = 0.4
    y_max = max(float(frame[["base_equity", "managed_equity"]].max().max()) * 1.08, 1.2)

    def x(when: pd.Timestamp) -> float:
        return LEFT + (when - start).days / max((end - start).days, 1) * (RIGHT - LEFT)

    def y(value: float) -> float:
        return BOTTOM - (value - y_min) / (y_max - y_min) * (BOTTOM - TOP)

    canvas.text(LEFT, 38, "HIGH-SHARPE 75/25 LGBM WITH VOLATILITY TARGET", ink, 4)
    canvas.text(LEFT, 88, "2020-01-02 TO 2026-08-25 | START NAV 1.00", muted, 2)
    canvas.line(LEFT, 122, LEFT + 42, 122, base_color, 4)
    canvas.text(LEFT + 54, 115, "BASE LGBM 7.06X | FULL SHARPE 1.106", ink, 2)
    canvas.line(LEFT + 520, 122, LEFT + 562, 122, managed_color, 4)
    canvas.text(LEFT + 574, 115, "VOL TARGET 5.16X | FULL SHARPE 1.133", ink, 2)

    for tick in range(1, 8):
        if tick > y_max:
            continue
        yy = y(float(tick))
        canvas.line(LEFT, yy, RIGHT, yy, grid)
        canvas.text(35, round(yy) - 7, f"{tick}.0X", muted, 2)
    for year in range(start.year, end.year + 1):
        when = max(start, pd.Timestamp(year, 1, 1))
        xx = round(x(when))
        canvas.line(xx, BOTTOM, xx, BOTTOM + 6, grid)
        canvas.text(xx - 24, BOTTOM + 18, str(year), muted, 2)

    for column, color in (
        ("base_equity", base_color),
        ("managed_equity", managed_color),
    ):
        points = [
            (x(when), y(float(value)))
            for when, value in frame[["date", column]].itertuples(index=False, name=None)
        ]
        for first, second in zip(points, points[1:]):
            canvas.line(*first, *second, color, 3)

    base_value = float(frame["base_equity"].iloc[-1])
    managed_value = float(frame["managed_equity"].iloc[-1])
    canvas.circle(round(x(end)), round(y(base_value)), 6, base_color)
    canvas.circle(round(x(end)), round(y(managed_value)), 6, managed_color)
    canvas.text(RIGHT + 18, round(y(base_value)) - 8, f"BASE {base_value:.2f}X", base_color, 2)
    canvas.text(RIGHT + 18, round(y(managed_value)) - 8, f"VOL TARGET {managed_value:.2f}X", managed_color, 2)
    canvas.text(LEFT, HEIGHT - 58, "VOL60 | TARGET 22% | EXPOSURE 65%-100% | CASH ONLY | 12BP CHANGES", muted, 2)
    canvas.text(LEFT, HEIGHT - 30, "HISTORICAL DIAGNOSTIC ONLY | 2024-LATEST REVEALED SHARPE 1.854", muted, 2)
    canvas.save(output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
