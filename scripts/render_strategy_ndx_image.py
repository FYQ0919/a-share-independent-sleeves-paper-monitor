from __future__ import annotations

from pathlib import Path
import struct
import zlib

import httpx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
WIDTH = 1400
HEIGHT = 820
MARGIN = {"left": 104, "right": 225, "top": 145, "bottom": 108}


class Canvas:
    def __init__(self, width: int, height: int, color: tuple[int, int, int]):
        self.width = width
        self.height = height
        self.pixels = bytearray(color * (width * height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int], alpha: float = 1.0):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        index = (y * self.width + x) * 3
        if alpha >= 1:
            self.pixels[index:index + 3] = bytes(color)
            return
        for offset, value in enumerate(color):
            current = self.pixels[index + offset]
            self.pixels[index + offset] = round(current * (1 - alpha) + value * alpha)

    def line(self, x0: float, y0: float, x1: float, y1: float, color, width: int = 1):
        dx = x1 - x0
        dy = y1 - y0
        steps = max(1, round(max(abs(dx), abs(dy))))
        radius = max(0, width // 2)
        for step in range(steps + 1):
            x = round(x0 + dx * step / steps)
            y = round(y0 + dy * step / steps)
            for ox in range(-radius, radius + 1):
                for oy in range(-radius, radius + 1):
                    if ox * ox + oy * oy <= (radius + 0.5) ** 2:
                        self.set_pixel(x + ox, y + oy, color)

    def dashed_line(self, x0, y0, x1, y1, color, dash: int = 7, gap: int = 6):
        length = max(1, round(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5))
        for start in range(0, length, dash + gap):
            end = min(start + dash, length)
            self.line(
                x0 + (x1 - x0) * start / length,
                y0 + (y1 - y0) * start / length,
                x0 + (x1 - x0) * end / length,
                y0 + (y1 - y0) * end / length,
                color,
            )

    def circle(self, cx: int, cy: int, radius: int, color):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.set_pixel(x, y, color)

    def text(self, x: int, y: int, value: str, color, scale: int = 2):
        cursor = x
        for char in value.upper():
            glyph = FONT.get(char, FONT["?"])
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        for ox in range(scale):
                            for oy in range(scale):
                                self.set_pixel(cursor + column * scale + ox, y + row * scale + oy, color)
            cursor += 6 * scale

    def save(self, path: Path):
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * stride:(y + 1) * stride])

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


FONT = {
    " ": ["00000"] * 7,
    "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    ":": ["00000", "00110", "00110", "00000", "00110", "00110", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "|": ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
    **{str(i): glyph for i, glyph in enumerate([
        ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
        ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ])},
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
        ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
        ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
        ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
        ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    ]))
}


def load_curves() -> pd.DataFrame:
    strategy = pd.read_csv(BASE_DIR / "reports" / "lgbm_new_factors_v1_curves.csv", parse_dates=["date"])[
        ["date", "selected_model_blend"]
    ]
    url = "https://api.nasdaq.com/api/quote/NDX/historical?assetclass=index&fromdate=2019-12-30&todate=2026-08-25&limit=4000"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    rows = httpx.get(url, headers=headers, timeout=30, trust_env=False).json()["data"]["tradesTable"]["rows"]
    ndx = pd.DataFrame(rows)
    ndx["date"] = pd.to_datetime(ndx["date"], format="%m/%d/%Y")
    ndx["ndx"] = pd.to_numeric(ndx["close"].astype(str).str.replace(r"[$,]", "", regex=True))
    start, end = pd.Timestamp("2020-01-02"), pd.Timestamp("2026-08-25")
    strategy = strategy[strategy["date"].between(start, end)].copy()
    ndx = ndx[ndx["date"].between(start, end)][["date", "ndx"]].sort_values("date")
    strategy["strategy"] = strategy["selected_model_blend"] / strategy["selected_model_blend"].iloc[0]
    ndx["ndx"] /= ndx["ndx"].iloc[0]
    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    return calendar.merge(strategy[["date", "strategy"]], on="date", how="left").merge(ndx, on="date", how="left").ffill().dropna()


def main():
    curve = load_curves()
    canvas = Canvas(WIDTH, HEIGHT, (247, 248, 246))
    green, orange, ink, muted, grid = (22, 125, 89), (208, 96, 54), (32, 40, 36), (102, 113, 107), (217, 223, 219)
    left, right = MARGIN["left"], WIDTH - MARGIN["right"]
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    y_min, y_max = 0.65, 4.65
    start, end = curve["date"].iloc[0], curve["date"].iloc[-1]

    def x(date):
        return left + (date - start).days / (end - start).days * (right - left)

    def y(value):
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    canvas.text(left, 42, "CURRENT 75/25 LGBM STRATEGY VS NASDAQ-100", ink, 4)
    canvas.text(left, 92, "2020-01-02 TO 2026-08-25  |  START NAV = 1.00", muted, 2)
    canvas.line(left, 122, left + 42, 122, green, 4)
    canvas.text(left + 54, 115, "STRATEGY  CAGR 24.34%", ink, 2)
    canvas.line(left + 360, 122, left + 402, 122, orange, 4)
    canvas.text(left + 414, 115, "NASDAQ-100  CAGR 19.64%", ink, 2)

    for tick in [1, 2, 3, 4]:
        py = y(tick)
        canvas.line(left, py, right, py, grid)
        canvas.text(35, round(py) - 7, f"{tick:.1f}X", muted, 2)
    canvas.dashed_line(left, y(1), right, y(1), muted)
    for year in range(2020, 2027):
        date = pd.Timestamp(f"{year}-01-01")
        if date < start:
            date = start
        px = round(x(date))
        canvas.line(px, bottom, px, bottom + 6, grid)
        canvas.text(px - 24, bottom + 18, str(year), muted, 2)

    for column, color in [("strategy", green), ("ndx", orange)]:
        points = [(x(date), y(value)) for date, value in curve[["date", column]].itertuples(index=False, name=None)]
        for first, second in zip(points, points[1:]):
            canvas.line(*first, *second, color, 3)

    strategy_value = float(curve["strategy"].iloc[-1])
    ndx_value = float(curve["ndx"].iloc[-1])
    canvas.circle(round(x(end)), round(y(strategy_value)), 6, green)
    canvas.circle(round(x(end)), round(y(ndx_value)), 6, orange)
    canvas.text(right + 18, round(y(strategy_value)) - 20, f"STRATEGY {strategy_value:.2f}X", green, 2)
    canvas.text(right + 18, round(y(strategy_value)) + 2, "TOTAL +325.22%", green, 2)
    canvas.text(right + 18, round(y(ndx_value)) - 20, f"NASDAQ-100 {ndx_value:.2f}X", orange, 2)
    canvas.text(right + 18, round(y(ndx_value)) + 2, "TOTAL +229.22%", orange, 2)
    canvas.text(left, HEIGHT - 50, "TOP5 | REBALANCE 10D | NEXT-OPEN | 12BP ONE-WAY COST", muted, 2)
    canvas.text(left, HEIGHT - 26, "NDX PRICE INDEX: EXCLUDES DIVIDENDS, FEES, TAXES AND FX", muted, 2)
    output = BASE_DIR / "reports" / "current_strategy_vs_nasdaq100_2020_2026.png"
    canvas.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
