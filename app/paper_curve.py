from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import shutil
from typing import Dict

import numpy as np
import pandas as pd

from app.storage import Storage
from scripts.render_strategy_ndx_image import Canvas


WIDTH = 1400
HEIGHT = 820
LEFT = 105
RIGHT = 1135
TOP = 185
BOTTOM = 675


class PaperEquityCurveService:
    """Export an honest forward-paper NAV history and a WeCom-ready PNG."""

    def __init__(self, storage: Storage, report_dir: Path):
        self.storage = storage
        self.output_dir = Path(report_dir) / "paper_equity"

    def generate(
        self,
        stock_account_id: str,
        hedge_account_id: str | None = None,
    ) -> Dict:
        stock = self._history(stock_account_id, "stock_nav")
        if stock.empty:
            raise RuntimeError("stock paper account has no snapshots")
        if hedge_account_id:
            hedge = self._history(hedge_account_id, "composite_nav")
            frame = stock.merge(hedge, on="date", how="outer").sort_values("date")
        else:
            frame = stock.copy()
            frame["composite_nav"] = frame["stock_nav"]
            frame["active_hedge_ratio"] = 0.0
            frame["target_hedge_ratio"] = 0.0
        frame["stock_nav"] = frame["stock_nav"].ffill()
        frame["composite_nav"] = frame["composite_nav"].fillna(frame["stock_nav"])
        frame["active_hedge_ratio"] = frame["active_hedge_ratio"].ffill().fillna(0.0)
        frame["target_hedge_ratio"] = frame["target_hedge_ratio"].ffill().fillna(0.0)
        frame = frame.dropna(subset=["stock_nav", "composite_nav"])
        if frame.empty:
            raise RuntimeError("paper NAV history has no aligned observations")

        initial_stock = float(frame["stock_nav"].iloc[0])
        initial_composite = float(frame["composite_nav"].iloc[0])
        frame["stock_equity"] = frame["stock_nav"].div(initial_stock)
        frame["composite_equity"] = frame["composite_nav"].div(initial_composite)
        frame["stock_return"] = frame["stock_equity"].sub(1.0)
        frame["composite_return"] = frame["composite_equity"].sub(1.0)
        frame["composite_drawdown"] = frame["composite_equity"].div(
            frame["composite_equity"].cummax()
        ).sub(1.0)

        latest_date = frame["date"].iloc[-1].date()
        account_slug = self._safe_slug(hedge_account_id or stock_account_id)
        directory = self.output_dir / account_slug
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "curve.csv"
        json_path = directory / "latest.json"
        dated_image = directory / f"equity_{latest_date:%Y%m%d}.png"
        latest_image = directory / "latest.png"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10f")

        summary = self._summary(frame, stock_account_id, hedge_account_id)
        self._render(frame, summary, dated_image)
        shutil.copyfile(dated_image, latest_image)
        summary["artifacts"] = {
            "curve_csv": str(csv_path.resolve()),
            "daily_image": str(dated_image.resolve()),
            "latest_image": str(latest_image.resolve()),
        }
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            **summary,
            "image_path": str(dated_image.resolve()),
            "latest_image_path": str(latest_image.resolve()),
            "json_path": str(json_path.resolve()),
            "csv_path": str(csv_path.resolve()),
        }

    def generate_independent_sleeves(self, monitor_account_id: str) -> Dict:
        rows = self.storage.strategy_decision_history(monitor_account_id, 10_000)
        records = []
        for row in rows:
            signal_date = row.get("signal_date")
            sleeves = row.get("sleeves") or {}
            trend = sleeves.get("trend") or {}
            lgbm = sleeves.get("lgbm") or {}
            if not signal_date or row.get("nav") is None:
                continue
            records.append({
                "date": pd.Timestamp(signal_date),
                "combined_nav": float(row["nav"]),
                "trend_nav": float(trend.get("nav", 0.0)),
                "lgbm_nav": float(lgbm.get("nav", 0.0)),
                "active_hedge_ratio": float(trend.get("active_hedge_ratio", 0.0)),
                "target_hedge_ratio": float(trend.get("target_hedge_ratio", 0.0)),
                "initial_capital": float(row.get("initial_capital", 0.0)),
                "trend_initial_capital": float(trend.get("initial_capital", 0.0)),
                "lgbm_initial_capital": float(lgbm.get("initial_capital", 0.0)),
            })
        if not records:
            raise RuntimeError("independent-sleeve paper monitor has no snapshots")
        frame = (
            pd.DataFrame(records)
            .sort_values("date")
            .drop_duplicates("date", keep="last")
        )
        initial = frame.iloc[0]
        denominators = {
            "combined_equity": float(initial["initial_capital"]),
            "trend_equity": float(initial["trend_initial_capital"]),
            "lgbm_equity": float(initial["lgbm_initial_capital"]),
        }
        sources = {
            "combined_equity": "combined_nav",
            "trend_equity": "trend_nav",
            "lgbm_equity": "lgbm_nav",
        }
        for target, source in sources.items():
            denominator = denominators[target]
            if denominator <= 0:
                raise RuntimeError("independent-sleeve initial capital is invalid")
            frame[target] = frame[source].div(denominator)
        frame["combined_drawdown"] = frame["combined_equity"].div(
            frame["combined_equity"].cummax()
        ).sub(1.0)

        latest = frame.iloc[-1]
        summary = {
            "status": "forward_paper_only",
            "monitor_account_id": monitor_account_id,
            "start_date": frame["date"].iloc[0].date().isoformat(),
            "end_date": latest["date"].date().isoformat(),
            "observations": int(len(frame)),
            "composite_nav": float(latest["combined_nav"]),
            "composite_return": float(latest["combined_equity"] - 1.0),
            "trend_return": float(latest["trend_equity"] - 1.0),
            "lgbm_return": float(latest["lgbm_equity"] - 1.0),
            "max_drawdown": float(frame["combined_drawdown"].min()),
            "active_hedge_ratio": float(latest["active_hedge_ratio"]),
            "target_hedge_ratio": float(latest["target_hedge_ratio"]),
            "backfilled": False,
            "sleeve_rebalance": "none",
        }
        directory = self.output_dir / self._safe_slug(monitor_account_id)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "curve.csv"
        json_path = directory / "latest.json"
        dated_image = directory / f"equity_{latest['date']:%Y%m%d}.png"
        latest_image = directory / "latest.png"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.10f")
        self._render_independent(frame, summary, dated_image)
        shutil.copyfile(dated_image, latest_image)
        summary["artifacts"] = {
            "curve_csv": str(csv_path.resolve()),
            "daily_image": str(dated_image.resolve()),
            "latest_image": str(latest_image.resolve()),
        }
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            **summary,
            "image_path": str(dated_image.resolve()),
            "latest_image_path": str(latest_image.resolve()),
            "json_path": str(json_path.resolve()),
            "csv_path": str(csv_path.resolve()),
        }

    def _history(self, account_id: str, nav_column: str) -> pd.DataFrame:
        rows = self.storage.strategy_decision_history(account_id, 10_000)
        records = []
        for row in rows:
            signal_date = row.get("signal_date")
            nav = row.get("nav")
            if not signal_date or nav is None:
                continue
            record = {"date": pd.Timestamp(signal_date), nav_column: float(nav)}
            if nav_column == "composite_nav":
                record["active_hedge_ratio"] = float(
                    row.get("active_hedge_ratio", 0.0)
                )
                record["target_hedge_ratio"] = float(
                    row.get("target_hedge_ratio", 0.0)
                )
            records.append(record)
        if not records:
            columns = ["date", nav_column]
            if nav_column == "composite_nav":
                columns.extend(["active_hedge_ratio", "target_hedge_ratio"])
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(records)
            .sort_values("date")
            .drop_duplicates("date", keep="last")
        )

    @staticmethod
    def _summary(frame: pd.DataFrame, stock_account_id: str, hedge_account_id: str | None):
        latest = frame.iloc[-1]
        return {
            "status": "forward_paper_only",
            "stock_account_id": stock_account_id,
            "hedge_account_id": hedge_account_id,
            "start_date": frame["date"].iloc[0].date().isoformat(),
            "end_date": latest["date"].date().isoformat(),
            "observations": int(len(frame)),
            "stock_nav": float(latest["stock_nav"]),
            "composite_nav": float(latest["composite_nav"]),
            "stock_return": float(latest["stock_return"]),
            "composite_return": float(latest["composite_return"]),
            "max_drawdown": float(frame["composite_drawdown"].min()),
            "active_hedge_ratio": float(latest["active_hedge_ratio"]),
            "target_hedge_ratio": float(latest["target_hedge_ratio"]),
            "backfilled": False,
        }

    @staticmethod
    def _render(frame: pd.DataFrame, summary: Dict, output: Path) -> None:
        canvas = Canvas(WIDTH, HEIGHT, (247, 248, 246))
        composite_color = (33, 128, 91)
        stock_color = (23, 107, 135)
        ink = (32, 40, 36)
        muted = (102, 113, 107)
        grid = (217, 223, 219)
        dates = frame["date"].tolist()
        values = frame[["composite_equity", "stock_equity"]].to_numpy(dtype=float)
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        padding = max((value_max - value_min) * 0.18, 0.005)
        y_min = value_min - padding
        y_max = value_max + padding

        def x(index: int) -> float:
            if len(frame) == 1:
                return LEFT
            return LEFT + (RIGHT - LEFT) * index / (len(frame) - 1)

        def y(value: float) -> float:
            return BOTTOM - (value - y_min) / max(y_max - y_min, 1e-12) * (BOTTOM - TOP)

        canvas.text(LEFT, 38, "FORWARD PAPER EQUITY CURVE", ink, 4)
        canvas.text(
            LEFT,
            88,
            f"{summary['start_date']} TO {summary['end_date']} | OBSERVATIONS {summary['observations']} | START NAV 1.0000",
            muted,
            2,
        )
        canvas.line(LEFT, 123, LEFT + 42, 123, composite_color, 4)
        canvas.text(
            LEFT + 54,
            116,
            f"HEDGED {float(frame['composite_equity'].iloc[-1]):.4f}X | RETURN {summary['composite_return']:+.2%}",
            ink,
            2,
        )
        canvas.line(LEFT + 500, 123, LEFT + 542, 123, stock_color, 4)
        canvas.text(
            LEFT + 554,
            116,
            f"STOCK {float(frame['stock_equity'].iloc[-1]):.4f}X | RETURN {summary['stock_return']:+.2%}",
            ink,
            2,
        )

        for tick in np.linspace(y_min, y_max, 5):
            py = y(float(tick))
            canvas.line(LEFT, py, RIGHT, py, grid)
            canvas.text(20, round(py) - 7, f"{tick:.3f}X", muted, 2)
        baseline = y(1.0)
        if TOP <= baseline <= BOTTOM:
            canvas.dashed_line(LEFT, baseline, RIGHT, baseline, muted)

        label_indices = sorted(
            set(
                round(value)
                for value in np.linspace(0, max(len(frame) - 1, 0), min(5, len(frame)))
            )
        )
        for index in label_indices:
            px = round(x(index))
            canvas.line(px, BOTTOM, px, BOTTOM + 6, grid)
            canvas.text(max(LEFT, px - 48), BOTTOM + 18, dates[index].strftime("%Y-%m-%d"), muted, 2)

        for column, color, width in (
            ("stock_equity", stock_color, 2),
            ("composite_equity", composite_color, 4),
        ):
            points = [
                (x(index), y(float(value)))
                for index, value in enumerate(frame[column].tolist())
            ]
            for first, second in zip(points, points[1:]):
                canvas.line(*first, *second, color, width)
            canvas.circle(round(points[-1][0]), round(points[-1][1]), 6, color)

        composite_y = round(y(float(frame["composite_equity"].iloc[-1])))
        stock_y = round(y(float(frame["stock_equity"].iloc[-1])))
        if abs(composite_y - stock_y) < 28:
            composite_y -= 18
            stock_y += 18
        canvas.text(RIGHT + 18, composite_y - 7, "HEDGED", composite_color, 2)
        canvas.text(RIGHT + 18, stock_y - 7, "STOCK", stock_color, 2)
        canvas.text(
            LEFT,
            HEIGHT - 62,
            f"MAX DRAWDOWN {summary['max_drawdown']:.2%} | ACTIVE HEDGE {summary['active_hedge_ratio']:.0%} | NEXT TARGET {summary['target_hedge_ratio']:.0%}",
            muted,
            2,
        )
        canvas.text(
            LEFT,
            HEIGHT - 34,
            "FORWARD PAPER ONLY | NEXT-OPEN EXECUTION | COSTS INCLUDED | NO HISTORICAL BACKFILL",
            muted,
            2,
        )
        canvas.save(output)

    @staticmethod
    def _render_independent(frame: pd.DataFrame, summary: Dict, output: Path) -> None:
        canvas = Canvas(WIDTH, HEIGHT, (247, 248, 246))
        colors = {
            "combined_equity": (33, 128, 91),
            "trend_equity": (23, 107, 135),
            "lgbm_equity": (194, 65, 59),
        }
        ink = (32, 40, 36)
        muted = (102, 113, 107)
        grid = (217, 223, 219)
        values = frame[list(colors)].to_numpy(dtype=float)
        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        padding = max((value_max - value_min) * 0.18, 0.005)
        y_min, y_max = value_min - padding, value_max + padding

        def x(index: int) -> float:
            return LEFT if len(frame) == 1 else LEFT + (RIGHT - LEFT) * index / (len(frame) - 1)

        def y(value: float) -> float:
            return BOTTOM - (value - y_min) / max(y_max - y_min, 1e-12) * (BOTTOM - TOP)

        canvas.text(LEFT, 38, "75/25 INDEPENDENT SLEEVES PAPER MONITOR", ink, 4)
        canvas.text(
            LEFT,
            88,
            f"{summary['start_date']} TO {summary['end_date']} | OBSERVATIONS {summary['observations']} | NO CROSS-SLEEVE REBALANCE",
            muted,
            2,
        )
        legends = [
            ("combined_equity", "COMBINED", LEFT),
            ("trend_equity", "TREND 75%", LEFT + 330),
            ("lgbm_equity", "LGBM 25%", LEFT + 660),
        ]
        for column, label, px in legends:
            canvas.line(px, 123, px + 38, 123, colors[column], 4)
            canvas.text(px + 50, 116, f"{label} {float(frame[column].iloc[-1]):.4f}X", ink, 2)
        for tick in np.linspace(y_min, y_max, 5):
            py = y(float(tick))
            canvas.line(LEFT, py, RIGHT, py, grid)
            canvas.text(20, round(py) - 7, f"{tick:.3f}X", muted, 2)
        baseline = y(1.0)
        if TOP <= baseline <= BOTTOM:
            canvas.dashed_line(LEFT, baseline, RIGHT, baseline, muted)
        dates = frame["date"].tolist()
        indices = sorted(set(round(value) for value in np.linspace(0, len(frame) - 1, min(5, len(frame)))))
        for index in indices:
            px = round(x(index))
            canvas.line(px, BOTTOM, px, BOTTOM + 6, grid)
            canvas.text(max(LEFT, px - 48), BOTTOM + 18, dates[index].strftime("%Y-%m-%d"), muted, 2)
        for column, width in (("lgbm_equity", 2), ("trend_equity", 3), ("combined_equity", 4)):
            points = [(x(i), y(float(value))) for i, value in enumerate(frame[column])]
            for first, second in zip(points, points[1:]):
                canvas.line(*first, *second, colors[column], width)
            canvas.circle(round(points[-1][0]), round(points[-1][1]), 6, colors[column])
        canvas.text(
            LEFT,
            HEIGHT - 62,
            f"RETURN {summary['composite_return']:+.2%} | MAX DRAWDOWN {summary['max_drawdown']:.2%} | ACTIVE HEDGE {summary['active_hedge_ratio']:.0%}",
            muted,
            2,
        )
        canvas.text(
            LEFT,
            HEIGHT - 34,
            "FORWARD PAPER ONLY | NEXT-OPEN EXECUTION | COSTS INCLUDED | NO HISTORICAL BACKFILL",
            muted,
            2,
        )
        canvas.save(output)

    @staticmethod
    def _safe_slug(value: str) -> str:
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
