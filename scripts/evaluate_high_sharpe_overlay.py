from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.volatility_target_overlay import VolatilityTargetConfig, VolatilityTargetOverlay


SELECTION_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")


def performance(returns: pd.Series) -> dict[str, float | int | str]:
    clean = returns.astype(float)
    equity = (1.0 + clean).cumprod()
    volatility = float(clean.std(ddof=0) * np.sqrt(252.0))
    return {
        "start_date": clean.index[0].date().isoformat(),
        "end_date": clean.index[-1].date().isoformat(),
        "return_periods": int(len(clean)),
        "terminal_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": float(equity.iloc[-1] ** (252.0 / len(clean)) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": (
            float(clean.mean() * 252.0 / volatility) if volatility > 0 else 0.0
        ),
        "max_drawdown": float(equity.div(equity.cummax()).sub(1.0).min()),
    }


def evaluate(curve: pd.Series) -> tuple[pd.DataFrame, dict]:
    normalized = curve.astype(float).div(float(curve.iloc[0]))
    base_returns = normalized.pct_change().fillna(0.0)
    config = VolatilityTargetConfig()
    overlay = VolatilityTargetOverlay(config)
    result = overlay.apply(base_returns)
    result["base_equity"] = (1.0 + result["base_return"]).cumprod()

    windows = {
        "selection_2020_2023": result.loc[:SELECTION_END],
        "revealed_diagnostic_2024_latest": result.loc[TEST_START:],
        "full_2020_latest": result,
    }
    metrics = {}
    for name, frame in windows.items():
        metrics[name] = {
            "base_lgbm": performance(frame["base_return"]),
            "volatility_target_lgbm": performance(frame["managed_return"]),
            "average_exposure": float(frame["risk_exposure"].mean()),
            "exposure_turnover": float(frame["exposure_turnover"].sum()),
            "overlay_cost": float(frame["overlay_cost"].sum()),
        }

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_candidate_not_deployed",
        "strategy": "frozen 75/25 LGBM plus causal 60-session volatility target",
        "parameters": {
            "window": config.window,
            "min_observations": config.min_observations,
            "target_volatility": config.target_volatility,
            "minimum_exposure": config.minimum_exposure,
            "maximum_exposure": config.maximum_exposure,
            "exposure_change_cost_bps": config.exposure_change_cost_bps,
        },
        "selection_contract": {
            "parameter_source": "volatility overlay selected on the 2020-2023 research window",
            "selection_window": [result.index[0].date().isoformat(), "2023-12-31"],
            "revealed_diagnostic_window": [
                "2024-01-01",
                result.index[-1].date().isoformat(),
            ],
            "diagnostic_reused_for_other_research": True,
        },
        "metrics": metrics,
        "causality_audit": {
            "exposure_at_t_uses_returns_through": "T-1",
            "future_data_used": False,
            "cash_only": True,
            "leverage_allowed": False,
        },
        "production_change": False,
        "strict_no_lookahead_certified": False,
        "limitations": [
            "The 2024-latest diagnostic window has already been revealed and is not a fresh holdout.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "The overlay is evaluated on a stored strategy return stream, not a broker fill replay.",
            "A new forward paper window is required before production promotion.",
        ],
    }
    return result, payload


def render_svg(result: pd.DataFrame, output: Path) -> None:
    width, height = 1400, 820
    left, right, top, bottom = 105, 270, 120, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    series = {
        "base_equity": ("Base 75/25 LGBM", "#176B87"),
        "managed_equity": ("LGBM + volatility target", "#21805B"),
    }
    values = result[list(series)].to_numpy(dtype=float)
    minimum, maximum = float(np.nanmin(values)), float(np.nanmax(values))
    padding = max((maximum - minimum) * 0.06, 0.05)
    y_min, y_max = max(0.0, minimum - padding), maximum + padding

    def x(index: int) -> float:
        return left + plot_width * index / max(len(result) - 1, 1)

    def y(value: float) -> float:
        return top + plot_height * (y_max - value) / max(y_max - y_min, 1e-12)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        '<text x="105" y="42" font-family="Segoe UI,Arial" font-size="25" fill="#1F2933">Frozen 75/25 LGBM with causal volatility target</text>',
        '<text x="105" y="74" font-family="Segoe UI,Arial" font-size="13" fill="#52606D">2020-01-02 to 2026-08-25 | MA volatility window 60 | target 22% | exposure 65%-100% | 12 bp changes</text>',
        '<text x="105" y="97" font-family="Segoe UI,Arial" font-size="12" fill="#52606D">Historical diagnostic only; current-universe and point-in-time limitations apply.</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        yy = y(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}" stroke="#D9E2E8"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="Segoe UI,Arial" font-size="12" fill="#52606D">{tick:.1f}x</text>'
        )
    dates = pd.DatetimeIndex(result.index)
    for year in range(dates.min().year, dates.max().year + 1):
        index = int(np.argmin(np.abs((dates - pd.Timestamp(year, 1, 1)).days)))
        parts.append(
            f'<text x="{x(index):.2f}" y="{top + plot_height + 31}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="12" fill="#52606D">{year}</text>'
        )
    for column, (label, color) in series.items():
        points = " ".join(
            f"{x(index):.2f},{y(float(value)):.2f}"
            for index, value in enumerate(result[column])
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        value = float(result[column].iloc[-1])
        yy = y(value)
        parts.append(
            f'<circle cx="{left + plot_width}" cy="{yy:.2f}" r="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + plot_width + 18}" y="{yy + 5:.2f}" font-family="Segoe UI,Arial" font-size="13" fill="{color}">{html.escape(label)} {value:.2f}x</text>'
        )
    parts.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#ADB8C2"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="13" fill="#52606D">A-share trading date</text>',
            '</svg>',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def write_report(payload: dict, output: Path) -> None:
    metrics = payload["metrics"]
    labels = {
        "selection_2020_2023": "Selection 2020-2023",
        "revealed_diagnostic_2024_latest": "Revealed diagnostic 2024-latest",
        "full_2020_latest": "Full 2020-latest",
    }
    lines = [
        "# High-Sharpe volatility-target LGBM candidate",
        "",
        "Historical research candidate only; production and paper trading are unchanged.",
        "",
        "| Window | Curve | CAGR | Volatility | Sharpe | Max drawdown |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        block = metrics[key]
        for curve_key, curve_label in (
            ("base_lgbm", "Base 75/25 LGBM"),
            ("volatility_target_lgbm", "Volatility-target LGBM"),
        ):
            values = block[curve_key]
            lines.append(
                f"| {label} | {curve_label} | {values['annual_return']:.2%} | "
                f"{values['annual_volatility']:.2%} | {values['sharpe_zero_rate']:.3f} | "
                f"{values['max_drawdown']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Frozen risk rule",
            "",
            "- 60-session volatility estimated only from returns through T-1.",
            "- 22% annualized volatility target; long exposure clipped to 65%-100%.",
            "- Residual capital stays in cash; leverage and shorting are disabled.",
            "- 12 bp charged on every aggregate exposure change.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen 75/25 LGBM volatility-target overlay."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--column", default="current_best_lgbm")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "data" / "lgbm_high_sharpe_overlay_v1.json",
    )
    parser.add_argument(
        "--output-curves",
        type=Path,
        default=ROOT / "reports" / "lgbm_high_sharpe_overlay_v1_curves.csv",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "reports" / "lgbm_high_sharpe_overlay_v1.md",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=ROOT / "reports" / "lgbm_high_sharpe_overlay_v1.svg",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.input, parse_dates=["date"]).set_index("date").sort_index()
    if args.column not in source:
        raise ValueError(f"missing strategy curve column: {args.column}")
    result, payload = evaluate(source[args.column])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_curves.parent.mkdir(parents=True, exist_ok=True)
    payload["artifacts"] = {
        "audit_json": relative_path(args.output_json),
        "curve_csv": relative_path(args.output_curves),
        "report": relative_path(args.output_report),
        "chart": relative_path(args.output_svg),
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.to_csv(args.output_curves, encoding="utf-8-sig", float_format="%.10f")
    write_report(payload, args.output_report)
    render_svg(result, args.output_svg)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
