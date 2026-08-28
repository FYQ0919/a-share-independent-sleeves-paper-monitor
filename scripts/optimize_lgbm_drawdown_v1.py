from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_ridge_topk_v1 import SOURCE_OPTIMIZATION


SOURCE_CURVE = BASE_DIR / "reports" / "lgbm_vs_nasdaq_2020_2026.csv"
OUTPUT_PATH = BASE_DIR / "data" / "lgbm_drawdown_optimization_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_drawdown_optimization_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_drawdown_optimized_curves.csv"
SELECTION_START = date(2020, 1, 1)
SELECTION_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2026, 8, 25)
COST_RATE = 12 / 10_000


CANDIDATES = [
    {"name": "vol32_floor65", "target_vol": 0.32, "floor": 0.65},
    {"name": "vol28_floor65", "target_vol": 0.28, "floor": 0.65},
    {"name": "vol28_floor50", "target_vol": 0.28, "floor": 0.50},
    {"name": "vol24_floor50", "target_vol": 0.24, "floor": 0.50},
    {"name": "vol60_t20_floor65", "vol_window": 60, "target_vol": 0.20, "floor": 0.65},
    {"name": "vol60_t20_floor70", "vol_window": 60, "target_vol": 0.20, "floor": 0.70},
    {"name": "vol60_t22_floor60", "vol_window": 60, "target_vol": 0.22, "floor": 0.60},
    {"name": "vol60_t22_floor65", "vol_window": 60, "target_vol": 0.22, "floor": 0.65},
    {"name": "vol60_t22_floor70", "vol_window": 60, "target_vol": 0.22, "floor": 0.70},
    {"name": "vol60_t24_floor65", "vol_window": 60, "target_vol": 0.24, "floor": 0.65},
    {"name": "trend120_cap70", "trend_cap": 0.70, "severe_trend_cap": 0.45},
    {
        "name": "trend120_hyst75_45",
        "trend_hysteresis": True,
        "trend_cap": 0.75,
        "severe_trend_cap": 0.45,
    },
    {
        "name": "trend120_hyst75_40",
        "trend_hysteresis": True,
        "trend_cap": 0.75,
        "severe_trend_cap": 0.40,
    },
    {
        "name": "trend120_hyst70_40",
        "trend_hysteresis": True,
        "trend_cap": 0.70,
        "severe_trend_cap": 0.40,
    },
    {
        "name": "trend120_hyst80_35",
        "trend_hysteresis": True,
        "trend_cap": 0.80,
        "severe_trend_cap": 0.35,
    },
    {
        "name": "vol32_floor65_trend70",
        "target_vol": 0.32,
        "floor": 0.65,
        "trend_cap": 0.70,
        "severe_trend_cap": 0.45,
    },
    {
        "name": "vol28_floor65_trend70",
        "target_vol": 0.28,
        "floor": 0.65,
        "trend_cap": 0.70,
        "severe_trend_cap": 0.45,
    },
    {
        "name": "vol32_floor65_trend_hyst75_40",
        "target_vol": 0.32,
        "floor": 0.65,
        "trend_hysteresis": True,
        "trend_cap": 0.75,
        "severe_trend_cap": 0.40,
    },
    {
        "name": "vol28_floor50_trend70",
        "target_vol": 0.28,
        "floor": 0.50,
        "trend_cap": 0.70,
        "severe_trend_cap": 0.45,
    },
    {"name": "dd15_cap50", "dd_trigger": 0.15, "dd_cap": 0.50, "cooldown": 10},
    {
        "name": "vol32_floor65_dd15",
        "target_vol": 0.32,
        "floor": 0.65,
        "dd_trigger": 0.15,
        "dd_cap": 0.50,
        "cooldown": 10,
    },
    {
        "name": "vol28_floor65_dd15",
        "target_vol": 0.28,
        "floor": 0.65,
        "dd_trigger": 0.15,
        "dd_cap": 0.50,
        "cooldown": 10,
    },
    {
        "name": "vol28_floor65_trend70_dd15",
        "target_vol": 0.28,
        "floor": 0.65,
        "trend_cap": 0.70,
        "severe_trend_cap": 0.45,
        "dd_trigger": 0.15,
        "dd_cap": 0.50,
        "cooldown": 10,
    },
]


def metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.fillna(0.0).astype(float)
    equity = (1 + clean).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1)
    days = max(1, len(clean))
    annual_return = float(equity.iloc[-1] ** (252 / days) - 1)
    volatility = float(clean.std(ddof=0) * np.sqrt(252))
    sharpe = float(clean.mean() * 252 / volatility) if volatility else 0.0
    max_drawdown = float(drawdown.min())
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
    }


def lagged_annualized_volatility(returns: pd.Series, window: int) -> pd.Series:
    return (
        returns.shift(1)
        .rolling(window, min_periods=max(10, window // 2))
        .std(ddof=0)
        * np.sqrt(252)
    )


def load_inputs() -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    trading_dates = pd.DatetimeIndex(sorted(history["date"].unique()))
    trading_dates = trading_dates[
        (trading_dates >= pd.Timestamp(SELECTION_START))
        & (trading_dates <= pd.Timestamp(TEST_END))
    ]

    comparison = pd.read_csv(SOURCE_CURVE, parse_dates=["date"]).set_index("date")
    comparison = comparison.reindex(trading_dates).ffill().dropna()
    comparison["base_return"] = comparison["lgbm"].pct_change().fillna(0.0)

    close = history.pivot(index="date", columns="code", values="close").reindex(comparison.index)
    stock_returns = close.pct_change(fill_method=None)
    pool_return = stock_returns.mean(axis=1, skipna=True).fillna(0.0)
    pool_equity = (1 + pool_return).cumprod()
    lagged_pool = pool_equity.shift(1)
    comparison["pool_trend_120"] = lagged_pool.div(
        lagged_pool.rolling(120, min_periods=60).mean()
    ).sub(1)
    for window in (20, 60):
        comparison[f"lagged_vol_{window}"] = lagged_annualized_volatility(
            comparison["base_return"], window
        )
    return comparison, pool_return, snapshot_id, warnings


def apply_overlay(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    equity = 1.0
    peak = 1.0
    previous_exposure = 0.0
    cooldown_remaining = 0
    trend_regime = "full"
    rows = []
    for when, row in frame.iterrows():
        exposure = 1.0
        target_vol = config.get("target_vol")
        vol_window = int(config.get("vol_window", 20))
        observed_vol_key = f"lagged_vol_{vol_window}"
        observed_vol = (
            float(row[observed_vol_key])
            if pd.notna(row[observed_vol_key])
            else None
        )
        if target_vol and observed_vol and observed_vol > 0:
            exposure = min(exposure, max(float(config.get("floor", 0.5)), target_vol / observed_vol))

        trend = float(row["pool_trend_120"]) if pd.notna(row["pool_trend_120"]) else 0.0
        if config.get("trend_hysteresis"):
            risk_entry = float(config.get("risk_entry", -0.01))
            risk_exit = float(config.get("risk_exit", 0.02))
            severe_entry = float(config.get("severe_entry", -0.08))
            severe_exit = float(config.get("severe_exit", -0.04))
            if trend_regime == "severe":
                if trend >= risk_exit:
                    trend_regime = "full"
                elif trend >= severe_exit:
                    trend_regime = "risk_off"
            elif trend_regime == "risk_off":
                if trend <= severe_entry:
                    trend_regime = "severe"
                elif trend >= risk_exit:
                    trend_regime = "full"
            elif trend <= severe_entry:
                trend_regime = "severe"
            elif trend <= risk_entry:
                trend_regime = "risk_off"

            if trend_regime == "risk_off":
                exposure = min(exposure, float(config["trend_cap"]))
            elif trend_regime == "severe":
                exposure = min(exposure, float(config["severe_trend_cap"]))
        else:
            if config.get("trend_cap") is not None and trend < 0:
                exposure = min(exposure, float(config["trend_cap"]))
            if config.get("severe_trend_cap") is not None and trend < -0.08:
                exposure = min(exposure, float(config["severe_trend_cap"]))
        if cooldown_remaining > 0:
            exposure = min(exposure, float(config.get("dd_cap", 1.0)))
            cooldown_remaining -= 1

        exposure_turnover = abs(exposure - previous_exposure)
        overlay_cost = exposure_turnover * COST_RATE
        managed_return = exposure * float(row["base_return"]) - overlay_cost
        equity *= 1 + managed_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if config.get("dd_trigger") and drawdown <= -float(config["dd_trigger"]):
            cooldown_remaining = max(cooldown_remaining, int(config.get("cooldown", 10)))

        rows.append({
            "date": when,
            "return": managed_return,
            "equity": equity,
            "drawdown": drawdown,
            "exposure": exposure,
            "exposure_turnover": exposure_turnover,
            "overlay_cost": overlay_cost,
            "trend_regime": trend_regime,
        })
        previous_exposure = exposure
    return pd.DataFrame(rows).set_index("date")


def period_metrics(curve: pd.DataFrame, start: date, end: date) -> dict[str, float]:
    selected = curve.loc[pd.Timestamp(start):pd.Timestamp(end)]
    output = metrics(selected["return"])
    output.update({
        "average_exposure": float(selected["exposure"].mean()),
        "exposure_turnover": float(selected["exposure_turnover"].sum()),
        "overlay_cost": float(selected["overlay_cost"].sum()),
    })
    return output


def selection_rank(row: dict) -> tuple:
    selected = row["selection"]
    return (
        selected["calmar"],
        selected["sharpe"],
        selected["annual_return"],
        -selected["overlay_cost"],
    )


def main() -> None:
    frame, _, snapshot_id, warnings = load_inputs()
    baseline_curve = pd.DataFrame(index=frame.index)
    baseline_curve["return"] = frame["base_return"]
    baseline_curve["equity"] = (1 + baseline_curve["return"]).cumprod()
    baseline_curve["drawdown"] = baseline_curve["equity"].div(
        baseline_curve["equity"].cummax()
    ).sub(1)
    baseline_curve["exposure"] = 1.0
    baseline_curve["exposure_turnover"] = 0.0
    baseline_curve["overlay_cost"] = 0.0
    baseline_selection = period_metrics(baseline_curve, SELECTION_START, SELECTION_END)
    baseline_test = period_metrics(baseline_curve, TEST_START, TEST_END)

    candidates = []
    curves = {}
    for config in CANDIDATES:
        curve = apply_overlay(frame, config)
        curves[config["name"]] = curve
        selection = period_metrics(curve, SELECTION_START, SELECTION_END)
        eligible = (
            abs(selection["max_drawdown"]) <= abs(baseline_selection["max_drawdown"]) * 0.85
            and selection["annual_return"] >= baseline_selection["annual_return"] * 0.75
            and selection["sharpe"] >= baseline_selection["sharpe"]
        )
        candidates.append({
            "name": config["name"],
            "config": config,
            "selection": selection,
            "selection_gate_passed": eligible,
        })
    eligible_rows = [item for item in candidates if item["selection_gate_passed"]]
    winner = max(eligible_rows, key=selection_rank) if eligible_rows else max(candidates, key=selection_rank)
    winner_curve = curves[winner["name"]]
    winner_test = period_metrics(winner_curve, TEST_START, TEST_END)
    test_checks = {
        "drawdown_improved_by_15pct": (
            abs(winner_test["max_drawdown"]) <= abs(baseline_test["max_drawdown"]) * 0.85
        ),
        "annual_return_retained_80pct": (
            winner_test["annual_return"] >= baseline_test["annual_return"] * 0.80
        ),
        "sharpe_not_worse": winner_test["sharpe"] >= baseline_test["sharpe"],
    }
    promotion_passed = bool(eligible_rows) and all(test_checks.values())

    exported = pd.DataFrame(index=frame.index)
    exported["lgbm_full_exposure"] = baseline_curve["equity"]
    exported["lgbm_drawdown_control"] = winner_curve["equity"]
    exported["risk_exposure"] = winner_curve["exposure"]
    exported["full_exposure_drawdown"] = baseline_curve["drawdown"]
    exported["controlled_drawdown"] = winner_curve["drawdown"]
    exported.index.name = "date"
    exported.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    candidates.sort(key=selection_rank, reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted" if not promotion_passed else "candidate_passed_not_deployed",
        "objective": "reduce LGBM drawdown while retaining annual return and Sharpe",
        "windows": {
            "selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_test_diagnostic": [TEST_START.isoformat(), TEST_END.isoformat()],
        },
        "execution": "causal exposure decided from prior observations; cash only; no leverage; 12bp charged on exposure changes",
        "selection_rule": "select only on 2020-2023; require 15% relative drawdown reduction, at least 75% annual-return retention, and non-worse Sharpe",
        "universe_snapshot_run_id": snapshot_id,
        "baseline": {"selection": baseline_selection, "test": baseline_test},
        "winner": {
            **winner,
            "test": winner_test,
            "test_checks": test_checks,
        },
        "promotion_passed": promotion_passed,
        "production_change": False,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "candidates": candidates,
        "warnings": warnings + [
            "2024-2026 was already revealed in prior research and is diagnostic only.",
            "The current Top50 universe is backfilled and has survivorship bias.",
            "The overlay is evaluated on the stored LGBM return stream rather than a broker fill replay.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        "\n".join([
            "# LGBM 回撤控制研究 v1",
            "",
            f"- 状态：{'候选通过但未部署' if promotion_passed else '未通过晋级门禁'}",
            f"- 选择窗口：{SELECTION_START} 至 {SELECTION_END}",
            f"- 揭示验证：{TEST_START} 至 {TEST_END}",
            f"- 胜出候选：`{winner['name']}`",
            f"- 选择期最大回撤：{baseline_selection['max_drawdown']:.2%} -> {winner['selection']['max_drawdown']:.2%}",
            f"- 选择期年化：{baseline_selection['annual_return']:.2%} -> {winner['selection']['annual_return']:.2%}",
            f"- 验证期最大回撤：{baseline_test['max_drawdown']:.2%} -> {winner_test['max_drawdown']:.2%}",
            f"- 验证期年化：{baseline_test['annual_return']:.2%} -> {winner_test['annual_return']:.2%}",
            f"- 验证期夏普：{baseline_test['sharpe']:.3f} -> {winner_test['sharpe']:.3f}",
            f"- 平均风险仓位：{winner_test['average_exposure']:.1%}",
            "- 生产修改：否",
            "",
            "所有仓位信号至少滞后一日；未使用2024-2026重新排序候选。当前结果仍受历史股票池和复权时点偏差影响。",
        ]),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "curves": str(CURVE_PATH),
        "baseline_selection": baseline_selection,
        "winner": payload["winner"],
        "promotion_passed": promotion_passed,
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
