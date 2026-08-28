from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from optimize_lgbm_drawdown_v1 import apply_overlay, load_inputs, metrics


OUTPUT_PATH = BASE_DIR / "data" / "lgbm_growth_control_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_growth_control_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_growth_control_v1_curves.csv"
OVERLAY_WEIGHT = 0.25
OVERLAY_CONFIG = {
    "name": "vol32_floor65",
    "target_vol": 0.32,
    "floor": 0.65,
}


def blend_growth_control(
    base_returns: pd.Series,
    overlay_curve: pd.DataFrame,
    overlay_weight: float = OVERLAY_WEIGHT,
) -> pd.DataFrame:
    if not 0 <= overlay_weight <= 1:
        raise ValueError("覆盖层权重必须在0和1之间")
    aligned = overlay_curve.reindex(base_returns.index)
    if aligned[["return", "exposure"]].isna().any().any():
        raise ValueError("覆盖层与基础收益日期未对齐")
    output = pd.DataFrame(index=base_returns.index)
    output["base_return"] = base_returns.astype(float)
    output["overlay_return"] = aligned["return"].astype(float)
    output["strategy_return"] = (
        output["base_return"].mul(1 - overlay_weight)
        + output["overlay_return"].mul(overlay_weight)
    )
    output["exposure"] = 1 - overlay_weight + overlay_weight * aligned["exposure"]
    output["equity"] = (1 + output["strategy_return"]).cumprod()
    output["drawdown"] = output["equity"].div(output["equity"].cummax()).sub(1)
    return output


def main() -> None:
    frame, _, snapshot_id, warnings = load_inputs()
    overlay_curve = apply_overlay(frame, OVERLAY_CONFIG)
    growth = blend_growth_control(frame["base_return"], overlay_curve)
    baseline_metrics = metrics(frame["base_return"])
    growth_metrics = metrics(growth["strategy_return"])
    growth_metrics.update({
        "average_exposure": float(growth["exposure"].mean()),
        "minimum_exposure": float(growth["exposure"].min()),
        "annual_return_retention": growth_metrics["annual_return"]
        / baseline_metrics["annual_return"],
        "relative_drawdown_improvement": 1
        - abs(growth_metrics["max_drawdown"]) / abs(baseline_metrics["max_drawdown"]),
    })
    checks = {
        "annual_return_retained_97_5pct": growth_metrics["annual_return"]
        >= baseline_metrics["annual_return"] * 0.975,
        "maximum_drawdown_improved": growth_metrics["max_drawdown"]
        >= baseline_metrics["max_drawdown"],
        "sharpe_not_lower": growth_metrics["sharpe"] >= baseline_metrics["sharpe"],
        "average_exposure_at_least_98pct": growth_metrics["average_exposure"] >= 0.98,
    }
    passed = all(checks.values())

    exported = pd.DataFrame(index=frame.index)
    exported["current_lgbm"] = (1 + frame["base_return"]).cumprod()
    exported["growth_control_lgbm"] = growth["equity"]
    exported["growth_control_exposure"] = growth["exposure"]
    exported["current_lgbm_drawdown"] = exported["current_lgbm"].div(
        exported["current_lgbm"].cummax()
    ).sub(1)
    exported["growth_control_drawdown"] = growth["drawdown"]
    exported.index.name = "date"
    exported.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "historical_gate_passed_not_deployed" if passed else "research_only_rejected",
        "strategy": "75% expanding LGBM Top5 + 25% causal 32% volatility-target sleeve",
        "objective": "maximize annual-return retention while slightly reducing drawdown",
        "window": [frame.index.min().date().isoformat(), frame.index.max().date().isoformat()],
        "overlay_weight": OVERLAY_WEIGHT,
        "overlay_config": OVERLAY_CONFIG,
        "execution": "base strategy already includes next-open fills and 12bp costs; overlay uses lagged volatility and charges 12bp on exposure changes",
        "baseline": baseline_metrics,
        "growth_control": growth_metrics,
        "historical_gate": {"passed": passed, "checks": checks},
        "universe_snapshot_run_id": snapshot_id,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "production_change": False,
        "forward_paper_required": True,
        "warnings": warnings + [
            "2020-2026 has already been revealed and cannot serve as a fresh promotion test.",
            "The overlay is evaluated on the stored expanding-LGBM return stream, not a broker fill replay.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        "\n".join([
            "# LGBM 年化优先回撤控制 v1",
            "",
            f"- 状态：{'历史门禁通过，未部署' if passed else '历史门禁未通过'}",
            f"- 区间：{payload['window'][0]} 至 {payload['window'][1]}",
            f"- 当前LGBM：年化 {baseline_metrics['annual_return']:.2%}，回撤 {baseline_metrics['max_drawdown']:.2%}，夏普 {baseline_metrics['sharpe']:.3f}",
            f"- 年化优先版：年化 {growth_metrics['annual_return']:.2%}，回撤 {growth_metrics['max_drawdown']:.2%}，夏普 {growth_metrics['sharpe']:.3f}",
            f"- 年化保留：{growth_metrics['annual_return_retention']:.2%}",
            f"- 平均股票仓位：{growth_metrics['average_exposure']:.2%}",
            "- 生产修改：否",
            "",
            "该配置只把25%组合交给温和波动覆盖层，75%始终跟随原LGBM。2020-2026已揭盲，必须进入新的前向模拟盘后才能晋级。",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
