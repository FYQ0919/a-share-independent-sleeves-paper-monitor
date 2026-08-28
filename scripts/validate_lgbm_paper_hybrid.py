from __future__ import annotations

"""Controlled hybrid validation: frozen LightGBM selection plus paper overlays."""

from datetime import date, datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

from app.backtest_engine import BacktestEngine
from app.config import BASE_DIR as APP_BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    fit_expanding_lgbm_predictions,
    topk_dropout_backtest,
)


START = date(2023, 1, 1)
END = date(2025, 8, 31)
REBALANCE_DAYS = 10
TOP_K = 5
COST_BPS = 12.0
OUTPUT_PATH = APP_BASE_DIR / "data" / "lgbm_paper_hybrid_validation.json"
REPORT_PATH = APP_BASE_DIR / "reports" / "lgbm_paper_hybrid_validation.md"
CURVE_PATH = APP_BASE_DIR / "reports" / "lgbm_paper_hybrid_curves.csv"


def _history_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    return frame.sort_values(["code", "date"]).drop_duplicates(["code", "date"])


def paper_risk_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = _history_frame(history)
    grouped = frame.groupby("code", group_keys=False)
    frame["ret_1"] = grouped["close"].pct_change(fill_method=None)
    frame["volatility_60"] = grouped["ret_1"].transform(
        lambda s: s.rolling(60, min_periods=60).std(ddof=0)
    )
    frame["sharpe_60"] = grouped["ret_1"].transform(
        lambda s: s.rolling(60, min_periods=60).mean()
    ).div(frame["volatility_60"].replace(0, np.nan)).mul(np.sqrt(252.0))
    return frame[["date", "code", "volatility_60", "sharpe_60"]]


def market_stress_gate(history: pd.DataFrame, mode: str = "adaptive") -> pd.Series:
    """As-of market regime gate reused from the project's risk proxy.

    The gate requires both a trend break and elevated realized volatility.  It
    is deliberately independent of LightGBM predictions and uses only closes
    available on the signal date.
    """
    data = _history_frame(history)
    close = data.pivot(index="date", columns="code", values="close").sort_index()
    proxy = close.mean(axis=1, skipna=True)
    daily = proxy.pct_change(fill_method=None)
    trend = proxy.div(proxy.rolling(120, min_periods=60).mean()).sub(1.0)
    volatility = daily.rolling(20, min_periods=10).std().mul(np.sqrt(252.0))
    if mode == "adaptive":
        return ((trend < -0.03) & (volatility > 0.30)).fillna(False)
    if mode == "strong":
        return ((trend < -0.08) & (volatility > 0.35)).fillna(False)
    if mode == "always":
        return pd.Series(True, index=close.index)
    raise ValueError(f"unknown risk gate mode: {mode}")


def _risk_budget(drawdown: float, cooldown: int) -> tuple[float, int, str]:
    if cooldown > 0:
        return 0.0, cooldown - 1, "cooldown"
    if drawdown <= -0.06:
        return 0.0, 1, "dd_over_6pct"
    if drawdown <= -0.04:
        return 0.40, 0, "dd_4_to_6pct"
    if drawdown <= -0.02:
        return 0.80, 0, "dd_2_to_4pct"
    return 1.0, 0, "normal"


def _weights(
    holdings: list[str],
    signal_date: pd.Timestamp,
    risk_features: pd.DataFrame,
    score_lookup: pd.Series,
    mode: str,
) -> pd.Series:
    selected = risk_features[
        (risk_features["date"] == signal_date)
        & risk_features["code"].isin(holdings)
    ].set_index("code").reindex(holdings)
    vol = selected["volatility_60"].astype(float).replace(0, np.nan)
    iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if iv.sum() <= 0:
        iv = pd.Series(1.0, index=holdings)
    iv = iv / iv.sum()
    if mode == "equal":
        return pd.Series(1.0 / len(holdings), index=holdings)
    if mode == "inverse_vol":
        return iv
    sharpe = selected["sharpe_60"].clip(lower=0.0).fillna(0.0)
    if sharpe.sum() <= 0:
        sharpe = pd.Series(1.0, index=holdings)
    sharpe = sharpe / sharpe.sum()
    if mode == "paper":
        return 0.5 * iv + 0.5 * sharpe
    if mode == "lgbm_iv_blend":
        scores = score_lookup.reindex(holdings).astype(float).clip(lower=0.0).fillna(0.0)
        if scores.sum() <= 0:
            scores = pd.Series(1.0, index=holdings)
        scores = scores / scores.sum()
        return 0.5 * iv + 0.5 * scores
    raise ValueError(f"unknown weight mode: {mode}")


def _select_holdings(
    cross: pd.Series,
    planned: list[str],
    top_k: int,
) -> tuple[list[str], list[str]]:
    cross = cross.dropna().sort_values(ascending=False)
    if len(cross) < top_k:
        return [], []
    if not planned:
        return cross.head(top_k).index.astype(str).tolist(), []
    available = [code for code in planned if code in cross.index]
    missing = [code for code in planned if code not in cross.index]
    ranked_holdings = sorted(available, key=lambda code: float(cross.loc[code]))
    challengers = [code for code in cross.index.astype(str) if code not in planned]
    drop_count = min(1, len(ranked_holdings) + len(missing), len(challengers))
    dropped = missing[:drop_count]
    dropped.extend(ranked_holdings[: drop_count - len(dropped)])
    target = [code for code in planned if code not in dropped]
    target.extend(challengers[:drop_count])
    if len(target) < top_k:
        target.extend(code for code in cross.index.astype(str) if code not in target)
    return target[:top_k], dropped


def simulate_hybrid(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    risk_features: pd.DataFrame,
    weight_mode: str,
    risk_overlay: bool,
    risk_gate: pd.Series | None = None,
) -> dict:
    data = _history_frame(history)
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))]
    opens = data.pivot(index="date", columns="code", values="open").reindex(dates)
    columns = opens.columns
    score_lookup = scores.set_index(["date", "code"])["score"]
    current = pd.Series(0.0, index=columns)
    pending: dict[int, pd.Series] = {}
    planned: list[str] = []
    planned_raw = pd.Series(0.0, index=columns)
    desired_exposure = 0.0
    equity = 1.0
    peak = 1.0
    cooldown = 0
    periods = []
    exposure_history = []
    gate_history = []
    halt_count = 0

    for i in range(len(dates) - 1):
        signal_date = dates[i]
        turnover = 0.0
        if i in pending:
            target = pending.pop(i).reindex(columns).fillna(0.0)
            turnover = float(target.sub(current).abs().sum())
            current = target

        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        gate_active = bool(risk_gate.loc[signal_date]) if risk_overlay and risk_gate is not None and signal_date in risk_gate.index else risk_overlay
        if risk_overlay and gate_active:
            exposure, cooldown, risk_state = _risk_budget(drawdown, cooldown)
            if risk_state == "dd_over_6pct":
                peak = equity
            if exposure == 0.0 and risk_state != "normal":
                halt_count += 1
        else:
            exposure, risk_state = 1.0, "normal"
            if risk_overlay:
                risk_state = "gate_off"
                cooldown = 0
        gate_history.append(bool(gate_active) if risk_overlay else False)

        try:
            cross = scores[scores["date"] == signal_date].set_index("code")["score"]
        except KeyError:
            cross = pd.Series(dtype=float)
        scheduled = i % REBALANCE_DAYS == 0
        if scheduled:
            selected, _ = _select_holdings(cross, planned, TOP_K)
            if selected:
                planned = selected
                planned_raw = pd.Series(0.0, index=columns)
                raw = _weights(planned, signal_date, risk_features, cross, weight_mode)
                planned_raw.loc[raw.index] = raw

        if planned:
            target = planned_raw.mul(exposure)
            # Keep the current LGBM hold-and-drift behavior between main
            # rebalances.  The paper overlay only trades when its exposure
            # budget changes; otherwise it must not become a daily rebalance.
            exposure_changed = abs(exposure - desired_exposure) > 1e-12
            if scheduled or (risk_overlay and exposure_changed):
                pending[i + 1] = target
                desired_exposure = exposure

        returns = opens.iloc[i + 1].div(opens.iloc[i]).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = float(current.mul(returns).sum())
        cost = turnover * COST_BPS / 10_000.0
        net = gross - cost
        exposure_history.append(float(current.sum()))
        periods.append({
            "period_start": signal_date.strftime("%Y-%m-%d"),
            "period_end": dates[i + 1].strftime("%Y-%m-%d"),
            "strategy_return": net,
            "benchmark_return": float(returns.mean()),
            "turnover": turnover,
            "cost": cost,
            "exposure": float(current.sum()),
            "risk_state": risk_state,
        })
        equity *= max(0.0, 1.0 + net)
        peak = max(peak, equity)
        if current.sum() > 0 and 1.0 + gross > 0:
            current = current.mul(1.0 + returns).div(1.0 + gross)

    period_frame = pd.DataFrame(periods)
    strategy = pd.Series(period_frame["strategy_return"].to_numpy(), index=period_frame.index)
    benchmark = pd.Series(period_frame["benchmark_return"].to_numpy(), index=period_frame.index)
    equity_curve = (1.0 + strategy).cumprod()
    benchmark_curve = (1.0 + benchmark).cumprod()
    metrics = BacktestEngine._metrics(
        strategy, benchmark, equity_curve, benchmark_curve,
        period_frame["turnover"], period_frame["cost"],
    )
    years = max((dates[-1] - dates[0]).days / 365.2425, 1 / 365.2425)
    metrics.update({
        "annual_turnover": float(period_frame["turnover"].sum() / years),
        "total_turnover": float(period_frame["turnover"].sum()),
        "average_active_exposure": float(np.mean(exposure_history)) if exposure_history else 0.0,
        "risk_halt_count": int(halt_count),
        "cooldown_count": int((period_frame["risk_state"] == "cooldown").sum()),
        "gate_active_days": int(sum(gate_history)),
        "gate_active_fraction": float(np.mean(gate_history)) if gate_history else 0.0,
    })
    return {
        "metrics": metrics,
        "return_periods": periods,
        "weight_mode": weight_mode,
        "risk_overlay": risk_overlay,
        "risk_gate": "always" if risk_gate is None else "timed",
    }


def curve(result: dict, name: str) -> pd.Series:
    periods = pd.DataFrame(result["return_periods"])
    periods["period_start"] = pd.to_datetime(periods["period_start"])
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    values = (1.0 + periods.set_index("period_end")["strategy_return"].astype(float)).cumprod()
    return pd.concat([pd.Series([1.0], index=[periods["period_start"].iloc[0]]), values]).rename(name)


def yearly(result: dict) -> dict:
    periods = pd.DataFrame(result["return_periods"])
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    return {
        str(year): float((1.0 + group["strategy_return"].astype(float)).prod() - 1.0)
        for year, group in periods.groupby(periods["period_end"].dt.year)
    }


def main() -> None:
    source = json.loads((APP_BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history = _history_frame(history)
    history = history[history["date"] <= pd.Timestamp(END)].copy()
    frame, feature_columns = build_alpha158_lite(history)
    baseline_prediction = baseline_scores(frame, START, END)
    prediction, audit = fit_expanding_lgbm_predictions(frame, feature_columns, WINNER_CONFIG, START, END)
    lgbm_scores = blend_scores(frame, prediction, baseline_prediction, WINNER_CONFIG["baseline_weight"])
    risk_features = paper_risk_features(history)
    adaptive_gate = market_stress_gate(history, "adaptive")
    strong_gate = market_stress_gate(history, "strong")

    cases = {
        "lgbm_current_equal_no_risk": ("equal", False),
        "lgbm_inverse_vol": ("inverse_vol", False),
        "lgbm_paper_weight": ("paper", False),
        "lgbm_paper_risk_overlay": ("equal", True),
        "lgbm_full_hybrid": ("paper", True),
        "lgbm_score_iv_risk": ("lgbm_iv_blend", True),
    }
    results = {}
    for name, (weight_mode, risk_overlay) in cases.items():
        results[name] = simulate_hybrid(
            history, lgbm_scores, risk_features, weight_mode, risk_overlay,
            risk_gate=None,
        )
    timed_cases = {
        "lgbm_timed_risk_adaptive": ("equal", adaptive_gate),
        "lgbm_timed_full_adaptive": ("paper", adaptive_gate),
        "lgbm_timed_score_iv_adaptive": ("lgbm_iv_blend", adaptive_gate),
        "lgbm_timed_risk_strong": ("equal", strong_gate),
    }
    for name, (weight_mode, gate) in timed_cases.items():
        results[name] = simulate_hybrid(
            history, lgbm_scores, risk_features, weight_mode, True,
            risk_gate=gate,
        )

    # Independent check that the custom equal-weight simulator agrees with the
    # established TopK/drop-one implementation before interpreting overlays.
    reference = topk_dropout_backtest(history, lgbm_scores, START, END, cost_bps=COST_BPS)
    custom = results["lgbm_current_equal_no_risk"]["metrics"]
    reference_metrics = reference["metrics"]
    parity = {
        "annual_return_abs_diff": abs(custom["annual_return"] - reference_metrics["annual_return"]),
        "sharpe_abs_diff": abs(custom["sharpe"] - reference_metrics["sharpe"]),
        "total_return_abs_diff": abs(custom["total_return"] - reference_metrics["total_return"]),
        "passed": abs(custom["total_return"] - reference_metrics["total_return"]) < 1e-8,
    }

    curve_frame = pd.concat({name: curve(result, name) for name, result in results.items()}, axis=1, join="outer").sort_index().ffill()
    curve_frame = curve_frame.div(curve_frame.iloc[0]).ffill()
    curve_frame.index.name = "date"
    curve_frame.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    summaries = {
        name: {"metrics": result["metrics"], "yearly_returns": yearly(result)}
        for name, result in results.items()
    }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "window": {"start": START.isoformat(), "end": END.isoformat()},
        "universe_snapshot_run_id": snapshot_id,
        "requested_codes": len(requested_codes),
        "loaded_codes": int(history["code"].nunique()),
        "execution_contract": "signal close; next-open execution; Top5; 10-session main rebalance; 12bp one-way cost",
        "lgbm": {"model_version": "lgbm_ranker_top5_v1", "config": WINNER_CONFIG, "feature_count": len(feature_columns), "audit": audit},
        "paper_overlays": {
            "inverse_volatility": "rolling 60-session inverse volatility inside LightGBM Top5",
            "paper_weight": "50% inverse volatility + 50% nonnegative rolling Sharpe inside LightGBM Top5",
            "risk_overlay": "80% / 40% / 0% exposure at 2% / 4% / 6% drawdown; one-session cooldown; high-water reset after forced exit",
            "timed_gate_adaptive": "pre-existing market proxy: 120-session trend < -3% AND 20-session annualized volatility > 30%",
            "timed_gate_strong": "pre-existing market proxy: 120-session trend < -8% AND 20-session annualized volatility > 35%",
        },
        "comparisons": summaries,
        "reference_parity": parity,
        "curve_data": str(CURVE_PATH.relative_to(APP_BASE_DIR)),
        "warnings": warnings + [
            "这是冻结Top50回填上的研究型对比，不是严格point-in-time收益",
            "混合实验只改权重和风险层，LightGBM模型、Top5和10日调仓保持不变",
            "所有论文参数在本实验中预先固定，未以25%收益为目标调参",
            "结果未修改生产LGBM或模拟盘配置",
            "择时门控只能降低风控触发频率，不能保证任何未来年化收益；门控阈值未在本窗口内优化",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{float(x):.2%}"

    report = [
        "# LightGBM + 论文方法混合验证",
        "",
        f"- 区间：{START.isoformat()} 至 {END.isoformat()}；股票池请求 {len(requested_codes)} 只，实际加载 {history['code'].nunique()} 只。",
        "- 固定协议：LightGBM Top5、每10个交易日主调仓、收盘信号、下一交易日开盘执行、单边12bp。",
        "- 混合层只改组合权重和论文回撤暴露，不改LightGBM模型与选股分数。",
        "",
        "| 方案 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 年化换手 | 平均暴露 | 总成本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "lgbm_current_equal_no_risk": "当前LGBM等权",
        "lgbm_inverse_vol": "LGBM + 逆波动率",
        "lgbm_paper_weight": "LGBM + 论文50/50权重",
        "lgbm_paper_risk_overlay": "LGBM + 论文回撤风控",
        "lgbm_full_hybrid": "LGBM + 论文权重 + 风控",
        "lgbm_score_iv_risk": "LGBM分数/逆波动率 + 风控",
        "lgbm_timed_risk_adaptive": "LGBM + 中等压力择时风控",
        "lgbm_timed_full_adaptive": "LGBM论文权重 + 中等压力择时风控",
        "lgbm_timed_score_iv_adaptive": "LGBM分数/逆波动率 + 中等压力择时风控",
        "lgbm_timed_risk_strong": "LGBM + 强压力择时风控",
    }
    for name, label in labels.items():
        metrics = summaries[name]["metrics"]
        report.append(
            f"| {label} | {pct(metrics['annual_return'])} | {pct(metrics['annual_volatility'])} | {metrics['sharpe']:.3f} | {pct(metrics['max_drawdown'])} | {metrics['annual_turnover']:.2f}x | {pct(metrics['average_active_exposure'])} | {pct(metrics['total_cost'])} |"
        )
    base = summaries["lgbm_current_equal_no_risk"]["metrics"]
    full = summaries["lgbm_full_hybrid"]["metrics"]
    report.extend([
        "",
        f"- 完整混合相对当前LGBM：年化收益差 {pct(full['annual_return'] - base['annual_return'])}，Sharpe差 {full['sharpe'] - base['sharpe']:.3f}，最大回撤变化 {pct(full['max_drawdown'] - base['max_drawdown'])}。",
        f"- 自定义回测与既有TopK/drop-one模拟器总收益绝对差：{parity['total_return_abs_diff']:.2e}，{'通过' if parity['passed'] else '未通过'}。",
        "",
        "## 解释边界",
        "",
        "本实验只说明论文组合层是否能改善当前LightGBM的风险收益，不证明论文声称的25%年化收益。股票池仍是当前Top50历史回填，论文缺失的市值、bid-ask spread和完整GP alpha模块也没有补造。",
        "",
        f"- 结果 JSON：`{OUTPUT_PATH.relative_to(APP_BASE_DIR)}`",
        f"- 曲线 CSV：`{CURVE_PATH.relative_to(APP_BASE_DIR)}`",
    ])
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "report": str(REPORT_PATH), "curve": str(CURVE_PATH), "comparisons": summaries, "reference_parity": parity}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
