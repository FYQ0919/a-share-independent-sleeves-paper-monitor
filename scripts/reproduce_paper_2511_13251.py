from __future__ import annotations

"""Reproduce the executable parts of arXiv:2511.13251v1.

The paper leaves several universe-filter values unspecified.  This script keeps
those values explicit, runs both the paper-native close-to-close protocol and
the project's common next-open protocol, and compares the latter with the
frozen LightGBM ranker.  It is research-only and never changes production
configuration.
"""

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
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    fit_expanding_lgbm_predictions,
    research_summary,
    topk_dropout_backtest,
)
from app.lgbm_strategy import WINNER_CONFIG


PAPER_START = date(2023, 1, 1)
PAPER_END = date(2025, 8, 31)
ROLLING_WINDOW = 60
TOP_QUANTILE = 0.20
MIN_AVG_AMOUNT = 10_000_000.0
PAPER_COST_BPS = 5.0
UNIFIED_COST_BPS = 12.0
OUTPUT_PATH = APP_BASE_DIR / "data" / "paper_2511_13251_reproduction.json"
REPORT_PATH = APP_BASE_DIR / "reports" / "paper_2511_13251_reproduction.md"
CURVE_PATH = APP_BASE_DIR / "reports" / "paper_2511_13251_comparison_curve.csv"


def _clean_history(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    return frame.sort_values(["code", "date"]).drop_duplicates(["code", "date"])


def build_paper_signals(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build as-of Sharpe ranks and 50/50 inverse-vol/Sharpe weights."""
    frame = _clean_history(history)
    grouped = frame.groupby("code", group_keys=False)
    frame["ret_1"] = grouped["close"].pct_change(fill_method=None)
    frame["rolling_mean_return"] = grouped["ret_1"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    )
    frame["rolling_volatility"] = grouped["ret_1"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std(ddof=0)
    )
    frame["rolling_sharpe"] = frame["rolling_mean_return"].div(
        frame["rolling_volatility"].replace(0, np.nan)
    ).mul(np.sqrt(252.0))
    frame["amount_20"] = grouped["amount"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    frame["eligible"] = (
        frame["tradestatus"].eq(1)
        & frame["is_st"].eq(0)
        & frame["close"].gt(1)
        & frame["amount_20"].ge(MIN_AVG_AMOUNT)
        & frame["rolling_sharpe"].notna()
        & frame["rolling_volatility"].gt(0)
    )

    signal_rows = []
    for signal_date, cross in frame.groupby("date", sort=True):
        if signal_date < pd.Timestamp(PAPER_START) or signal_date > pd.Timestamp(PAPER_END):
            continue
        eligible = cross[cross["eligible"]].copy()
        if eligible.empty:
            signal_rows.append({"date": signal_date, "holdings": [], "weights": {}})
            continue
        n_select = max(1, int(np.ceil(len(eligible) * TOP_QUANTILE)))
        selected = eligible.nlargest(n_select, "rolling_sharpe").copy()
        inv_vol = 1.0 / selected["rolling_volatility"]
        iv_weight = inv_vol / inv_vol.sum()
        sharpe_positive = selected["rolling_sharpe"].clip(lower=0.0)
        if sharpe_positive.sum() > 0:
            sharpe_weight = sharpe_positive / sharpe_positive.sum()
        else:
            sharpe_weight = pd.Series(1.0 / len(selected), index=selected.index)
        weights = (0.5 * iv_weight + 0.5 * sharpe_weight).to_numpy(dtype=float)
        signal_rows.append({
            "date": signal_date,
            "holdings": selected["code"].astype(str).tolist(),
            "weights": dict(zip(selected["code"].astype(str), weights)),
            "selected_count": int(len(selected)),
            "eligible_count": int(len(eligible)),
        })
    signals = pd.DataFrame(signal_rows).set_index("date")
    audit = {
        "rolling_window_sessions": ROLLING_WINDOW,
        "top_quantile": TOP_QUANTILE,
        "min_average_amount": MIN_AVG_AMOUNT,
        "market_cap_filter": "unavailable: no point-in-time market-cap field in current cache",
        "bid_ask_spread_filter": "unavailable: daily cache has no bid/ask quotes; amount proxy used",
        "signal_dates": int(len(signals)),
        "nonempty_signal_dates": int(sum(bool(x) for x in signals["holdings"])),
        "mean_selected_count": float(
            pd.to_numeric(signals.get("selected_count", pd.Series(dtype=float)), errors="coerce").mean()
        ) if "selected_count" in signals else 0.0,
    }
    return signals, audit


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


def simulate_paper_strategy(
    history: pd.DataFrame,
    signals: pd.DataFrame,
    execution: str,
    cost_bps: float,
) -> dict:
    """Sequentially simulate close-to-close or next-open execution.

    Signals use only data through the signal close.  Next-open targets are
    scheduled one session later; close targets take effect for the following
    close-to-close interval.  Portfolio weights drift with realized returns.
    """
    if execution not in {"close_to_close", "next_open"}:
        raise ValueError("execution must be close_to_close or next_open")
    data = _clean_history(history)
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(PAPER_START)) & (dates <= pd.Timestamp(PAPER_END))]
    close = data.pivot(index="date", columns="code", values="close").reindex(dates)
    open_prices = data.pivot(index="date", columns="code", values="open").reindex(dates)
    columns = close.columns
    periods = []
    current = pd.Series(0.0, index=columns)
    equity = 1.0
    peak = 1.0
    cooldown = 0
    pending = {}
    halt_count = 0
    exposure_history = []

    def apply_target(target: pd.Series, turnover_date: pd.Timestamp) -> float:
        nonlocal current
        target = target.reindex(columns).fillna(0.0)
        turnover = float(target.sub(current).abs().sum())
        current = target
        return turnover

    for i in range(len(dates) - 1):
        current_date = dates[i]
        # A next-open target generated at yesterday's close becomes active now.
        turnover = 0.0
        if execution == "next_open" and i in pending:
            turnover = apply_target(pending.pop(i), current_date)

        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        exposure, cooldown, risk_state = _risk_budget(drawdown, cooldown)
        if exposure == 0.0 and risk_state != "normal":
            halt_count += 1
        if risk_state == "dd_over_6pct":
            # The paper specifies a one-day cooldown.  Resetting the internal
            # high-water mark here prevents a forced exit from becoming a
            # permanent cash lock while retaining the one-session pause.
            peak = equity
        row = signals.loc[current_date] if current_date in signals.index else None
        raw_weights = row["weights"] if row is not None and isinstance(row["weights"], dict) else {}
        target = pd.Series(raw_weights, dtype=float).reindex(columns).fillna(0.0).mul(exposure)
        if execution == "close_to_close":
            turnover += apply_target(target, current_date)
        else:
            pending[i + 1] = target

        prices = close if execution == "close_to_close" else open_prices
        asset_returns = prices.iloc[i + 1].div(prices.iloc[i]).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = float(current.mul(asset_returns).sum())
        cost = turnover * cost_bps / 10_000.0
        net = gross - cost
        exposure_history.append(float(current.sum()))
        periods.append({
            "period_start": current_date.strftime("%Y-%m-%d"),
            "period_end": dates[i + 1].strftime("%Y-%m-%d"),
            "strategy_return": net,
            "benchmark_return": float(asset_returns.mean()),
            "turnover": turnover,
            "cost": cost,
            "exposure": float(current.sum()),
            "risk_state": risk_state,
        })
        equity *= max(0.0, 1.0 + net)
        peak = max(peak, equity)
        if current.sum() > 0 and 1.0 + gross > 0:
            current = current.mul(1.0 + asset_returns).div(1.0 + gross)

    returns = pd.DataFrame(periods)
    strategy_returns = pd.Series(returns["strategy_return"].to_numpy(), index=returns.index)
    benchmark_returns = pd.Series(returns["benchmark_return"].to_numpy(), index=returns.index)
    strategy_equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    metrics = BacktestEngine._metrics(
        strategy_returns, benchmark_returns, strategy_equity, benchmark_equity,
        returns["turnover"], returns["cost"],
    )
    years = max((dates[-1] - dates[0]).days / 365.2425, 1 / 365.2425)
    metrics.update({
        "average_active_exposure": float(np.mean(exposure_history)) if exposure_history else 0.0,
        "annual_turnover": float(returns["turnover"].sum() / years),
        "total_turnover": float(returns["turnover"].sum()),
        "risk_halt_count": int(halt_count),
        "cooldown_count": int((returns["risk_state"] == "cooldown").sum()),
        "holding_days": int(sum(bool(x) for x in signals["holdings"])) if not signals.empty else 0,
    })
    return {
        "execution": execution,
        "cost_bps_one_way": cost_bps,
        "metrics": metrics,
        "return_periods": periods,
        "rebalances": int((returns["turnover"] > 1e-12).sum()),
    }


def yearly_returns(result: dict) -> dict:
    periods = pd.DataFrame(result["return_periods"])
    if periods.empty:
        return {}
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    return {
        str(year): float((1.0 + group["strategy_return"].astype(float)).prod() - 1.0)
        for year, group in periods.groupby(periods["period_end"].dt.year)
    }


def normalized_curve(result: dict, name: str) -> pd.Series:
    periods = pd.DataFrame(result["return_periods"])
    periods["period_start"] = pd.to_datetime(periods["period_start"])
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    curve = (1.0 + periods.set_index("period_end")["strategy_return"].astype(float)).cumprod()
    return pd.concat([pd.Series([1.0], index=[periods["period_start"].iloc[0]]), curve]).rename(name)


def main() -> None:
    source = json.loads((APP_BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history = _clean_history(history)
    history = history[history["date"] <= pd.Timestamp(PAPER_END)].copy()
    signals, signal_audit = build_paper_signals(history)
    paper_native = simulate_paper_strategy(history, signals, "close_to_close", PAPER_COST_BPS)
    paper_next_open = simulate_paper_strategy(history, signals, "next_open", UNIFIED_COST_BPS)

    frame, feature_columns = build_alpha158_lite(history)
    baseline_prediction = baseline_scores(frame, PAPER_START, PAPER_END)
    lgbm_prediction, lgbm_audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, PAPER_START, PAPER_END
    )
    lgbm_scores = blend_scores(frame, lgbm_prediction, baseline_prediction, WINNER_CONFIG["baseline_weight"])
    lgbm_result = topk_dropout_backtest(history, lgbm_scores, PAPER_START, PAPER_END, cost_bps=UNIFIED_COST_BPS)
    lgbm_summary = research_summary(lgbm_result, PAPER_START, PAPER_END)
    lgbm_metrics = {**lgbm_result["metrics"], **lgbm_summary["performance"]}

    curves = pd.concat({
        "paper_next_open": normalized_curve(paper_next_open, "paper_next_open"),
        "lgbm_top5": normalized_curve(
            {"return_periods": lgbm_result["return_periods"]}, "lgbm_top5"
        ),
    }, axis=1, join="outer").sort_index().ffill()
    curves = curves.dropna(how="all")
    common_start = curves.index.min()
    curves = curves.div(curves.loc[common_start]).ffill()
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")

    comparison = {
        "paper_native_close_to_close": {
            "metrics": paper_native["metrics"],
            "yearly_returns": yearly_returns(paper_native),
        },
        "paper_unified_next_open": {
            "metrics": paper_next_open["metrics"],
            "yearly_returns": yearly_returns(paper_next_open),
        },
        "current_lgbm_top5_unified": {
            "metrics": lgbm_metrics,
            "activity": lgbm_summary["activity"],
            "yearly_returns": yearly_returns({"return_periods": lgbm_result["return_periods"]}),
        },
    }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "paper": {
            "arxiv": "2511.13251v1",
            "title": "Sharpe-Driven Stock Selection and Liquidity-Constrained Portfolio Optimization: Evidence from the Chinese Equity Market",
            "reported_period": [PAPER_START.isoformat(), PAPER_END.isoformat()],
            "reported_claims": {"annual_return": 0.25, "sharpe": 1.71, "max_drawdown": -0.082},
            "implemented_core": [
                "rolling Sharpe universe ranking",
                "50% inverse-volatility + 50% nonnegative-Sharpe weights",
                "drawdown exposure: 100% / 80% / 40% / 0% at 2% / 4% / 6%",
            ],
            "proxy_assumptions": signal_audit,
        },
        "window": {"start": PAPER_START.isoformat(), "end": PAPER_END.isoformat()},
        "universe_snapshot_run_id": snapshot_id,
        "requested_codes": len(requested_codes),
        "loaded_codes": int(history["code"].nunique()),
        "lgbm": {
            "model_version": "lgbm_ranker_top5_v1",
            "config": WINNER_CONFIG,
            "feature_count": len(feature_columns),
            "audit": lgbm_audit,
        },
        "execution_protocols": {
            "paper_native": "signal close; close-to-close return; daily rebalance; 5 bps one-way",
            "unified": "signal close; next-open execution; paper daily rebalance versus LGBM Top5/10-session; 12 bps one-way",
        },
        "comparison": comparison,
        "curve_data": str(CURVE_PATH.relative_to(APP_BASE_DIR)),
        "warnings": warnings + [
            "论文没有公开 rolling window、top quantile、流动性/市值/价差阈值；本结果使用显式代理参数",
            "当前缓存没有 point-in-time market cap 或 bid/ask spread，成交额阈值只是流动性 proxy",
            "当前 Top50 是历史回填的存续/成分偏差股票池，前复权价格不是严格 as-of 价格",
            "论文中的 GP alpha evolution 未纳入，因正文没有可执行算子、搜索空间和随机种子细节",
            "结果是研究回测，不能视为论文数字的逐值复现，也不自动替换生产 LGBM",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{float(x):.2%}"

    lgbm_report_metrics = {
        **comparison["current_lgbm_top5_unified"]["metrics"],
        **comparison["current_lgbm_top5_unified"]["activity"],
    }
    rows = [
        ("论文原生：close-to-close/5bp", comparison["paper_native_close_to_close"]["metrics"]),
        ("论文统一：next-open/12bp", comparison["paper_unified_next_open"]["metrics"]),
        ("当前 LightGBM：Top5/10日/12bp", lgbm_report_metrics),
    ]
    report = [
        "# arXiv:2511.13251v1 复现与 LightGBM 对比",
        "",
        f"- 区间：{PAPER_START.isoformat()} 至 {PAPER_END.isoformat()}；请求股票 {len(requested_codes)} 只，实际加载 {history['code'].nunique()} 只。",
        f"- 冻结股票池快照：`{snapshot_id}`。",
        "- 论文原生口径：滚动 Sharpe 选股、每日调仓、close-to-close、单边 5bp。",
        "- 统一口径：信号收盘、下一交易日开盘执行、单边 12bp；LGBM 使用冻结 Top5、每10个交易日调仓。",
        "",
        "| 方法 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 年化换手 | 平均股票暴露 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in rows:
        report.append(
            f"| {label} | {pct(metrics.get('total_return', 0))} | {pct(metrics.get('annual_return', 0))} | {pct(metrics.get('annual_volatility', 0))} | {metrics.get('sharpe', 0):.3f} | {pct(metrics.get('max_drawdown', 0))} | {metrics.get('annual_turnover', metrics.get('average_turnover', 0)):.2f}x | {pct(metrics.get('average_active_exposure', 1))} |"
        )
    paper_u = comparison["paper_unified_next_open"]["metrics"]
    lgbm_u = comparison["current_lgbm_top5_unified"]["metrics"]
    report += [
        "",
        f"- 统一执行口径下，LightGBM 相对论文策略：年化收益差 {pct(lgbm_u.get('annual_return', 0) - paper_u.get('annual_return', 0))}，Sharpe 差 {lgbm_u.get('sharpe', 0) - paper_u.get('sharpe', 0):.3f}。",
        "",
        "## 复现边界",
        "",
        "论文未公开滚动窗口、Top quantile、流动性阈值、市值阈值和买卖价差数据源；默认代理为 60 日、Top20%、20日平均成交额不低于1000万元。市值和 bid-ask spread 在现有日线缓存中不可得，因此不能宣称严格逐字复现。",
        "论文提到 GP alpha evolution，但未提供足够的算子集合、树深、适应度、搜索轮数和防过拟合协议；本版只复现正文中可执行且有明确公式的三阶段核心。",
        "当前股票池为当前 Top50 的历史回填，存在成分与存续偏差；价格为当前版本前复权日线。所有结果仅是研究回测，生产 LGBM 未被修改。",
        "",
        f"- 结果 JSON：`{OUTPUT_PATH.relative_to(APP_BASE_DIR)}`",
        f"- 曲线 CSV：`{CURVE_PATH.relative_to(APP_BASE_DIR)}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "report": str(REPORT_PATH), "curve": str(CURVE_PATH), "comparison": comparison}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
