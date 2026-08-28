from dataclasses import asdict, dataclass
from datetime import date, datetime
import math
from typing import Dict, List
from uuid import uuid4

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    mode: str
    start_date: date
    end_date: date
    codes: List[str]
    top_n: int = 5
    rebalance_days: int = 20
    initial_capital: float = 1_000_000
    cost_bps: float = 12.0
    strategy: str = "multi_factor"
    position_adjustment: str = "none"
    weight_band: float = 0.03
    rebalance_policy: str = "fixed"
    min_hold_days: int = 5
    rank_buffer: int = 20
    score_gap: float = 15.0
    max_replacements: int = 1
    entry_rank: int = 3
    max_adaptive_per_cycle: int = 1
    universe_policy: str = "current_snapshot"
    price_adjustment_policy: str = "current_vintage_qfq"
    strict_no_lookahead: bool = False
    risk_overlay: str = "trend_volatility"
    drawdown_limit: float = 0.25
    drawdown_cooldown_days: int = 10


def _rank(series: pd.Series, ascending: bool = True, neutral: float = 0.5):
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() < 2:
        return pd.Series(neutral, index=series.index, dtype=float)
    return numeric.rank(pct=True, ascending=ascending).fillna(neutral)


def _finite(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else default


class BacktestEngine:
    FACTOR_WEIGHTS = {"momentum": 0.35, "trend": 0.15, "liquidity": 0.15, "value": 0.15, "risk": 0.20}
    STRATEGIES = {
        "multi_factor": {
            "label": "原始多因子",
            "weights": FACTOR_WEIGHTS,
        },
        "contrarian": {
            "label": "逆向增强",
            "weights": {
                "medium_reversal": 0.33,
                "inefficiency": 0.23,
                "liquidity_cooling": 0.19,
                "low_volatility": 0.05,
                "risk": 0.10,
                "value": 0.10,
            },
        },
    }

    def run(self, history: pd.DataFrame, config: BacktestConfig, warnings: List[str]):
        self._validate_research_contract(history, config)
        strategy = self.STRATEGIES.get(config.strategy)
        if strategy is None:
            raise ValueError("未知回测策略")
        factor_weights = getattr(self, "_factor_weights_override", strategy["weights"])
        frame = self._features(history)
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        dates = dates[(dates >= pd.Timestamp(config.start_date)) & (dates <= pd.Timestamp(config.end_date))]
        if len(dates) < max(30, config.rebalance_days * 2):
            raise ValueError("回测区间内有效交易日不足")

        open_prices = frame.pivot(index="date", columns="code", values="open").reindex(dates)
        period_returns = open_prices.shift(-1).div(open_prices).sub(1).replace([np.inf, -np.inf], np.nan)
        if "universe_member" in frame.columns:
            membership = (
                frame.pivot(index="date", columns="code", values="universe_member")
                .reindex(index=dates, columns=open_prices.columns)
                .fillna(False)
                .astype(bool)
            )
            period_returns = period_returns.where(membership)
        benchmark_returns = period_returns.mean(axis=1, skipna=True).fillna(0)
        risk_exposure = self._risk_exposure(frame, dates, config.risk_overlay)

        targets, rebalance_records = self._build_targets(
            frame, dates, open_prices.columns, config, factor_weights, risk_exposure
        )

        if not rebalance_records:
            raise ValueError("没有产生有效调仓记录，请扩大股票池或回测区间")
        realized_returns = period_returns.fillna(0)
        current_weights = pd.Series(0.0, index=open_prices.columns)
        daily_returns = pd.Series(0.0, index=dates)
        turnovers = pd.Series(0.0, index=dates)
        active = pd.Series(False, index=dates)
        days_since_main_rebalance = None
        interim_adjustment_count = 0
        portfolio_equity = 1.0
        peak_equity = 1.0
        halt_until = -1
        risk_halt_count = 0
        for date_index, trade_date in enumerate(dates):
            target = targets.loc[trade_date]
            drawdown = portfolio_equity / peak_equity - 1 if peak_equity > 0 else 0.0
            cooldown_active = date_index < halt_until
            limit_breached = config.drawdown_limit > 0 and drawdown <= -config.drawdown_limit
            if cooldown_active or limit_breached:
                if current_weights.sum() > 1e-12:
                    target = pd.Series(0.0, index=open_prices.columns)
                    risk_halt_count += 1
                else:
                    target = pd.Series(np.nan, index=open_prices.columns)
                if limit_breached:
                    halt_until = max(halt_until, date_index + max(1, config.drawdown_cooldown_days))
                    # Reset the internal risk clock after a forced exit so
                    # cash can re-enter after the cooldown instead of being
                    # blocked forever by the old high-water mark.
                    peak_equity = portfolio_equity
            if target.notna().any():
                target = target.fillna(0.0)
                turnovers.loc[trade_date] = target.sub(current_weights).abs().sum()
                current_weights = target
                days_since_main_rebalance = 0
            elif days_since_main_rebalance is not None:
                days_since_main_rebalance += 1
                adjustment_target = self._interim_adjustment_target(
                    current_weights, config.position_adjustment, config.weight_band, days_since_main_rebalance
                )
                if adjustment_target is not None:
                    turnover = adjustment_target.sub(current_weights).abs().sum()
                    if turnover > 1e-12:
                        turnovers.loc[trade_date] = turnover
                        current_weights = adjustment_target
                        interim_adjustment_count += 1
            active.loc[trade_date] = current_weights.sum() > 0
            returns_today = realized_returns.loc[trade_date]
            gross_return = float(current_weights.mul(returns_today).sum())
            daily_returns.loc[trade_date] = gross_return
            portfolio_growth = 1 + gross_return
            if current_weights.sum() > 0 and portfolio_growth > 0:
                current_weights = current_weights.mul(1 + returns_today).div(portfolio_growth)
            # Track the realized account curve using only information known at
            # this close; the next loop iteration can therefore apply a valid
            # next-open circuit breaker.
            day_cost = float(turnovers.loc[trade_date] * config.cost_bps / 10_000)
            portfolio_equity *= max(0.0, 1 + gross_return - day_cost)
            peak_equity = max(peak_equity, portfolio_equity)

        costs = turnovers * config.cost_bps / 10_000
        strategy_returns = (daily_returns - costs).where(active | costs.gt(0), 0)

        strategy_equity = (1 + strategy_returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()
        drawdown = strategy_equity.div(strategy_equity.cummax()).sub(1)
        metrics = self._metrics(strategy_returns, benchmark_returns, strategy_equity, benchmark_equity, turnovers, costs)
        curves = [
            {
                "date": trade_date.strftime("%Y-%m-%d"),
                "strategy": round(_finite(strategy_equity.loc[trade_date]), 6),
                "benchmark": round(_finite(benchmark_equity.loc[trade_date]), 6),
                "drawdown": round(_finite(drawdown.loc[trade_date]), 6),
            }
            for trade_date in dates
        ]
        return_periods = [
            {
                "period_start": dates[index].strftime("%Y-%m-%d"),
                "period_end": dates[index + 1].strftime("%Y-%m-%d"),
                "strategy_return": round(_finite(strategy_returns.iloc[index]), 10),
                "benchmark_return": round(_finite(benchmark_returns.iloc[index]), 10),
                "turnover": round(_finite(turnovers.iloc[index]), 10),
                "cost": round(_finite(costs.iloc[index]), 10),
            }
            for index in range(len(dates) - 1)
        ]
        config_payload = asdict(config)
        config_payload["start_date"] = config.start_date.isoformat()
        config_payload["end_date"] = config.end_date.isoformat()
        return {
            "backtest_id": f"bt-{datetime.now():%Y%m%d%H%M%S}-{uuid4().hex[:6]}",
            "created_at": datetime.now().isoformat(),
            "config": config_payload,
            "data": {
                "symbols_requested": len(config.codes),
                "symbols_loaded": int(frame["code"].nunique()),
                "trading_days": len(dates),
                "benchmark": "股票池每日等权",
                "execution": "收盘生成信号，下一交易日开盘生效",
                "portfolio_accounting": (
                    "持仓权重随收益漂移，仅在调仓执行日交易并扣费"
                    if config.position_adjustment == "none"
                    else "持仓权重随收益漂移，主调仓和持有期仓位调整均按实际权重差交易并扣费"
                ),
                "position_adjustment": config.position_adjustment,
                "interim_adjustment_count": interim_adjustment_count,
                "risk_overlay": config.risk_overlay,
                "risk_exposure_range": [
                    round(float(risk_exposure.min()), 3),
                    round(float(risk_exposure.max()), 3),
                ],
                "drawdown_limit": config.drawdown_limit,
                "drawdown_cooldown_days": config.drawdown_cooldown_days,
                "risk_halt_count": risk_halt_count,
                "rebalance_policy": config.rebalance_policy,
                "policy_label": "每日自适应" if config.rebalance_policy == "adaptive" else "固定周期",
                "scheduled_rebalances": sum(item["trigger"] == "scheduled" for item in rebalance_records),
                "adaptive_rebalances": sum(item["trigger"] != "scheduled" for item in rebalance_records),
                "strategy": strategy["label"],
                "factor_weights": factor_weights,
                "lookahead_contract": {
                    "strict_no_lookahead": config.strict_no_lookahead,
                    "universe_policy": config.universe_policy,
                    "price_adjustment_policy": config.price_adjustment_policy,
                    "period_label": "收益归属于结束交易日，分段统计不得跨边界重复使用",
                },
            },
            "metrics": metrics,
            "curves": curves,
            "return_periods": return_periods,
            "rebalances": rebalance_records,
            "warnings": warnings + (["逆向增强策略旨在捕捉大盘股阶段性均值回归，不保证在趋势行情持续有效"] if config.strategy == "contrarian" else []) + (["每日自适应策略使用排名缓冲和分差阈值提前换股，仍需独立样本与模拟盘验证"] if config.rebalance_policy == "adaptive" else []) + (["趋势/波动风险覆盖层会在市场转弱时降低股票暴露，剩余资金保留现金"] if config.risk_overlay != "none" else []) + [
                "股票池来自当前候选或手工输入，可能存在成分与存续偏差",
                "未使用当前行业标签回填历史，板块因子不参与本回测",
            ],
        }

    @staticmethod
    def _validate_research_contract(history: pd.DataFrame, config: BacktestConfig) -> None:
        if history.empty:
            raise ValueError("历史数据为空")
        required = {"date", "code", "open", "close"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"历史数据缺少字段: {', '.join(sorted(missing))}")
        keys = history[["date", "code"]].copy()
        keys["date"] = pd.to_datetime(keys["date"])
        if keys.duplicated(["date", "code"]).any():
            raise ValueError("历史数据存在重复 date+code，严格回测已拒绝")
        if not config.strict_no_lookahead:
            return
        blockers = []
        if config.universe_policy != "point_in_time":
            blockers.append("股票池不是逐日历史时点成分")
        if "universe_member" not in history.columns:
            blockers.append("缺少 universe_member 逐日成分标记")
        if config.price_adjustment_policy not in {"raw_point_in_time", "asof_adjusted"}:
            blockers.append("价格数据不是原始时点价或 as-of 复权价")
        if "available_date" not in history.columns:
            blockers.append("缺少 available_date 可用性时间戳")
        else:
            available = pd.to_datetime(history["available_date"], errors="coerce")
            observed = pd.to_datetime(history["date"], errors="coerce")
            if available.isna().any() or (available > observed).any():
                blockers.append("存在行数据在信号日尚不可用")
        if blockers:
            raise ValueError("严格无前视门禁未通过: " + "；".join(blockers))

    def _build_targets(self, frame, dates, columns, config, factor_weights, risk_exposure=None):
        if config.rebalance_policy not in {"fixed", "adaptive"}:
            raise ValueError("未知调仓策略")
        targets = pd.DataFrame(index=dates, columns=columns, dtype=float)
        records = []
        planned_holdings: List[str] = []
        entry_indices: Dict[str, int] = {}
        pending_targets = {}
        adaptive_in_cycle = 0

        for signal_index, signal_date in enumerate(dates):
            if signal_index in pending_targets:
                next_holdings = pending_targets.pop(signal_index)
                previous = set(planned_holdings)
                planned_holdings = list(next_holdings)
                for code in set(planned_holdings) - previous:
                    entry_indices[code] = signal_index
                for code in previous - set(planned_holdings):
                    entry_indices.pop(code, None)

            scheduled = signal_index % config.rebalance_days == 0
            if scheduled:
                adaptive_in_cycle = 0
            if config.rebalance_policy == "fixed" and not scheduled:
                continue
            execution_index = signal_index + 1
            if execution_index >= len(dates) - 1:
                continue
            cross = self._scored_cross(frame, signal_date, factor_weights)
            if cross.empty:
                continue

            trigger = "scheduled"
            reason = f"第{config.rebalance_days}交易日主调仓"
            if scheduled or not planned_holdings:
                selected_codes = cross.head(min(config.top_n, len(cross)))["code"].astype(str).tolist()
            else:
                selected_codes, reason = self._adaptive_holdings(
                    cross,
                    planned_holdings,
                    entry_indices,
                    signal_index,
                    config,
                    allow_regular=adaptive_in_cycle < config.max_adaptive_per_cycle,
                )
                if set(selected_codes) == set(planned_holdings):
                    continue
                trigger = "risk" if reason.startswith("风险退出") else "adaptive"
                if trigger == "adaptive":
                    adaptive_in_cycle += 1

            execution_date = dates[execution_index]
            selected = cross.set_index("code").reindex(selected_codes).dropna(subset=["score"]).reset_index()
            if selected.empty:
                continue
            selected_codes = selected["code"].astype(str).tolist()
            exposure = float(risk_exposure.get(execution_date, 1.0)) if risk_exposure is not None else 1.0
            target_weight = exposure / len(selected_codes)
            targets.loc[execution_date, :] = 0.0
            for code in selected_codes:
                targets.loc[execution_date, code] = target_weight
            pending_targets[execution_index] = selected_codes
            records.append({
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "execution_date": execution_date.strftime("%Y-%m-%d"),
                "trigger": trigger,
                "reason": reason,
                "exposure": round(exposure, 3),
                "holdings": [
                    {
                        "code": str(row["code"]),
                        "name": str(row["name"]),
                        "score": round(float(row["score"]), 1),
                        "weight": round(target_weight, 4),
                    }
                    for _, row in selected.iterrows()
                ],
            })
        return targets, records

    @staticmethod
    def _risk_exposure(frame, dates, mode: str) -> pd.Series:
        """Compute an as-of market risk budget for each execution date.

        The proxy uses only close data available before the signal date. It is
        deliberately coarse and cash-preserving: no leverage and no shorting.
        """
        if mode == "none":
            return pd.Series(1.0, index=dates, dtype=float)
        all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"].unique())))
        close = frame.pivot(index="date", columns="code", values="close").reindex(all_dates)
        proxy = close.mean(axis=1, skipna=True)
        daily = proxy.pct_change(fill_method=None)
        trend = proxy.div(proxy.rolling(120, min_periods=60).mean()).sub(1)
        volatility = daily.rolling(20, min_periods=10).std().mul(np.sqrt(252))
        exposure = pd.Series(1.0, index=all_dates, dtype=float)
        if mode == "balanced":
            exposure = exposure.mask(trend < -0.08, 0.90)
            exposure = exposure.mask(trend < -0.15, 0.75)
            exposure = exposure.mask(trend < -0.25, 0.50)
            exposure = exposure.mask(volatility > 0.50, exposure * 0.90)
            exposure = exposure.mask(volatility > 0.70, exposure * 0.80)
            floor = 0.40
        elif mode == "adaptive":
            # Require both a trend break and elevated volatility before
            # reducing exposure, preserving upside in ordinary pullbacks.
            exposure = exposure.mask((trend < -0.03) & (volatility > 0.30), 0.80)
            exposure = exposure.mask((trend < -0.08) & (volatility > 0.35), 0.55)
            exposure = exposure.mask((trend < -0.15) & (volatility > 0.45), 0.30)
            floor = 0.25
        else:
            exposure = exposure.mask(trend < -0.03, 0.75)
            exposure = exposure.mask(trend < -0.08, 0.50)
            exposure = exposure.mask(trend < -0.15, 0.25)
            exposure = exposure.mask(volatility > 0.35, exposure * 0.80)
            exposure = exposure.mask(volatility > 0.50, exposure * 0.75)
            floor = 0.20
        return exposure.reindex(dates).fillna(1.0).clip(floor, 1.0)

    def _scored_cross(self, frame, signal_date, factor_weights):
        cross = frame[frame["date"] == signal_date].copy()
        if "universe_member" in cross.columns:
            cross = cross[cross["universe_member"].fillna(False).astype(bool)]
        cross = cross[
            (cross["tradestatus"] == 1)
            & (cross["is_st"] == 0)
            & (cross["close"] > 1)
            & (cross["amount_20"] >= 10_000_000)
        ].dropna(subset=[
            "ret_5", "ret_20", "ret_60", "volatility_20", "amount_20",
            "amount_60", "absolute_return_60", "ma_20",
        ])
        if cross.empty:
            return cross
        return self._score(cross, factor_weights).sort_values(
            ["score", "amount_20"], ascending=[False, False]
        ).reset_index(drop=True)

    @staticmethod
    def _adaptive_holdings(cross, holdings, entry_indices, signal_index, config, allow_regular=True):
        ranked_codes = cross["code"].astype(str).tolist()
        rank_by_code = {code: rank for rank, code in enumerate(ranked_codes, start=1)}
        score_by_code = dict(zip(ranked_codes, cross["score"].astype(float)))
        selected = list(holdings)
        reasons = []

        for _ in range(max(1, config.max_replacements)):
            challengers = [code for code in ranked_codes[: config.entry_rank] if code not in selected]
            if not challengers:
                break
            best_entry = challengers[0]
            forced_exits = [code for code in selected if code not in rank_by_code]
            if forced_exits:
                exit_code = forced_exits[0]
                should_replace = True
                reason = f"风险退出 {exit_code}"
            else:
                if not allow_regular:
                    break
                eligible_exits = [
                    code for code in selected
                    if signal_index - entry_indices.get(code, signal_index) >= config.min_hold_days
                ]
                if not eligible_exits:
                    break
                exit_code = max(eligible_exits, key=lambda code: rank_by_code.get(code, 10**6))
                rank_trigger = rank_by_code.get(exit_code, 10**6) > config.rank_buffer
                score_advantage = score_by_code[best_entry] - score_by_code.get(exit_code, 0.0)
                should_replace = rank_trigger or score_advantage >= config.score_gap
                reason = (
                    f"排名缓冲退出 {exit_code}"
                    if rank_trigger
                    else f"分差触发 {best_entry}-{exit_code} {score_advantage:.1f}分"
                )
            if not should_replace:
                break
            selected[selected.index(exit_code)] = best_entry
            reasons.append(reason)
        selected.sort(key=lambda code: rank_by_code.get(code, 10**6))
        return selected, "；".join(reasons) if reasons else "保持持仓"

    @staticmethod
    def _interim_adjustment_target(current_weights, mode: str, weight_band: float, days_since_main: int):
        if mode == "none":
            return None
        active = current_weights[current_weights > 0]
        if active.empty:
            return None
        equal_weight = 1 / len(active)
        should_adjust = mode == "daily_equal"
        if mode == "midcycle_equal":
            should_adjust = days_since_main == 5
        elif mode == "band_equal":
            should_adjust = active.sub(equal_weight).abs().max() >= weight_band
        elif mode != "daily_equal":
            if mode not in {"midcycle_equal", "band_equal"}:
                raise ValueError("未知持有期仓位调整模式")
        if not should_adjust:
            return None
        target = pd.Series(0.0, index=current_weights.index)
        target.loc[active.index] = equal_weight
        return target

    def _features(self, history: pd.DataFrame):
        frame = history.copy().sort_values(["code", "date"])
        frame["date"] = pd.to_datetime(frame["date"])
        grouped = frame.groupby("code", group_keys=False)
        frame["ret_1"] = grouped["close"].pct_change()
        frame["ret_5"] = grouped["close"].pct_change(5)
        frame["ret_20"] = grouped["close"].pct_change(20)
        frame["ret_60"] = grouped["close"].pct_change(60)
        frame["ret_120"] = grouped["close"].pct_change(120)
        frame["ma_20"] = grouped["close"].transform(lambda values: values.rolling(20).mean())
        frame["amount_5"] = grouped["amount"].transform(lambda values: values.rolling(5).mean())
        frame["amount_20"] = grouped["amount"].transform(lambda values: values.rolling(20).mean())
        frame["amount_60"] = grouped["amount"].transform(lambda values: values.rolling(60).mean())
        frame["volatility_20"] = grouped["ret_1"].transform(lambda values: values.rolling(20).std())
        frame["volatility_60"] = grouped["ret_1"].transform(lambda values: values.rolling(60).std())
        frame["downside_volatility_60"] = grouped["ret_1"].transform(
            lambda values: values.clip(upper=0).pow(2).rolling(60).mean().pow(0.5)
        )
        frame["absolute_return_60"] = grouped["ret_1"].transform(
            lambda values: values.abs().rolling(60).sum()
        )
        frame["absolute_return_120"] = grouped["ret_1"].transform(
            lambda values: values.abs().rolling(120).sum()
        )
        frame["high_60"] = grouped["close"].transform(lambda values: values.rolling(60).max())
        frame["drawdown_60"] = frame["close"].div(frame["high_60"]).sub(1)
        frame["high_120"] = grouped["close"].transform(lambda values: values.rolling(120).max())
        frame["drawdown_120"] = frame["close"].div(frame["high_120"]).sub(1)
        frame["intraday_return"] = frame["close"].div(frame["open"]).sub(1)
        frame["intraday_strength_20"] = grouped["intraday_return"].transform(
            lambda values: values.rolling(20).mean()
        )
        return frame

    def _score(self, cross: pd.DataFrame, factor_weights=None):
        factor_weights = factor_weights or self.FACTOR_WEIGHTS
        cross["momentum_score"] = _rank(cross["ret_20"]) * 0.55 + _rank(cross["ret_60"]) * 0.45
        cross["trend_score"] = _rank(cross["close"].div(cross["ma_20"]).sub(1))
        cross["liquidity_score"] = _rank(np.log1p(cross["amount_20"]))
        momentum_skip = (1 + cross["ret_60"]).div(1 + cross["ret_5"]).sub(1)
        trend_efficiency = cross["ret_60"].div(cross["absolute_return_60"].replace(0, np.nan))
        liquidity_change = cross["amount_20"].div(cross["amount_60"]).sub(1)
        long_momentum = (1 + cross["ret_120"]).div(1 + cross["ret_20"]).sub(1)
        long_trend_efficiency = cross["ret_120"].div(
            cross["absolute_return_120"].replace(0, np.nan)
        )
        cross["medium_reversal_score"] = _rank(-momentum_skip)
        cross["inefficiency_score"] = _rank(-trend_efficiency)
        cross["liquidity_cooling_score"] = _rank(-liquidity_change)
        cross["long_momentum_score"] = _rank(long_momentum)
        cross["trend_quality_score"] = _rank(long_trend_efficiency)
        cross["low_volatility_score"] = 1 - _rank(cross["volatility_60"])
        cross["downside_quality_score"] = 1 - _rank(cross["downside_volatility_60"])
        cross["short_reversal_score"] = _rank(-cross["ret_5"])
        cross["price_position_score"] = _rank(cross["drawdown_120"])
        cross["intraday_quality_score"] = _rank(cross["intraday_strength_20"])
        pe = cross["pe"].where(cross["pe"].between(3, 100))
        pb = cross["pb"].where(cross["pb"].between(0.2, 15))
        cross["value_score"] = ((1 - _rank(pe)) * 0.6 + (1 - _rank(pb)) * 0.4).where(pe.notna() | pb.notna(), 0.5)
        cross["risk_score"] = (1 - _rank(cross["volatility_20"])) * 0.6 + _rank(cross["drawdown_60"]) * 0.4
        cross["score"] = sum(cross[f"{name}_score"] * weight for name, weight in factor_weights.items()) * 100
        return cross

    @staticmethod
    def _metrics(strategy, benchmark, equity, benchmark_equity, turnovers, costs) -> Dict[str, float]:
        active = strategy.loc[equity.index]
        days = max(1, len(active))
        annual_return = equity.iloc[-1] ** (252 / days) - 1
        annual_volatility = active.std(ddof=0) * np.sqrt(252)
        annualized_mean = active.mean() * 252
        downside = np.sqrt(np.minimum(active, 0).pow(2).mean()) * np.sqrt(252)
        max_drawdown = equity.div(equity.cummax()).sub(1).min()
        excess_daily = active - benchmark.reindex(active.index).fillna(0)
        benchmark_variance = benchmark.var(ddof=0)
        beta = active.cov(benchmark) / benchmark_variance if benchmark_variance > 0 else 0
        alpha = (active.mean() - beta * benchmark.mean()) * 252
        information_ratio = excess_daily.mean() / excess_daily.std(ddof=0) * np.sqrt(252) if excess_daily.std(ddof=0) > 0 else 0
        return {
            "total_return": round(_finite(equity.iloc[-1] - 1), 6),
            "annual_return": round(_finite(annual_return), 6),
            "benchmark_return": round(_finite(benchmark_equity.iloc[-1] - 1), 6),
            "benchmark_annual_return": round(_finite(benchmark_equity.iloc[-1] ** (252 / days) - 1), 6),
            "excess_return": round(_finite(equity.iloc[-1] - benchmark_equity.iloc[-1]), 6),
            "annual_volatility": round(_finite(annual_volatility), 6),
            "max_drawdown": round(_finite(max_drawdown), 6),
            "sharpe": round(_finite(annualized_mean / annual_volatility if annual_volatility > 0 else 0), 4),
            "sortino": round(_finite(annualized_mean / downside if downside > 0 else 0), 4),
            "calmar": round(_finite(annual_return / abs(max_drawdown) if max_drawdown < 0 else 0), 4),
            "win_rate": round(_finite((active > 0).mean()), 6),
            "alpha": round(_finite(alpha), 6),
            "beta": round(_finite(beta), 4),
            "information_ratio": round(_finite(information_ratio), 4),
            "average_turnover": round(_finite(turnovers[turnovers > 0].mean()), 6),
            "total_cost": round(_finite(costs.sum()), 6),
            "rebalance_count": int((turnovers > 0).sum()),
        }
