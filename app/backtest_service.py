from datetime import date
from threading import Lock
from typing import Dict, List

from app.backtest_data import BaoStockHistoryProvider, DemoHistoryProvider, parse_codes
from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import Settings
from app.storage import Storage


class BacktestService:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        self.engine = BacktestEngine()
        self._lock = Lock()

    def run(
        self,
        mode: str,
        start_date: date,
        end_date: date,
        codes: List[str],
        top_n: int,
        rebalance_days: int,
        initial_capital: float,
        cost_bps: float,
        strategy: str = "multi_factor",
        rebalance_policy: str = "fixed",
        min_hold_days: int = 5,
        rank_buffer: int = 8,
        score_gap: float = 8.0,
        risk_overlay: str = "trend_volatility",
        drawdown_limit: float = 0.25,
        drawdown_cooldown_days: int = 10,
    ):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("已有回测任务正在执行，请稍后再试")
        try:
            clean_codes = parse_codes(codes)
            if mode not in {"demo", "live"}:
                raise ValueError("mode 仅支持 demo 或 live")
            if start_date >= end_date:
                raise ValueError("开始日期必须早于结束日期")
            if (end_date - start_date).days > 365 * 12 + 3:
                raise ValueError("单次回测最长支持 12 年")
            if mode == "live" and len(clean_codes) < 2:
                raise ValueError("真实回测至少需要 2 只沪深 A 股")
            if len(clean_codes) > 60:
                raise ValueError("单次真实回测最多支持 60 只股票")
            if strategy not in self.engine.STRATEGIES:
                raise ValueError("未知回测策略")
            if rebalance_policy not in {"fixed", "adaptive"}:
                raise ValueError("未知调仓机制")
            if risk_overlay not in {"none", "balanced", "adaptive", "trend_volatility"}:
                raise ValueError("未知风险覆盖层")
            if not 0 <= drawdown_limit <= 0.8:
                raise ValueError("回撤熔断阈值必须在0到80%之间")
            if not 1 <= drawdown_cooldown_days <= 60:
                raise ValueError("回撤冷静期必须在1到60个交易日之间")
            names = self._latest_names()
            provider = DemoHistoryProvider() if mode == "demo" else BaoStockHistoryProvider(self.settings.cache_dir)
            history, warnings = provider.load(clean_codes, names, start_date, end_date)
            available = history["code"].nunique()
            config = BacktestConfig(
                mode=mode,
                start_date=start_date,
                end_date=end_date,
                codes=sorted(history["code"].unique().tolist()),
                top_n=min(max(1, top_n), available),
                rebalance_days=min(max(5, rebalance_days), 60),
                initial_capital=max(10_000, initial_capital),
                cost_bps=min(max(0, cost_bps), 100),
                strategy=strategy,
                rebalance_policy=rebalance_policy,
                min_hold_days=min(max(1, min_hold_days), 20),
                rank_buffer=min(max(top_n, rank_buffer), available),
                score_gap=min(max(0, score_gap), 50),
                max_replacements=1,
                entry_rank=min(3, available),
                max_adaptive_per_cycle=1,
                risk_overlay=risk_overlay,
                drawdown_limit=drawdown_limit,
                drawdown_cooldown_days=drawdown_cooldown_days,
            )
            result = self.engine.run(history, config, warnings)
            self.storage.save_backtest(result)
            return result
        finally:
            self._lock.release()

    def default_codes(self):
        latest = self._latest_universe_run()
        if not latest:
            return []
        return [item["code"] for item in latest.get("candidates", [])[: self.settings.top_n]]

    def _latest_names(self) -> Dict[str, str]:
        latest = self._latest_universe_run()
        if not latest:
            return {}
        return {item["code"]: item["name"] for item in latest.get("candidates", [])[: self.settings.top_n]}

    def _latest_universe_run(self):
        """Return the newest run with a complete configured universe.

        Intraday/live snapshots can be temporarily short when the provider
        filters suspended or illiquid symbols.  Backtests should keep using
        the latest complete Top-N snapshot instead of silently shrinking to
        that transient candidate list.
        """
        required = max(2, int(self.settings.top_n))
        runs = [
            run for run in self.storage.history(200)
            if len(run.get("candidates", [])) >= required
        ]
        return max(runs, key=lambda run: run["created_at"]) if runs else None
