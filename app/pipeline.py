from datetime import datetime
from threading import Lock
from typing import Optional
from uuid import uuid4

from app.config import Settings
from app.adaptive_portfolio import AdaptivePortfolioService
from app.data.akshare_provider import AkShareProvider
from app.data.demo import DemoProvider
from app.macro_environment import MacroEnvironmentService
from app.models import Candidate, RunResult
from app.notifications import NotificationService
from app.paper_account import LgbmPaperAccountService
from app.reporting import ReportGenerator
from app.selection import SelectionEngine
from app.storage import Storage
from app.strategy_signal import StrategySignalService


class ResearchPipeline:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        self.selection = SelectionEngine(
            min_amount=settings.min_amount,
            min_market_cap=settings.min_market_cap,
            weights=settings.factor_weights,
            technology_reserve=settings.technology_reserve,
        )
        self.reporter = ReportGenerator(
            report_dir=settings.report_dir,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        self.notifier = NotificationService(settings)
        self.macro_environment = MacroEnvironmentService(settings.cache_dir)
        lgbm_enabled = settings.strategy_model_backend.startswith("lgbm_")
        lgbm_label = (
            "75/25 Alpha158-Barra LGBM"
            if settings.lgbm_model_version == "lgbm_new_factors_blend_v1_candidate"
            else "LGBM LambdaRank Top5"
        )
        blend_candidate = (
            settings.lgbm_model_version == "lgbm_new_factors_blend_v1_candidate"
        )
        protocol_id = (
            f"{settings.lgbm_model_version}_fixed10"
            if blend_candidate else settings.lgbm_model_version
        )
        self.paper_account = (
            LgbmPaperAccountService(
                storage,
                initial_capital=settings.paper_initial_capital,
                cost_bps=settings.paper_cost_bps,
                lot_size=settings.paper_lot_size,
                account_id=f"{protocol_id}_paper_account_v1",
                strategy_label=f"{lgbm_label} 模拟盘",
            )
            if settings.strategy_model_backend == "lgbm_active"
            else None
        )
        self.portfolio = AdaptivePortfolioService(
            storage,
            top_n=settings.strategy_top_n,
            rebalance_days=settings.strategy_rebalance_days,
            min_hold_days=settings.strategy_min_hold_days,
            rank_buffer=settings.strategy_rank_buffer,
            score_gap=settings.strategy_score_gap,
            strategy_id=(
                f"{protocol_id}_portfolio_v1"
                if lgbm_enabled else AdaptivePortfolioService.STRATEGY_ID
            ),
            strategy_label=(lgbm_label if lgbm_enabled else "逆向增强 Top5"),
            adaptive_enabled=not blend_candidate,
        )
        self.strategy_signal = StrategySignalService(
            settings.cache_dir,
            top_n=settings.strategy_top_n,
            rebalance_days=settings.strategy_rebalance_days,
            portfolio=self.portfolio,
            strategy_backend=settings.strategy_model_backend,
            lgbm_model_dir=settings.lgbm_model_dir,
            lgbm_history_days=settings.lgbm_history_days,
            paper_account=self.paper_account,
        )
        self._lock = Lock()

    def run(self, mode: Optional[str] = None, notify: bool = False) -> RunResult:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("已有分析任务正在执行，请稍后再试")
        try:
            selected_mode = (mode or self.settings.data_mode).lower()
            if selected_mode not in {"demo", "live"}:
                raise ValueError("mode 仅支持 demo 或 live")
            provider = DemoProvider() if selected_mode == "demo" else AkShareProvider(
                self.settings.cache_dir, self.settings.max_sectors
            )
            stocks, sectors, indices, breadth = provider.load()
            candidates = self.selection.select(stocks, sectors, self.settings.top_n)
            now = datetime.now()
            macro_environment = self.macro_environment.assess(now, breadth, selected_mode)
            warnings = []
            if breadth.get("provider_error"):
                warnings.append(breadth["provider_error"])
            warnings.extend(breadth.get("source_warnings", []))
            warnings.extend(f"宏观评测: {item}" for item in macro_environment.get("warnings", []))
            strategy_picks = []
            strategy_signal = {}
            if selected_mode == "live" and self.settings.strategy_push_enabled:
                try:
                    signal_candidates = self._strategy_candidates(candidates)
                    strategy_picks, strategy_signal, signal_warnings = self.strategy_signal.build(
                        signal_candidates, now.date()
                    )
                    warnings.extend(signal_warnings)
                    if signal_candidates is not candidates:
                        warnings.append("LGBM融合模拟盘使用模型元数据绑定的冻结Top50股票池")
                except Exception as exc:
                    warnings.append(f"{self.settings.strategy_model_backend} Top5 生成失败: {exc}")
            report = self.reporter.generate(
                now, indices, sectors, candidates, breadth, strategy_picks, strategy_signal,
                macro_environment,
            )
            if report["warning"]:
                warnings.append(report["warning"])
            notifications = []
            if notify:
                notification_title = (
                    f"LGBM模拟盘日报｜{now.strftime('%Y-%m-%d')}"
                    if self.settings.strategy_model_backend == "lgbm_active"
                    else f"A股量化研究日报｜{now.strftime('%Y-%m-%d')}"
                )
                notifications = self.notifier.send(
                    title=notification_title,
                    summary=report["summary"],
                    report=report["markdown"],
                    candidates=candidates,
                    sectors=sectors,
                    strategy_picks=strategy_picks,
                    strategy_signal=strategy_signal,
                    macro_environment=macro_environment,
                )
            status = self._market_status(now, selected_mode)
            result = RunResult(
                run_id=f"{now.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}",
                created_at=now,
                mode=selected_mode,
                market_status=status,
                indices=indices,
                sectors=sectors,
                candidates=candidates,
                market_breadth=breadth,
                report_markdown=report["markdown"],
                report_summary=report["summary"],
                strategy_picks=strategy_picks,
                strategy_signal=strategy_signal,
                macro_environment=macro_environment,
                notifications=notifications,
                warnings=warnings,
            )
            self.storage.save_run(result)
            return result
        finally:
            self._lock.release()

    def _strategy_candidates(self, current_candidates):
        if not (
            self.settings.strategy_model_backend.startswith("lgbm_")
            and self.settings.lgbm_universe_mode == "frozen_snapshot"
        ):
            return current_candidates
        metadata_path = self.settings.lgbm_model_dir / "metadata.json"
        try:
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("无法读取冻结 LGBM 股票池元数据") from exc
        run_id = str(metadata.get("universe_snapshot_run_id") or "")
        snapshot = self.storage.get_run(run_id) if run_id else None
        rows = snapshot.get("candidates", []) if snapshot else []
        if len(rows) < 50:
            raise RuntimeError("冻结 LGBM Top50 股票池快照缺失，拒绝改用动态股票池")
        return [Candidate(**item) for item in rows[:50]]

    @staticmethod
    def _market_status(now: datetime, mode: str) -> str:
        if mode == "demo":
            return "演示数据"
        if now.weekday() >= 5:
            return "休市"
        minute = now.hour * 60 + now.minute
        if 570 <= minute <= 690 or 780 <= minute <= 900:
            return "交易中"
        return "已收盘" if minute > 900 else "盘前"

    def push_latest(self):
        latest = self.storage.latest()
        if not latest:
            raise RuntimeError("当前没有可推送的研报")
        from app.models import Candidate, Sector, StrategyPick

        candidates = [Candidate(**item) for item in latest["candidates"]]
        sectors = [Sector(**item) for item in latest["sectors"]]
        strategy_picks = [StrategyPick(**item) for item in latest.get("strategy_picks", [])]
        return self.notifier.send(
            title=(
                f"LGBM模拟盘日报｜{latest['created_at'][:10]}"
                if latest.get("strategy_signal", {}).get("paper_account")
                else f"A股量化研究日报｜{latest['created_at'][:10]}"
            ),
            summary=latest["report_summary"],
            report=latest["report_markdown"],
            candidates=candidates,
            sectors=sectors,
            strategy_picks=strategy_picks,
            strategy_signal=latest.get("strategy_signal", {}),
            macro_environment=latest.get("macro_environment", {}),
        )
