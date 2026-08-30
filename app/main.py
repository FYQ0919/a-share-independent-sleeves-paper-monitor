from contextlib import asynccontextmanager
import asyncio
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.backtest_service import BacktestService
from app.adaptive_portfolio import AdaptivePortfolioService
from app.config import BASE_DIR, settings
from app.local_auth import LocalWebAuth
from app.pipeline import ResearchPipeline
from app.paper_account import LgbmPaperAccountService
from app.scheduler import build_scheduler
from app.storage import Storage
from app.web_auth import FeishuWebAuth


storage = Storage(settings.database_path)
pipeline = ResearchPipeline(settings, storage)
backtest_service = BacktestService(settings, storage)
scheduler = build_scheduler(settings, pipeline)


def _lgbm_research_status():
    """Expose a compact, read-only audit summary for the frozen LGBM artifact."""
    metadata_path = settings.lgbm_model_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "status": "model_missing"}
    model_version = metadata.get("model_version")
    research_path = (
        BASE_DIR / "data" / "lgbm_new_factors_v1.json"
        if model_version == "lgbm_new_factors_blend_v1_candidate"
        else BASE_DIR / "data" / "qlib_lgbm_ranker_research_v1.json"
    )
    research_summary = {}
    try:
        research = json.loads(research_path.read_text(encoding="utf-8"))
        gate = research.get("promotion_gate") or research.get("test_gate", {})
        winner = research.get("selected_feature_batch") or research.get("selection_winner")
        winner_name = winner.get("name") if isinstance(winner, dict) else winner
        research_summary = {
            "research_status": research.get("status", "unknown"),
            "research_winner": winner_name,
            "research_gate_passed": bool(gate.get("passed", False)),
            "production_change": bool(research.get("production_change", False)),
        }
    except (OSError, ValueError):
        research_summary = {"research_status": "not_available"}
    return {
        "available": True,
        "status": metadata.get("artifact_status", "unknown"),
        "model_version": metadata.get("model_version"),
        "feature_schema": metadata.get("feature_schema") or "alpha158_plus_barra",
        "feature_count": metadata.get("feature_count") or (
            metadata.get("candidate_model_audit") or {}
        ).get("feature_count"),
        "training_rows": metadata.get("training_rows") or (
            metadata.get("candidate_model_audit") or {}
        ).get("training_rows"),
        "trained_through": metadata.get("max_training_label_end_date") or max(
            (
                str(value)
                for value in [
                    (metadata.get("baseline_model_audit") or {}).get(
                        "max_training_label_end_date"
                    ),
                    (metadata.get("candidate_model_audit") or {}).get(
                        "max_training_label_end_date"
                    ),
                ]
                if value
            ),
            default=None,
        ),
        "strict_no_lookahead_certified": bool(metadata.get("strict_no_lookahead_certified", False)),
        "forward_paper_required_days": 126,
        **research_summary,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    if storage.latest() is None:
        await asyncio.to_thread(pipeline.run, "demo", False)
    if settings.schedule_enabled and not scheduler.running:
        scheduler.start()
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(title="量研台", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
if settings.web_auth_provider == "local":
    web_auth = LocalWebAuth(settings, templates)
elif settings.web_auth_provider in {"disabled", "feishu"}:
    web_auth = FeishuWebAuth(settings, templates)
else:
    raise RuntimeError("WEB_AUTH_PROVIDER 仅支持 disabled、local 或 feishu")
web_auth.install(app)


class RunRequest(BaseModel):
    mode: str = "demo"
    notify: bool = False


class BacktestRequest(BaseModel):
    mode: str = "demo"
    strategy: str = "contrarian"
    rebalance_policy: str = "adaptive"
    min_hold_days: int = 5
    rank_buffer: int = 20
    score_gap: float = 15
    start_date: str
    end_date: str
    codes: list[str] = Field(default_factory=list)
    top_n: int = 5
    rebalance_days: int = 10
    initial_capital: float = 1_000_000
    cost_bps: float = 12
    risk_overlay: str = "trend_volatility"
    drawdown_limit: float = 0.25
    drawdown_cooldown_days: int = 10


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "web_login_enabled": web_auth.enabled,
            "auth_logout_path": web_auth.logout_path,
        },
    )


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="backtest.html",
        context={
            "web_login_enabled": web_auth.enabled,
            "auth_logout_path": web_auth.logout_path,
        },
    )


@app.get("/factor-mining", response_class=HTMLResponse)
async def factor_mining_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="factor_mining.html",
        context={
            "web_login_enabled": web_auth.enabled,
            "auth_logout_path": web_auth.logout_path,
        },
    )


@app.get("/paper-monitor", response_class=HTMLResponse)
async def paper_monitor_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="paper_monitor.html",
        context={
            "web_login_enabled": web_auth.enabled,
            "auth_logout_path": web_auth.logout_path,
        },
    )


@app.get("/healthz")
async def health():
    return {
        "ok": True,
        "version": app.version,
        "has_data": storage.latest() is not None,
        "web_login_enabled": web_auth.enabled,
        "web_auth_provider": settings.web_auth_provider,
        "web_login_ready": web_auth.status.ready,
    }


@app.get("/api/dashboard")
async def dashboard():
    latest = storage.latest()
    strategy_id = pipeline.portfolio.strategy_id
    job = scheduler.get_job("daily_research") if scheduler.running else None
    factor_job = scheduler.get_job("monthly_factor_update") if scheduler.running else None
    paper_account_id = (
        pipeline.paper_account.account_id
        if pipeline.paper_account
        else f"{settings.lgbm_model_version}_paper_account_v1"
    )
    paper_account_state = storage.load_strategy_state(paper_account_id)
    hedge_account_id = pipeline.index_hedge.account_id if pipeline.index_hedge else None
    hedge_account_state = (
        storage.load_strategy_state(hedge_account_id) if hedge_account_id else None
    )
    monitor_service = getattr(pipeline, "sleeve_monitor", None)
    monitor_account_id = monitor_service.account_id if monitor_service else None
    monitor_state = (
        storage.load_strategy_state(monitor_account_id) if monitor_account_id else None
    )
    return {
        "latest": latest,
        "settings": {
            "default_mode": settings.data_mode,
            "top_n": settings.top_n,
            "schedule_enabled": settings.schedule_enabled,
            "schedule_time": f"{settings.schedule_hour:02d}:{settings.schedule_minute:02d}",
            "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None,
            "factor_auto_update_enabled": settings.factor_auto_update_enabled,
            "factor_next_run": (
                factor_job.next_run_time.isoformat()
                if factor_job and factor_job.next_run_time else None
            ),
            "channels": settings.configured_channels,
            "llm_enabled": bool(settings.llm_api_key),
            "strategy_model_backend": settings.strategy_model_backend,
            "lgbm_model_version": settings.lgbm_model_version,
            "lgbm_universe_mode": settings.lgbm_universe_mode,
            "index_hedge_enabled": settings.index_hedge_enabled,
            "independent_sleeves_enabled": settings.independent_sleeves_enabled,
            "trend_sleeve_weight": settings.trend_sleeve_weight,
            "lgbm_sleeve_weight": settings.lgbm_sleeve_weight,
            "trend_hedge_ratio": settings.trend_hedge_ratio,
            "lgbm_model_status": pipeline.strategy_signal.model_status(),
            "lgbm_research": _lgbm_research_status(),
        },
        "strategy": {
            "state": storage.load_strategy_state(strategy_id),
            "history": storage.strategy_decision_history(strategy_id, 12),
            "paper_account_id": paper_account_id,
            "paper_account": (
                paper_account_state.get("last_snapshot")
                if paper_account_state else None
            ),
            "paper_account_state": paper_account_state,
            "paper_history": storage.strategy_decision_history(
                paper_account_id, 30
            ),
            "index_hedge_account_id": hedge_account_id,
            "index_hedge": (
                hedge_account_state.get("last_snapshot")
                if hedge_account_state else None
            ),
            "index_hedge_state": hedge_account_state,
            "index_hedge_history": (
                storage.strategy_decision_history(hedge_account_id, 30)
                if hedge_account_id else []
            ),
            "paper_curve": (latest or {}).get("strategy_signal", {}).get(
                "paper_curve"
            ),
            "paper_monitor_account_id": monitor_account_id,
            "paper_monitor": (
                monitor_state.get("last_snapshot") if monitor_state else None
            ),
        },
    }


@app.get("/api/paper-monitor")
async def paper_monitor(limit: int = 400):
    monitor = getattr(pipeline, "sleeve_monitor", None)
    if not monitor:
        return {
            "enabled": False,
            "status": "disabled",
            "message": "独立袖套模拟盘未启用",
            "config": {
                "trend_initial_weight": settings.trend_sleeve_weight,
                "lgbm_initial_weight": settings.lgbm_sleeve_weight,
                "trend_maximum_hedge": settings.trend_hedge_ratio,
            },
            "snapshot": None,
            "history": [],
        }
    state = storage.load_strategy_state(monitor.account_id) or {}
    rows = storage.strategy_decision_history(
        monitor.account_id, min(max(int(limit), 1), 2_000)
    )
    history = []
    for row in reversed(rows):
        sleeves = row.get("sleeves") or {}
        trend = sleeves.get("trend") or {}
        lgbm = sleeves.get("lgbm") or {}
        initial = float(row.get("initial_capital") or settings.paper_initial_capital)
        trend_initial = float(
            trend.get("initial_capital") or initial * settings.trend_sleeve_weight
        )
        lgbm_initial = float(
            lgbm.get("initial_capital") or initial * settings.lgbm_sleeve_weight
        )
        history.append({
            "date": row.get("signal_date"),
            "nav": row.get("nav"),
            "equity": float(row.get("nav", initial)) / initial,
            "drawdown": row.get("drawdown", 0.0),
            "trend_nav": trend.get("nav"),
            "trend_equity": float(trend.get("nav", trend_initial)) / trend_initial,
            "lgbm_nav": lgbm.get("nav"),
            "lgbm_equity": float(lgbm.get("nav", lgbm_initial)) / lgbm_initial,
            "trend_actual_weight": trend.get("actual_weight"),
            "lgbm_actual_weight": lgbm.get("actual_weight"),
            "active_hedge_ratio": trend.get("active_hedge_ratio", 0.0),
            "target_hedge_ratio": trend.get("target_hedge_ratio", 0.0),
        })
    latest = storage.latest() or {}
    curve = (latest.get("strategy_signal") or {}).get("paper_curve")
    return {
        "enabled": True,
        "status": "active" if state.get("last_snapshot") else "awaiting_first_signal",
        "account_id": monitor.account_id,
        "config": {
            "initial_capital": settings.paper_initial_capital,
            "trend_initial_weight": settings.trend_sleeve_weight,
            "lgbm_initial_weight": settings.lgbm_sleeve_weight,
            "trend_maximum_hedge": settings.trend_hedge_ratio,
            "hedge_lookback": settings.index_hedge_lookback,
            "rebalance_days": settings.strategy_rebalance_days,
            "top_n": settings.strategy_top_n,
            "stock_cost_bps_one_way": settings.paper_cost_bps,
            "hedge_change_cost_bps": settings.index_hedge_cost_bps,
            "promotion_required_sessions": 126,
            "forward_start_after": "2026-08-25",
        },
        "snapshot": state.get("last_snapshot"),
        "history": history,
        "curve": curve,
    }


@app.get("/api/factor-mining/latest")
async def latest_factor_mining():
    result_path = BASE_DIR / "data" / "factor_mining" / "latest.json"
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="暂无因子挖掘结果，请先运行 scripts/run_factor_mining.py",
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="因子挖掘结果损坏") from exc


@app.post("/api/factor-mining/update")
async def update_factor_mining():
    from scripts.update_factor_registry import run_auto_update

    try:
        result = await asyncio.to_thread(run_auto_update)
        return {
            "status": result["status"],
            "generated_at": result["generated_at"],
            "registry_version": result["registry_version"],
            "generated_factor_count": result["generated_factor_count"],
            "selected_generated_factor_count": result["selected_generated_factor_count"],
            "production_change": False,
        }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"自动因子更新失败: {exc}") from exc


@app.get("/api/history")
async def history(limit: int = 20):
    return {"items": storage.history(min(max(limit, 1), 50))}


@app.get("/api/backtests")
async def backtests(limit: int = 12):
    return {
        "latest": storage.latest_backtest(),
        "history": storage.backtest_history(min(max(limit, 1), 30)),
        "default_codes": backtest_service.default_codes(),
    }


@app.post("/api/backtests/run")
async def run_backtest(payload: BacktestRequest):
    from datetime import date

    try:
        result = await asyncio.to_thread(
            backtest_service.run,
            payload.mode,
            date.fromisoformat(payload.start_date),
            date.fromisoformat(payload.end_date),
            payload.codes,
            payload.top_n,
            payload.rebalance_days,
            payload.initial_capital,
            payload.cost_bps,
            payload.strategy,
            payload.rebalance_policy,
            payload.min_hold_days,
            payload.rank_buffer,
            payload.score_gap,
            payload.risk_overlay,
            payload.drawdown_limit,
            payload.drawdown_cooldown_days,
        )
        return result
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"回测任务失败: {exc}") from exc


@app.post("/api/run")
async def run_analysis(payload: RunRequest):
    try:
        result = await asyncio.to_thread(pipeline.run, payload.mode, payload.notify)
        return result.to_dict()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"分析任务失败: {exc}") from exc


@app.post("/api/push")
async def push_latest():
    if not settings.configured_channels:
        raise HTTPException(status_code=400, detail="尚未配置飞书、企业微信、通用Webhook或邮件渠道")
    try:
        return {"results": await asyncio.to_thread(pipeline.push_latest)}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/report/latest", response_class=PlainTextResponse)
async def latest_report():
    latest = storage.latest()
    if not latest:
        raise HTTPException(status_code=404, detail="暂无研报")
    return latest["report_markdown"]
