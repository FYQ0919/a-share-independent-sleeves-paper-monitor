from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Dict

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _lgbm_model_dir() -> Path:
    configured = os.getenv("LGBM_MODEL_DIR", "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path
    version = os.getenv("LGBM_MODEL_VERSION", "lgbm_ranker_top5_v1").strip()
    return BASE_DIR / "data" / "models" / version


@dataclass(frozen=True)
class Settings:
    data_mode: str = os.getenv("DATA_MODE", "demo").lower()
    top_n: int = _int("TOP_N", 50)
    technology_reserve: int = _int("TECHNOLOGY_RESERVE", 10)
    min_amount: float = _float("MIN_AMOUNT", 50_000_000)
    min_market_cap: float = _float("MIN_MARKET_CAP", 2_000_000_000)
    max_sectors: int = _int("MAX_SECTORS", 10)
    schedule_enabled: bool = _bool("SCHEDULE_ENABLED", True)
    schedule_hour: int = _int("SCHEDULE_HOUR", 18)
    schedule_minute: int = _int("SCHEDULE_MINUTE", 0)
    push_on_schedule: bool = _bool("PUSH_ON_SCHEDULE", True)
    factor_auto_update_enabled: bool = _bool("FACTOR_AUTO_UPDATE_ENABLED", True)
    factor_update_day: int = _int("FACTOR_UPDATE_DAY", 1)
    factor_update_hour: int = _int("FACTOR_UPDATE_HOUR", 19)
    factor_update_minute: int = _int("FACTOR_UPDATE_MINUTE", 30)
    strategy_push_enabled: bool = _bool("STRATEGY_PUSH_ENABLED", True)
    strategy_top_n: int = _int("STRATEGY_TOP_N", 5)
    strategy_rebalance_days: int = _int("STRATEGY_REBALANCE_DAYS", 10)
    strategy_min_hold_days: int = _int("STRATEGY_MIN_HOLD_DAYS", 5)
    strategy_rank_buffer: int = _int("STRATEGY_RANK_BUFFER", 20)
    strategy_score_gap: float = _float("STRATEGY_SCORE_GAP", 15.0)
    strategy_model_backend: str = os.getenv("STRATEGY_MODEL_BACKEND", "contrarian").lower()
    lgbm_model_version: str = os.getenv(
        "LGBM_MODEL_VERSION", "lgbm_ranker_top5_v1"
    ).strip()
    lgbm_universe_mode: str = os.getenv("LGBM_UNIVERSE_MODE", "dynamic").lower()
    lgbm_history_days: int = _int("LGBM_HISTORY_DAYS", 550)
    paper_initial_capital: float = _float("PAPER_INITIAL_CAPITAL", 1_000_000)
    paper_cost_bps: float = _float("PAPER_COST_BPS", 12.0)
    paper_lot_size: int = _int("PAPER_LOT_SIZE", 100)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    feishu_webhook_url: str = os.getenv("FEISHU_WEBHOOK_URL", "")
    wecom_webhook_url: str = os.getenv("WECOM_WEBHOOK_URL", "")
    generic_webhook_url: str = os.getenv("GENERIC_WEBHOOK_URL", "")
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = _int("SMTP_PORT", 465)
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_from: str = os.getenv("EMAIL_FROM", "")
    email_to: str = os.getenv("EMAIL_TO", "")
    database_path: Path = BASE_DIR / "data" / "research.db"
    report_dir: Path = BASE_DIR / "reports"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    lgbm_model_dir: Path = field(default_factory=_lgbm_model_dir)
    factor_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.28,
            "sector": 0.22,
            "liquidity": 0.18,
            "value": 0.17,
            "risk": 0.15,
        }
    )

    @property
    def configured_channels(self):
        channels = []
        if self.feishu_webhook_url:
            channels.append("飞书")
        if self.wecom_webhook_url:
            channels.append("企业微信")
        if self.generic_webhook_url:
            channels.append("通用 Webhook")
        if self.smtp_host and self.email_to:
            channels.append("邮件")
        return channels


settings = Settings()
