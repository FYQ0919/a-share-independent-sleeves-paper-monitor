from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.index_hedge import CSI300HedgePaperService
from app.notifications import NotificationService
from app.pipeline import ResearchPipeline
from app.storage import Storage


MODEL_VERSION = "lgbm_new_factors_blend_v1_candidate"
MODEL_DIR = Path("data/models") / MODEL_VERSION


class StaticIndexProvider:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def load(self, as_of: date):
        selected = self.frame.loc[: pd.Timestamp(as_of)].copy()
        return selected, {
            "mode": "test",
            "index": "CSI300",
            "fetched_through": selected.index[-1].date().isoformat(),
        }


def index_history() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=123)
    frame = pd.DataFrame(
        {
            "open": np.r_[np.full(121, 100.0), 100.0, 90.0],
            "close": np.linspace(120.0, 80.0, 123),
        },
        index=dates,
    )
    return frame


def stock_snapshot(nav: float = 1_000_000.0) -> dict:
    return {"nav": nav, "signal_date": "2026-06-19"}


def test_hedge_signal_ignores_future_index_data(tmp_path):
    history = index_history()
    signal_date = history.index[120].date().isoformat()
    first = CSI300HedgePaperService(
        Storage(tmp_path / "first.db"),
        tmp_path / "unused.csv",
        account_id="hedge_a",
        provider=StaticIndexProvider(history),
    ).update(stock_snapshot(), signal_date)

    changed = history.copy()
    changed.loc[changed.index > pd.Timestamp(signal_date), ["open", "close"]] *= 9.0
    second = CSI300HedgePaperService(
        Storage(tmp_path / "second.db"),
        tmp_path / "unused.csv",
        account_id="hedge_b",
        provider=StaticIndexProvider(changed),
    ).update(stock_snapshot(), signal_date)

    assert first["index_trend_ratio"] == second["index_trend_ratio"]
    assert first["target_hedge_ratio"] == second["target_hedge_ratio"] == 0.50
    assert first["active_hedge_ratio"] == 0.0


def test_pending_hedge_executes_at_next_open_with_cost_and_pnl(tmp_path):
    history = index_history()
    storage = Storage(tmp_path / "paper.db")
    service = CSI300HedgePaperService(
        storage,
        tmp_path / "unused.csv",
        account_id="csi300_hedge_paper",
        provider=StaticIndexProvider(history),
    )
    dates = [history.index[index].date().isoformat() for index in (120, 121, 122)]

    first = service.update(stock_snapshot(), dates[0])
    second = service.update(stock_snapshot(), dates[1])
    third = service.update(stock_snapshot(), dates[2])

    assert first["target_hedge_ratio"] == 0.50
    assert first["active_hedge_ratio"] == 0.0
    assert second["active_hedge_ratio"] == 0.50
    assert second["hedge_cost_today"] == pytest.approx(100.0)
    assert third["hedge_pnl_today"] > 45_000
    assert third["nav"] > 1_045_000
    assert service.update(stock_snapshot(), dates[2]) == third
    assert len(storage.strategy_decision_history(service.account_id)) == 3


def test_pipeline_uses_isolated_csi300_hedged_accounts(tmp_path):
    test_settings = replace(
        settings,
        strategy_model_backend="lgbm_active",
        lgbm_model_version=MODEL_VERSION,
        lgbm_model_dir=MODEL_DIR,
        lgbm_universe_mode="frozen_snapshot",
        independent_sleeves_enabled=False,
        index_hedge_enabled=True,
        database_path=tmp_path / "research.db",
        report_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
    )
    pipeline = ResearchPipeline(test_settings, Storage(test_settings.database_path))

    assert "csi300_hedged" in pipeline.portfolio.strategy_id
    assert "csi300_hedged50" in pipeline.portfolio.strategy_id
    assert "csi300_hedged" in pipeline.paper_account.account_id
    assert "csi300_hedged" in pipeline.index_hedge.account_id
    assert len(
        {
            pipeline.portfolio.strategy_id,
            pipeline.paper_account.account_id,
            pipeline.index_hedge.account_id,
        }
    ) == 3


def test_wecom_message_contains_stock_picks_and_hedge_instruction():
    hedge = {
        "nav": 1_000_000,
        "stock_nav": 1_000_000,
        "hedge_equity": 0,
        "active_hedge_ratio": 0,
        "target_hedge_ratio": 0.75,
        "hedge_pnl_today": 0,
        "index_close": 4609.18,
        "index_ma": 4733.39,
        "index_trend_ratio": 0.9738,
        "action_label": "下一交易日开盘将对冲提高至 75%",
        "parameters": {"lookback": 120},
    }
    message = NotificationService(settings)._compact_message(
        "CSI300对冲LGBM模拟盘｜2026-08-28",
        "最新信号。",
        [],
        [],
        [],
        {
            "index_hedge": hedge,
            "decision": {
                "signal_date": "2026-08-28",
                "action_label": "主调仓",
                "execution_date": "下一交易日开盘",
                "trade_required": True,
                "reason": "固定10日调仓",
                "cycle_day": 0,
                "next_scheduled_in": 10,
                "target_holdings": [
                    {"name": "中国核电", "code": "601985", "target_weight": 0.2}
                ],
                "buys": [
                    {"name": "中国核电", "code": "601985", "target_weight": 0.2}
                ],
                "sells": [],
                "rules": {"rebalance_days": 10},
            },
        },
        {},
    )

    assert "CSI300 指数对冲模拟盘" in message
    assert "下一交易日目标 75%" in message
    assert "中国核电(601985) 20%" in message
    assert "不连接券商" in message
