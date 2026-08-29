from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.models import StrategyPick
from app.sleeve_monitor import IndependentSleeveMonitorService
from app.storage import Storage
from app.trend_expert import EQUIPMENT_CODES, TrendExpertRanker, TrendPortfolioService


def _history(days=140):
    start = date(2026, 1, 2)
    dates = pd.bdate_range(start, periods=days)
    codes = sorted(EQUIPMENT_CODES) + [
        "600001", "600002", "600003", "600004", "600005", "600006"
    ]
    rows = []
    for code_number, code in enumerate(codes):
        equipment = code in EQUIPMENT_CODES
        drift = 0.004 if equipment else 0.0004 + code_number * 0.00005
        for index, day in enumerate(dates):
            close = 20.0 * (1.0 + drift) ** index
            amount = 100_000_000 * (1.0 + (0.004 if equipment else 0.0002) * index)
            rows.append({
                "date": day,
                "code": code,
                "name": code,
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": amount / close,
                "amount": amount,
                "turnover": 2.0,
                "pe": 25.0,
                "pb": 3.0,
                "tradestatus": 1,
                "is_st": 0,
            })
    return pd.DataFrame(rows)


def _base_rankings():
    codes = sorted(EQUIPMENT_CODES) + [
        "600001", "600002", "600003", "600004", "600005", "600006"
    ]
    return [
        StrategyPick(
            rank=index,
            code=code,
            name=code,
            score=100 - index,
            price=20.0,
            target_weight=0.2,
            factor_scores={
                "LGBM排名": 100 - index * 5,
                "基础策略排名": 90 - index * 4,
            },
            reason="test",
        )
        for index, code in enumerate(codes, start=1)
    ]


def test_trend_ranker_uses_only_signal_date_and_detects_equipment_regime():
    history = _history()
    signal_date = pd.Timestamp(history["date"].max()).date()
    ranker = TrendExpertRanker()
    original, metadata = ranker.rank(_base_rankings(), history, signal_date)
    assert metadata["strong_equipment"] is True
    assert metadata["maximum_replacements"] == 2

    future = history.copy()
    future_rows = future[future["date"].eq(future["date"].max())].copy()
    future_rows["date"] = pd.Timestamp(signal_date + timedelta(days=3))
    future_rows[["open", "high", "low", "close"]] *= 100.0
    changed, changed_metadata = ranker.rank(
        _base_rankings(), pd.concat([future, future_rows], ignore_index=True), signal_date
    )
    assert [item.code for item in changed] == [item.code for item in original]
    assert [item.score for item in changed] == [item.score for item in original]
    assert changed_metadata["strong_equipment"] is True


def test_trend_portfolio_is_idempotent_and_limits_replacements(tmp_path):
    storage = Storage(tmp_path / "paper.db")
    service = TrendPortfolioService(storage)
    dates = [date(2026, 8, 1) + timedelta(days=index) for index in range(30)]
    rankings = _base_rankings()
    first = service.evaluate(rankings, dates[0], dates, strong_equipment=False)
    same = service.evaluate(rankings, dates[0], dates, strong_equipment=True)
    assert same == first
    assert first["trade_required"] is True

    hold = service.evaluate(rankings, dates[1], dates, strong_equipment=False)
    assert hold["executed_order"] is not None
    rotated = rankings[2:] + rankings[:2]
    for index, item in enumerate(rotated, start=1):
        item.rank = index
        item.score = 100 - index
    decision = service.evaluate(rotated, dates[11], dates, strong_equipment=True)
    assert decision["trade_required"] is True
    assert len(decision["sells"]) <= 2
    assert len(decision["buys"]) <= 2


def test_independent_monitor_never_rebalances_between_sleeves(tmp_path):
    storage = Storage(tmp_path / "paper.db")
    monitor = IndependentSleeveMonitorService(
        storage, initial_capital=1_000_000, trend_initial_weight=0.75,
        lgbm_initial_weight=0.25,
    )
    trend_stock = {
        "nav": 825_000, "cash": 25_000, "exposure": 0.95,
        "positions": [], "trades": [], "transaction_cost_total": 1_000,
    }
    trend_hedged = {
        "nav": 810_000, "hedge_cost_total": 200, "active_hedge_ratio": 0.75,
        "target_hedge_ratio": 0.75, "action_label": "维持对冲",
        "index_close": 4000, "index_ma": 4100,
    }
    lgbm = {
        "nav": 275_000, "cash": 15_000, "exposure": 0.94,
        "positions": [], "trades": [], "transaction_cost_total": 300,
    }
    snapshot = monitor.update(
        signal_date="2026-08-28", trend_stock=trend_stock,
        trend_hedged=trend_hedged, trend_decision={}, trend_regime={},
        lgbm_stock=lgbm, lgbm_decision={},
    )
    assert snapshot["nav"] == pytest.approx(1_085_000)
    assert snapshot["cash_transfer_between_sleeves"] == 0.0
    assert snapshot["sleeves"]["trend"]["nav"] == 810_000
    assert snapshot["sleeves"]["lgbm"]["nav"] == 275_000
    assert snapshot["sleeves"]["trend"]["actual_weight"] == pytest.approx(
        810_000 / 1_085_000
    )
    assert monitor.update(
        signal_date="2026-08-28", trend_stock=trend_stock,
        trend_hedged=trend_hedged, trend_decision={}, trend_regime={},
        lgbm_stock=lgbm, lgbm_decision={},
    ) == snapshot
