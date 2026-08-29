import numpy as np
import pandas as pd
import pytest

from scripts.research_lgbm_dynamic_frequency_v1 import (
    CANDIDATES,
    build_causal_volatility_regime,
    dynamic_frequency_backtest,
    validate_candidates,
)
from scripts.research_qlib_ridge_topk_v1 import topk_dropout_backtest


def _market(periods: int = 45, codes: int = 8):
    dates = pd.bdate_range("2020-01-02", periods=periods)
    names = [f"6000{index:02d}" for index in range(codes)]
    history = pd.DataFrame([
        {
            "date": signal_date,
            "code": code,
            "open": 10.0 + code_index + date_index * (0.01 + code_index * 0.001),
            "close": 10.01 + code_index + date_index * (0.01 + code_index * 0.001),
        }
        for date_index, signal_date in enumerate(dates)
        for code_index, code in enumerate(names)
    ])
    scores = pd.DataFrame([
        {"date": signal_date, "code": code, "score": float(codes - code_index) / codes}
        for signal_date in dates
        for code_index, code in enumerate(names)
    ])
    regime = pd.DataFrame({"volatility_regime": "normal"}, index=dates)
    return dates, names, history, scores, regime


def test_dynamic_frequency_candidate_grid_is_frozen():
    assert len(validate_candidates()) == 64


def test_high_vol_shortens_and_calm_extends_schedule():
    dates, _, history, scores, regime = _market(periods=50)
    regime.loc[:, "volatility_regime"] = "high"
    high = dynamic_frequency_backtest(
        history, scores, regime, CANDIDATES["vol_5_10_15"], dates[0].date(), dates[-1].date()
    )
    regime.loc[:, "volatility_regime"] = "calm"
    calm = dynamic_frequency_backtest(
        history, scores, regime, CANDIDATES["vol_5_10_15"], dates[0].date(), dates[-1].date()
    )

    assert high["rebalances"][1]["elapsed_signal_sessions"] == 5
    assert high["rebalances"][1]["trigger"] == "high_vol"
    assert calm["rebalances"][1]["elapsed_signal_sessions"] == 15
    assert calm["rebalances"][1]["trigger"] == "max_interval"


def test_rank_deterioration_triggers_only_after_minimum_hold():
    dates, names, history, scores, regime = _market(periods=30)
    for date_index, signal_date in enumerate(dates):
        if date_index >= 3:
            scores.loc[
                (scores["date"].eq(signal_date)) & scores["code"].eq(names[5]), "score"
            ] = 2.0
            scores.loc[
                (scores["date"].eq(signal_date)) & scores["code"].eq(names[4]), "score"
            ] = 0.0
    result = dynamic_frequency_backtest(
        history, scores, regime, CANDIDATES["rank_5_15"], dates[0].date(), dates[-1].date()
    )

    trigger = result["rebalances"][1]
    assert trigger["trigger"] == "rank_deterioration"
    assert trigger["elapsed_signal_sessions"] == 5
    assert names[5] in trigger["holdings"]


def test_signal_executes_next_open_and_cost_is_not_charged_at_close():
    dates, _, history, scores, regime = _market(periods=25)
    result = dynamic_frequency_backtest(
        history, scores, regime, CANDIDATES["fixed_10"], dates[0].date(), dates[-1].date()
    )
    first = result["rebalances"][0]
    periods = pd.DataFrame(result["return_periods"]).set_index("period_start")

    assert first["signal_date"] == dates[0].date().isoformat()
    assert first["execution_date"] == dates[1].date().isoformat()
    assert periods.loc[dates[0].date().isoformat(), "turnover"] == pytest.approx(0.0)
    assert periods.loc[dates[0].date().isoformat(), "cost"] == pytest.approx(0.0)
    assert periods.loc[dates[1].date().isoformat(), "turnover"] == pytest.approx(1.0)
    assert periods.loc[dates[1].date().isoformat(), "cost"] == pytest.approx(0.0012)


def test_future_price_perturbation_cannot_change_earlier_volatility_schedule():
    dates, _, history, scores, _ = _market(periods=420)
    cutoff = dates[320]
    original_regime = build_causal_volatility_regime(history)
    changed = history.copy()
    future = changed["date"].gt(cutoff)
    changed.loc[future, "close"] *= np.linspace(1.0, 4.0, int(future.sum()))
    changed_regime = build_causal_volatility_regime(changed)

    pd.testing.assert_frame_equal(
        original_regime.loc[:cutoff],
        changed_regime.loc[:cutoff],
    )
    original_result = dynamic_frequency_backtest(
        history,
        scores,
        original_regime,
        CANDIDATES["vol_5_10_15"],
        dates[0].date(),
        dates[-1].date(),
    )
    changed_result = dynamic_frequency_backtest(
        changed,
        scores,
        changed_regime,
        CANDIDATES["vol_5_10_15"],
        dates[0].date(),
        dates[-1].date(),
    )
    original_schedule = [
        row for row in original_result["rebalances"]
        if pd.Timestamp(row["signal_date"]) <= cutoff
    ]
    changed_schedule = [
        row for row in changed_result["rebalances"]
        if pd.Timestamp(row["signal_date"]) <= cutoff
    ]
    assert changed_schedule == original_schedule


def test_fixed_10_reproduces_current_top5_backtest():
    dates, _, history, scores, regime = _market(periods=45)
    dynamic = dynamic_frequency_backtest(
        history, scores, regime, CANDIDATES["fixed_10"], dates[0].date(), dates[-1].date()
    )
    reference = topk_dropout_backtest(
        history, scores, dates[0].date(), dates[-1].date()
    )
    dynamic_periods = pd.DataFrame(dynamic["return_periods"])
    reference_periods = pd.DataFrame(reference["return_periods"])

    np.testing.assert_allclose(
        dynamic_periods["strategy_return"], reference_periods["strategy_return"], atol=1e-12
    )
    np.testing.assert_allclose(
        dynamic_periods["turnover"], reference_periods["turnover"], atol=1e-12
    )
    assert len(dynamic["rebalances"]) == len(reference["rebalances"])
