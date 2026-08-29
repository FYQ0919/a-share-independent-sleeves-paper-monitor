import pandas as pd
import pytest

from scripts.research_lgbm_dynamic_k_cash_v1 import (
    adjusted_holdings,
    dynamic_portfolio_backtest,
    validate_candidates,
)


def test_dynamic_candidate_grid_is_frozen_and_valid():
    manifest = validate_candidates()

    assert len(manifest) == 64


def test_adjusted_holdings_expands_contracts_and_limits_normal_dropout():
    cross = pd.Series(
        range(10, 0, -1),
        index=[f"6000{index:02d}" for index in range(10)],
        dtype=float,
    )
    planned = cross.index[:5].tolist()

    expanded, expanded_drops = adjusted_holdings(cross, planned, 10)
    contracted, contracted_drops = adjusted_holdings(cross, planned, 3)
    reranked_cross = cross.copy()
    reranked_cross.loc["600005"] = 20.0
    reranked, reranked_drops = adjusted_holdings(reranked_cross, planned, 5, n_drop=1)

    assert expanded == cross.index.tolist()
    assert expanded_drops == []
    assert contracted == planned[:3]
    assert contracted_drops == planned[3:]
    assert len(reranked) == 5
    assert "600005" in reranked
    assert len(reranked_drops) == 1


def test_risk_transition_to_cash_executes_next_open_and_charges_cost():
    dates = pd.bdate_range("2020-01-02", periods=25)
    codes = [f"6000{index:02d}" for index in range(6)]
    history = pd.DataFrame([
        {
            "date": signal_date,
            "code": code,
            "open": 10.0 + code_index + date_index * 0.02,
        }
        for date_index, signal_date in enumerate(dates)
        for code_index, code in enumerate(codes)
    ])
    scores = pd.DataFrame([
        {
            "date": signal_date,
            "code": code,
            "score": float(len(codes) - code_index),
        }
        for signal_date in dates
        for code_index, code in enumerate(codes)
    ])
    regime = pd.DataFrame(
        {"market_stress": [False] * 5 + [True] * 20}, index=dates
    )
    config = {
        "normal_k": 5,
        "stress_k": 0,
        "normal_exposure": 1.0,
        "stress_exposure": 0.0,
        "risk_rebalance": True,
    }

    result = dynamic_portfolio_backtest(
        history,
        scores,
        regime,
        config,
        dates[0].date(),
        dates[-1].date(),
    )
    exit_record = next(
        row for row in result["rebalances"] if row["trigger"] == "risk_transition"
    )
    returns = pd.DataFrame(result["return_periods"]).set_index("period_start")

    assert exit_record["signal_date"] == dates[5].date().isoformat()
    assert exit_record["execution_date"] == dates[6].date().isoformat()
    assert exit_record["target_k"] == 0
    assert exit_record["target_exposure"] == pytest.approx(0.0)
    assert result["exposure"].loc[dates[5]] > 0
    assert result["exposure"].loc[dates[6]] == pytest.approx(0.0)
    assert returns.loc[dates[6].date().isoformat(), "cost"] > 0
