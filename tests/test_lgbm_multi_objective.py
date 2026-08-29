import pandas as pd
import pytest

from scripts.research_lgbm_multi_objective_v1 import (
    CANDIDATES,
    RETURN_HURDLE,
    build_multihead_scores,
    multihead_portfolio_backtest,
    opportunity_target,
    validate_candidates,
)


def test_multi_objective_candidate_grid_is_frozen():
    manifest = validate_candidates()

    assert len(manifest) == 64
    assert list(CANDIDATES)[0] == "current_rank_top5"
    assert CANDIDATES["multihead_pareto_top5"]["return_weight"] == pytest.approx(0.025)
    assert CANDIDATES["multihead_pareto_top5"]["safety_weight"] == pytest.approx(0.05)


def test_auxiliary_heads_change_cross_sectional_score_without_raw_value_scaling():
    dates = pd.to_datetime(["2020-01-02"] * 3)
    current = pd.DataFrame(
        {
            "date": dates,
            "code": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    predictions = pd.DataFrame(
        {
            "expected_return": [0.0, 0.01, 0.02],
            "downside_probability": [0.20, 0.10, 0.02],
        }
    )

    output = build_multihead_scores(current, predictions, CANDIDATES["multihead_soft_top5"])

    assert output.loc[2, "score"] > output.loc[2, "score"] * 0
    assert output.loc[2, "score"] > current.loc[2, "score"] * 0.75
    assert output["expected_return"].tolist() == [0.0, 0.01, 0.02]


def test_opportunity_hurdle_can_hold_variable_k_or_cash():
    cross = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D", "E"],
            "score": [5, 4, 3, 2, 1],
            "expected_return": [0.02, 0.01, RETURN_HURDLE + 0.001, 0.0, -0.01],
            "downside_probability": [0.02, 0.03, 0.04, 0.02, 0.01],
        }
    )
    config = CANDIDATES["multihead_opportunity"]

    holdings, _, exposure = opportunity_target(cross, [], config)
    cash_holdings, dropped, cash_exposure = opportunity_target(
        cross.iloc[:2], ["A", "B", "C"], config
    )

    assert holdings == ["A", "B", "C"]
    assert exposure == pytest.approx(1.0)
    assert cash_holdings == []
    assert dropped == ["A", "B", "C"]
    assert cash_exposure == pytest.approx(0.0)


def test_risk_budget_reduces_exposure_but_respects_floor():
    cross = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "score": [3, 2, 1],
            "expected_return": [0.02, 0.02, 0.02],
            "downside_probability": [0.09, 0.09, 0.09],
        }
    )

    holdings, _, exposure = opportunity_target(
        cross, [], CANDIDATES["multihead_risk_budget"]
    )

    assert holdings == ["A", "B", "C"]
    assert exposure == pytest.approx(2 / 3)


def test_signal_at_close_executes_only_at_next_open_and_charges_cost():
    dates = pd.bdate_range("2020-01-02", periods=25)
    codes = [f"S{index}" for index in range(5)]
    history = pd.DataFrame(
        [
            {"date": trading_date, "code": code, "open": 10 + index + day * 0.01}
            for day, trading_date in enumerate(dates)
            for index, code in enumerate(codes)
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "date": trading_date,
                "code": code,
                "score": float(5 - index),
                "expected_return": 0.02,
                "downside_probability": 0.02,
            }
            for trading_date in dates
            for index, code in enumerate(codes)
        ]
    )

    result = multihead_portfolio_backtest(
        history,
        scores,
        CANDIDATES["current_rank_top5"],
        dates[0].date(),
        dates[-1].date(),
    )
    first = result["rebalances"][0]
    returns = pd.DataFrame(result["return_periods"]).set_index("period_start")

    assert first["signal_date"] == dates[0].date().isoformat()
    assert first["execution_date"] == dates[1].date().isoformat()
    assert result["exposure"].loc[dates[0]] == pytest.approx(0.0)
    assert result["exposure"].loc[dates[1]] == pytest.approx(1.0)
    assert returns.loc[dates[1].date().isoformat(), "cost"] > 0
