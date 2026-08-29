from scripts.research_lgbm_high_annual_v1 import historical_gate


def test_high_annual_gate_requires_return_and_risk_constraints():
    current = {"annual_return": 0.30, "max_drawdown": -0.30, "sharpe": 1.0}
    candidate = {"annual_return": 0.33, "max_drawdown": -0.32, "sharpe": 1.05}
    activity = {"annual_turnover": 10.5}
    current_activity = {"annual_turnover": 10.0}

    passed = historical_gate(candidate, current, activity, current_activity)
    failed = historical_gate(
        {**candidate, "annual_return": 0.29}, current, activity, current_activity
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["annual_return_higher"] is False
