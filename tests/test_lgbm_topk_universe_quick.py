import copy

from scripts.quick_validate_lgbm_topk_universe import annual_priority_passes


def _row(name, annual, drawdown, sharpe, turnover):
    return {
        "name": name,
        "selection": {
            "performance": {
                "annual_return": annual,
                "max_drawdown": drawdown,
                "sharpe": sharpe,
            },
            "activity": {"annual_turnover": turnover},
        },
    }


def test_quick_gate_prioritizes_annual_return_with_bounded_risk():
    baseline = _row("pool50_top5", 0.20, -0.30, 1.0, 10.0)
    candidate = _row("pool100_top10", 0.22, -0.31, 1.02, 12.0)
    assert annual_priority_passes(candidate, baseline)

    low_annual = copy.deepcopy(candidate)
    low_annual["selection"]["performance"]["annual_return"] = 0.19
    low_sharpe = copy.deepcopy(candidate)
    low_sharpe["selection"]["performance"]["sharpe"] = 0.99
    bad_drawdown = copy.deepcopy(candidate)
    bad_drawdown["selection"]["performance"]["max_drawdown"] = -0.321

    assert not annual_priority_passes(low_annual, baseline)
    assert not annual_priority_passes(low_sharpe, baseline)
    assert not annual_priority_passes(bad_drawdown, baseline)
