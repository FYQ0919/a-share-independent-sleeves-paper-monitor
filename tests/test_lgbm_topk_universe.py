import copy

from scripts.research_lgbm_topk_universe_v1 import (
    CANDIDATES,
    candidate_passes,
    validate_candidates,
)


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


def test_topk_universe_grid_is_frozen():
    manifest = validate_candidates()

    assert len(manifest) == 64
    assert [(row["pool_size"], row["top_k"]) for row in CANDIDATES.values()] == [
        (50, 5), (50, 10), (100, 5), (100, 10)
    ]


def test_candidate_must_improve_drawdown_without_sacrificing_return_or_sharpe():
    baseline = _row("pool50_top5", 0.20, -0.30, 1.0, 10.0)
    passing = _row("pool100_top10", 0.195, -0.26, 1.05, 12.0)
    weak_drawdown = copy.deepcopy(passing)
    weak_drawdown["selection"]["performance"]["max_drawdown"] = -0.28
    weak_return = copy.deepcopy(passing)
    weak_return["selection"]["performance"]["annual_return"] = 0.18
    weak_sharpe = copy.deepcopy(passing)
    weak_sharpe["selection"]["performance"]["sharpe"] = 0.95
    high_turnover = copy.deepcopy(passing)
    high_turnover["selection"]["activity"]["annual_turnover"] = 16.0

    assert candidate_passes(passing, baseline)
    assert not candidate_passes(weak_drawdown, baseline)
    assert not candidate_passes(weak_return, baseline)
    assert not candidate_passes(weak_sharpe, baseline)
    assert not candidate_passes(high_turnover, baseline)
