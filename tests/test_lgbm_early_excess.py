from __future__ import annotations

import pandas as pd

from scripts.research_lgbm_early_excess_v1 import (
    curve_metrics,
    portfolio_config,
    summary_objective,
)


def test_portfolio_config_changes_only_target_k() -> None:
    config = portfolio_config(8)

    assert config["min_k"] == 8
    assert config["max_k"] == 8
    assert config["return_weight"] == 0.0
    assert config["safety_weight"] == 0.0


def test_curve_metrics_uses_common_start_and_drawdown() -> None:
    curve = pd.Series(
        [2.0, 2.2, 1.98, 2.42],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
    )

    metrics = curve_metrics(curve)

    assert abs(metrics["terminal_value"] - 1.21) < 1e-12
    assert abs(metrics["max_drawdown"] + 0.10) < 1e-12


def test_objective_rewards_robust_excess_and_penalizes_turnover() -> None:
    def block(annual_return: float, benchmark: float, turnover: float = 10.0) -> dict:
        return {
            "performance": {
                "annual_return": annual_return,
                "benchmark_return": benchmark,
                "sharpe": 1.0,
                "max_drawdown": -0.20,
            },
            "activity": {"annual_turnover": turnover},
        }

    robust = summary_objective(
        block(0.20, 0.10), block(0.18, 0.10), block(0.22, 0.10)
    )
    fragile = summary_objective(
        block(0.20, 0.10), block(0.05, 0.10), block(0.35, 0.10)
    )
    high_turnover = summary_objective(
        block(0.20, 0.10, 30.0), block(0.18, 0.10), block(0.22, 0.10)
    )

    assert robust > fragile
    assert robust > high_turnover
