from dataclasses import replace
from datetime import date

import pytest

from app.backtest_audit import feature_prefix_invariance, future_perturbation_invariance
from app.backtest_data import DemoHistoryProvider
from app.backtest_engine import BacktestConfig, BacktestEngine
from scripts.analyze_strategy import period_summary


def _history_and_config():
    codes = [f"{600000 + index:06d}" for index in range(20)]
    history, warnings = DemoHistoryProvider().load(
        codes, {}, date(2024, 1, 1), date(2025, 12, 31)
    )
    config = BacktestConfig(
        "demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 10,
        1_000_000, 12, "contrarian",
    )
    return history, warnings, config


def test_features_are_prefix_invariant_when_future_rows_are_appended():
    history, _, _ = _history_and_config()
    audit = feature_prefix_invariance(BacktestEngine(), history, date(2025, 6, 30))

    assert audit["passed"] is True
    assert audit["mismatches"] == 0
    assert audit["rows_compared"] > 1_000


def test_future_perturbation_cannot_change_cutoff_scores():
    history, _, _ = _history_and_config()
    weights = BacktestEngine.STRATEGIES["contrarian"]["weights"]
    audit = future_perturbation_invariance(
        BacktestEngine(), history, date(2025, 6, 30), weights
    )

    assert audit["passed"] is True
    assert audit["future_rows_perturbed"] > 0
    assert audit["same_scores"] is True


def test_strict_mode_rejects_current_snapshot_and_current_vintage_qfq():
    history, warnings, config = _history_and_config()

    with pytest.raises(ValueError, match="严格无前视门禁未通过"):
        BacktestEngine().run(
            history, replace(config, strict_no_lookahead=True), warnings
        )


def test_strict_mode_accepts_explicit_point_in_time_contract():
    history, warnings, config = _history_and_config()
    history = history.copy()
    history["universe_member"] = True
    history["available_date"] = history["date"]
    strict = replace(
        config,
        strict_no_lookahead=True,
        universe_policy="point_in_time",
        price_adjustment_policy="raw_point_in_time",
    )

    result = BacktestEngine().run(history, strict, warnings)

    assert result["data"]["lookahead_contract"]["strict_no_lookahead"] is True
    assert result["data"]["lookahead_contract"]["universe_policy"] == "point_in_time"


def test_period_metrics_assign_cross_boundary_return_only_to_end_period():
    result = {
        "return_periods": [
            {
                "period_start": "2023-12-29",
                "period_end": "2024-01-02",
                "strategy_return": 0.10,
                "benchmark_return": 0.04,
            },
            {
                "period_start": "2024-01-02",
                "period_end": "2024-01-03",
                "strategy_return": -0.02,
                "benchmark_return": -0.01,
            },
        ]
    }

    with pytest.raises(ValueError):
        period_summary(result, date(2023, 1, 1), date(2023, 12, 31))
    summary = period_summary(result, date(2024, 1, 1), date(2024, 12, 31))

    assert summary["return_periods"] == 2
    assert summary["total_return"] == pytest.approx(1.10 * 0.98 - 1)
    assert summary["first_period_end"] == "2024-01-02"
