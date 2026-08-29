from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.compare_latest_hedged_lgbm_indices_2020_2026 import (
    curve_metrics,
    excess_metrics,
    normalize_common_curves,
)


def test_normalize_common_curves_uses_date_intersection_and_one_start() -> None:
    strategy_dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    benchmark_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"])
    strategy = pd.Series([2.0, 4.0, 8.0], index=strategy_dates)
    benchmark = pd.Series([10.0, 20.0, 40.0], index=benchmark_dates)

    curves = normalize_common_curves(strategy, {"benchmark": benchmark})

    assert curves.index.tolist() == list(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    assert curves.iloc[0].tolist() == [1.0, 1.0]
    assert curves.iloc[-1].tolist() == [2.0, 2.0]


def test_curve_metrics_uses_elapsed_calendar_time_for_cagr() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"])
    curve = pd.Series([1.0, 1.5, 2.0], index=dates)

    result = curve_metrics(curve)

    expected_years = 366.0 / 365.2425
    expected_cagr = 2.0 ** (1.0 / expected_years) - 1.0
    assert np.isclose(result["total_return"], 1.0)
    assert np.isclose(result["elapsed_calendar_years"], expected_years)
    assert np.isclose(result["annual_return"], expected_cagr)


def test_excess_metrics_relative_wealth_and_beta_formulas() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    benchmark_returns = pd.Series([0.10, -0.05, 0.02, 0.04], index=dates[1:])
    strategy_returns = benchmark_returns.mul(2.0)
    benchmark = pd.concat(
        [pd.Series([1.0], index=dates[:1]), (1.0 + benchmark_returns).cumprod()]
    )
    strategy = pd.concat(
        [pd.Series([1.0], index=dates[:1]), (1.0 + strategy_returns).cumprod()]
    )

    result = excess_metrics(strategy, benchmark)

    expected_relative_wealth = float(strategy.iloc[-1] / benchmark.iloc[-1])
    assert np.isclose(result["relative_wealth"], expected_relative_wealth)
    assert np.isclose(result["relative_cumulative_return"], expected_relative_wealth - 1.0)
    assert np.isclose(result["beta"], 2.0)
    assert np.isclose(result["annualized_jensen_alpha_zero_rate"], 0.0, atol=1e-12)
