import numpy as np
import pandas as pd

from scripts.compare_current_best_lgbm_ndx_2020_2026 import (
    build_display_curve,
    curve_metrics,
    yearly_returns,
)


def test_display_curve_uses_common_start_and_normalizes_both_series():
    strategy = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    ndx = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]),
    )

    result = build_display_curve(strategy, ndx)

    assert result.index[0] == pd.Timestamp("2020-01-03")
    assert result.index[-1] == pd.Timestamp("2020-01-06")
    np.testing.assert_allclose(result.iloc[0].to_numpy(dtype=float), 1.0)


def test_curve_metrics_reports_cagr_and_drawdown_from_native_observations():
    values = pd.Series(
        [1.0, 1.2, 0.9, 1.4],
        index=pd.to_datetime(["2020-01-02", "2020-04-01", "2020-07-01", "2021-01-02"]),
    )

    metrics = curve_metrics(values)

    assert np.isclose(metrics["total_return"], 0.4)
    assert np.isclose(metrics["terminal_value"], 1.4)
    assert np.isclose(metrics["max_drawdown"], -0.25)
    assert metrics["observations"] == 4


def test_yearly_returns_use_previous_year_end_as_the_next_year_base():
    values = pd.Series(
        [1.0, 1.1, 1.21, 1.331],
        index=pd.to_datetime(["2020-01-02", "2020-12-31", "2021-01-04", "2021-12-31"]),
    )

    result = yearly_returns(values)

    assert np.isclose(result["2020"], 0.10)
    assert np.isclose(result["2021"], 0.21)
