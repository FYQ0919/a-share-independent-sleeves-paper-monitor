import numpy as np
import pandas as pd
import pytest

from scripts.optimize_lgbm_drawdown_v1 import (
    COST_RATE,
    apply_overlay,
    lagged_annualized_volatility,
)


def test_lagged_volatility_does_not_use_same_day_or_future_returns():
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    original = lagged_annualized_volatility(returns, 60)

    changed = returns.copy()
    changed.iloc[60:] = 0.50
    recalculated = lagged_annualized_volatility(changed, 60)

    assert recalculated.iloc[59] == pytest.approx(original.iloc[59])
    assert recalculated.iloc[60] == pytest.approx(original.iloc[60])
    assert recalculated.iloc[61] != pytest.approx(original.iloc[61])


def test_volatility_overlay_applies_floor_and_exposure_change_cost():
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    frame = pd.DataFrame(
        {
            "base_return": [0.01, 0.02, -0.03, 0.04],
            "lagged_vol_60": [np.nan, 0.44, 0.11, 0.88],
            "pool_trend_120": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    curve = apply_overlay(
        frame,
        {"vol_window": 60, "target_vol": 0.22, "floor": 0.65},
    )

    expected_exposure = pd.Series([1.0, 0.65, 1.0, 0.65], index=index)
    expected_turnover = pd.Series([1.0, 0.35, 0.35, 0.35], index=index)
    expected_return = expected_exposure * frame["base_return"] - expected_turnover * COST_RATE
    pd.testing.assert_series_equal(
        curve["exposure"], expected_exposure, check_names=False, check_freq=False
    )
    pd.testing.assert_series_equal(
        curve["exposure_turnover"], expected_turnover, check_names=False, check_freq=False
    )
    pd.testing.assert_series_equal(
        curve["return"], expected_return, check_names=False, check_freq=False
    )
