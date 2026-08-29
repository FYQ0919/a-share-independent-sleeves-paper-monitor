import numpy as np
import pandas as pd
import pytest

from app.volatility_target_overlay import VolatilityTargetConfig, VolatilityTargetOverlay


def test_exposure_is_bounded_and_uses_only_prior_returns():
    index = pd.bdate_range("2020-01-02", periods=160)
    returns = pd.Series(np.sin(np.arange(160)) * 0.03, index=index)
    overlay = VolatilityTargetOverlay()
    original = overlay.exposure_for_realized_returns(returns)

    changed = returns.copy()
    changed.iloc[-1] = 0.90
    rerun = overlay.exposure_for_realized_returns(changed)

    pd.testing.assert_series_equal(original, rerun)
    assert original.between(0.65, 1.0).all()


def test_high_lagged_volatility_reaches_floor_without_leverage():
    index = pd.bdate_range("2020-01-02", periods=100)
    returns = pd.Series(np.tile([0.10, -0.10], 50), index=index)
    overlay = VolatilityTargetOverlay()

    exposure = overlay.exposure_for_realized_returns(returns)

    assert exposure.iloc[-1] == pytest.approx(0.65)
    assert overlay.next_session_exposure(returns) == pytest.approx(0.65)


def test_apply_charges_initial_and_change_costs():
    config = VolatilityTargetConfig(
        window=4,
        min_observations=2,
        target_volatility=0.10,
        minimum_exposure=0.50,
        exposure_change_cost_bps=12,
    )
    returns = pd.Series(
        [0.0, 0.20, -0.20, 0.20, -0.20, 0.01],
        index=pd.bdate_range("2020-01-02", periods=6),
    )

    result = VolatilityTargetOverlay(config).apply(returns)

    assert result["overlay_cost"].iloc[0] == pytest.approx(0.0012)
    assert result["risk_exposure"].between(0.50, 1.0).all()
    assert result["managed_equity"].gt(0).all()


def test_target_weights_leave_unallocated_cash():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    scaled = VolatilityTargetOverlay.scale_target_weights(weights, exposure=0.72)

    assert scaled.sum() == pytest.approx(0.72)
    assert scaled["A"] == pytest.approx(0.36)
