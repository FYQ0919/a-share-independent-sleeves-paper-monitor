import pandas as pd
import pytest

from scripts.research_lgbm_growth_control_v1 import blend_growth_control


def _inputs():
    dates = pd.bdate_range("2020-01-02", periods=4)
    base = pd.Series([0.01, -0.02, 0.03, -0.01], index=dates)
    overlay = pd.DataFrame(
        {
            "return": [0.005, -0.01, 0.015, -0.005],
            "exposure": [0.5, 0.5, 0.5, 0.5],
        },
        index=dates,
    )
    return base, overlay


def test_growth_control_blends_returns_and_exposure_linearly():
    base, overlay = _inputs()

    result = blend_growth_control(base, overlay, 0.25)

    assert result.iloc[0]["strategy_return"] == pytest.approx(0.00875)
    assert result.iloc[0]["exposure"] == pytest.approx(0.875)
    assert result.iloc[-1]["equity"] == pytest.approx(
        (1 + result["strategy_return"]).prod()
    )


def test_zero_and_full_overlay_weights_match_inputs():
    base, overlay = _inputs()

    zero = blend_growth_control(base, overlay, 0.0)
    full = blend_growth_control(base, overlay, 1.0)

    pd.testing.assert_series_equal(zero["strategy_return"], base, check_names=False)
    pd.testing.assert_series_equal(full["strategy_return"], overlay["return"], check_names=False)


def test_overlay_weight_is_bounded():
    base, overlay = _inputs()

    with pytest.raises(ValueError, match="覆盖层权重"):
        blend_growth_control(base, overlay, 1.1)
