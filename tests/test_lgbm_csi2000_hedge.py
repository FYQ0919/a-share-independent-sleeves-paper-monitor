import numpy as np
import pandas as pd

from scripts.research_lgbm_csi2000_hedge_v1 import (
    HEDGE_RATIO,
    LOOKBACK,
    apply_csi2000_hedge,
)


def sample_curves(periods: int = 150) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=periods)
    benchmark = pd.Series(np.linspace(1.0, 0.7, periods), index=index)
    return pd.DataFrame(
        {
            "current_best_lgbm": 1.0,
            "csi2000": benchmark,
            "csi300": 1.0,
        },
        index=index,
    )


def test_position_is_lagged_after_trailing_window():
    result = apply_csi2000_hedge(sample_curves())

    assert result["csi2000_hedge_ratio"].iloc[:LOOKBACK].eq(0.0).all()
    assert result["csi2000_hedge_ratio"].iloc[LOOKBACK] == HEDGE_RATIO


def test_future_close_cannot_change_existing_hedge_positions():
    original = sample_curves()
    changed = original.copy()
    changed.iloc[-1, changed.columns.get_loc("csi2000")] *= 10.0

    first = apply_csi2000_hedge(original)["csi2000_hedge_ratio"]
    second = apply_csi2000_hedge(changed)["csi2000_hedge_ratio"]

    pd.testing.assert_series_equal(first, second)
