from __future__ import annotations

import pandas as pd

from scripts.research_lgbm_csi300_hedge_v1 import hedge_returns


def test_hedge_signal_is_delayed_until_tradable_open() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    strategy_returns = pd.Series(0.0, index=dates)
    index = pd.DataFrame(
        {
            "open": [100, 100, 100, 90, 90, 90],
            "close": [100, 99, 98, 97, 96, 95],
        },
        index=dates,
    )

    hedged, position = hedge_returns(
        strategy_returns,
        index,
        lookback=2,
        threshold=1.0,
        hedge_ratio=0.5,
    )

    assert position.iloc[0] == 0.0
    assert position.iloc[1] == 0.0
    assert position.iloc[2] == 0.0
    assert position.iloc[3] == 0.5
    assert hedged.iloc[3] > 0.0
