import numpy as np
import pandas as pd
import pytest

from app.return_decomposition_factors import (
    SPECS,
    add_return_decomposition_factors,
    return_decomposition_factor_metadata,
)


def _frame(days: int = 100, stocks: int = 12) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2017-01-02", periods=days)
    for stock in range(stocks):
        for day, signal_date in enumerate(dates):
            close = (10 + stock) * np.exp(0.0005 * day + 0.02 * np.sin(day / 7 + stock))
            open_price = close * (1 + 0.005 * np.cos(day / 5 + stock))
            rows.append({
                "date": signal_date,
                "code": f"s{stock:02d}",
                "open": open_price,
                "high": max(open_price, close) * 1.01,
                "low": min(open_price, close) * 0.99,
                "close": close,
                "volume": 1_000_000 * (1 + stock / 10) * (1 + 0.1 * np.sin(day / 3)),
                "amount": close * 1_000_000,
            })
    return pd.DataFrame(rows).sample(frac=1.0, random_state=5).reset_index(drop=True)


def test_factor_batch_is_auditable_and_unique():
    assert len(SPECS) == 24
    assert len(SPECS) == len(set(SPECS))
    assert all(return_decomposition_factor_metadata(name) for name in SPECS)


def test_factors_are_bounded_and_preserve_input_order():
    source = _frame()
    output, features = add_return_decomposition_factors(source)
    assert features == list(SPECS)
    pd.testing.assert_frame_equal(output[source.columns], source)
    values = output[features].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert finite.min() >= -1.0 - 1e-12
    assert finite.max() <= 1.0 + 1e-12


def test_future_perturbation_cannot_change_prior_values():
    source = _frame()
    cutoff = pd.Timestamp("2017-04-28")
    original, features = add_return_decomposition_factors(source)
    changed = source.copy()
    future = changed["date"].gt(cutoff)
    changed.loc[future, ["open", "high", "low", "close", "volume"]] *= 7.0
    modified, modified_features = add_return_decomposition_factors(changed)
    assert features == modified_features
    left = original.loc[original["date"].le(cutoff), features]
    right = modified.loc[modified["date"].le(cutoff), features]
    np.testing.assert_allclose(left, right, equal_nan=True)


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="volume"):
        add_return_decomposition_factors(
            pd.DataFrame({"date": ["2020-01-01"], "code": ["A"]})
        )
