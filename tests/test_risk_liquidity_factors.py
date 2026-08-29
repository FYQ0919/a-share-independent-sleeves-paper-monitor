import numpy as np
import pandas as pd

from app.risk_liquidity_factors import (
    RISK_LIQUIDITY_FACTOR_SPECS,
    add_risk_liquidity_factors,
    risk_liquidity_factor_metadata,
)


def _daily_frame(days: int = 120, symbols: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-03", periods=days)
    rows = []
    for code_index in range(symbols):
        close = 20.0 + code_index
        for day_index, date in enumerate(dates):
            # Distinct but deterministic paths give each date a usable cross
            # section while keeping all values positive and OHLC-consistent.
            daily_return = (
                0.0015 * np.sin(day_index / 7.0 + code_index / 3.0)
                + 0.0006 * (code_index - symbols / 2.0)
            )
            open_price = close * (1.0 + 0.001 * np.cos(day_index / 5.0))
            close = close * (1.0 + daily_return)
            high = max(open_price, close) * (1.0 + 0.004 + 0.001 * (code_index % 3))
            low = min(open_price, close) * (1.0 - 0.004 - 0.0005 * (code_index % 2))
            amount = (8_000_000.0 + code_index * 220_000.0) * (
                1.0 + 0.18 * np.sin(day_index / 11.0 + code_index)
            )
            rows.append(
                {
                    "code": f"{600000 + code_index:06d}",
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": amount / close,
                    "amount": amount,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=17).reset_index(drop=True)


def test_factor_layer_has_auditable_specs_and_expected_column_count():
    frame = _daily_frame()
    extended, features = add_risk_liquidity_factors(frame)

    assert len(RISK_LIQUIDITY_FACTOR_SPECS) == 14
    assert len(features) == 14
    assert set(features) == set(RISK_LIQUIDITY_FACTOR_SPECS)
    assert set(features).issubset(extended.columns)
    assert set(frame.columns).issubset(extended.columns)
    for feature, metadata in RISK_LIQUIDITY_FACTOR_SPECS.items():
        assert risk_liquidity_factor_metadata(feature) == metadata
        assert {"family", "origin", "formula"}.issubset(metadata)
    assert risk_liquidity_factor_metadata("x_unknown") is None


def test_ranked_factor_values_are_finite_and_bounded():
    extended, features = add_risk_liquidity_factors(_daily_frame())
    values = extended[features].to_numpy(dtype=float)
    finite_or_missing = np.isfinite(values) | np.isnan(values)

    assert finite_or_missing.all()
    observed = values[np.isfinite(values)]
    assert len(observed) > 0
    assert observed.min() >= -1.0
    assert observed.max() <= 1.0


def test_future_market_perturbations_do_not_change_prior_factor_values():
    frame = _daily_frame()
    cutoff = pd.Timestamp("2023-05-31")
    original, features = add_risk_liquidity_factors(frame)

    perturbed_frame = frame.copy()
    future = perturbed_frame["date"].gt(cutoff)
    perturbed_frame.loc[future, ["open", "high", "low", "close"]] *= 4.0
    perturbed_frame.loc[future, ["volume", "amount"]] *= 6.0
    perturbed, perturbed_features = add_risk_liquidity_factors(perturbed_frame)

    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = perturbed[perturbed["date"].le(cutoff)].sort_values(["code", "date"])
    assert features == perturbed_features
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_missing_ohlcv_inputs_are_rejected():
    frame = _daily_frame().drop(columns=["amount"])
    try:
        add_risk_liquidity_factors(frame)
    except ValueError as error:
        assert "amount" in str(error)
    else:
        raise AssertionError("missing amount should be rejected")

