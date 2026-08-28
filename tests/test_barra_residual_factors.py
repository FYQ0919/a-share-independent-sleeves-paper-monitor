import numpy as np
import pandas as pd
import pytest

from app.barra_residual_factors import (
    BARRA_RESIDUAL_FACTOR_SPECS,
    SPECS,
    add_barra_residual_factors,
    barra_residual_factor_metadata,
)


def _daily_frame(days: int = 150, symbols: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-03", periods=days)
    rows: list[dict[str, object]] = []
    for code_index in range(symbols):
        close = 20.0 + 2.5 * code_index
        for day_index, date in enumerate(dates):
            market_wave = 0.004 * np.sin(day_index / 9.0)
            idio_wave = 0.0025 * np.cos(day_index / (4.0 + code_index / 2.0))
            drift = 0.0004 * (code_index - symbols / 2.0)
            daily_return = market_wave + idio_wave + drift
            open_price = close * (1.0 + 0.002 * np.cos(day_index / 5.0))
            close = close * (1.0 + daily_return)
            high = max(open_price, close) * (
                1.0 + 0.005 + 0.0004 * (code_index % 4)
            )
            low = min(open_price, close) * (
                1.0 - 0.005 - 0.0003 * (code_index % 3)
            )
            amount = (8_000_000.0 + code_index * 250_000.0) * (
                1.0 + 0.25 * np.sin(day_index / 12.0 + code_index)
            )
            rows.append(
                {
                    "code": f"600{code_index:03d}",
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": amount / close,
                    "amount": amount,
                    "untouched": f"row-{code_index}-{day_index}",
                }
            )
    # Deliberately exercise stable sorting independently of input order.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=23).reset_index(drop=True)


def test_layer_contract_is_auditable_and_has_fifteen_new_factors():
    frame = _daily_frame()
    frozen = frame.copy(deep=True)
    extended, features = add_barra_residual_factors(frame)

    assert SPECS is BARRA_RESIDUAL_FACTOR_SPECS
    assert len(BARRA_RESIDUAL_FACTOR_SPECS) == 15
    assert features == list(BARRA_RESIDUAL_FACTOR_SPECS)
    assert set(features).issubset(extended.columns)
    assert set(frame.columns).issubset(extended.columns)
    assert frame.equals(frozen)
    for feature, spec in BARRA_RESIDUAL_FACTOR_SPECS.items():
        assert barra_residual_factor_metadata(feature) == spec
        assert set(spec) == {"family", "origin", "formula"}
        assert spec["family"] in {
            "residual_risk",
            "downside_risk",
            "transaction_impact",
            "market_state_sensitivity",
        }
    assert barra_residual_factor_metadata("x_unknown") is None


def test_output_is_stably_sorted_ranked_and_finite_or_missing():
    extended, features = add_barra_residual_factors(_daily_frame())

    assert extended[["code", "date"]].equals(
        extended[["code", "date"]].sort_values(
            ["code", "date"], kind="mergesort"
        )
    )
    values = extended[features].to_numpy(dtype=float)
    assert (np.isfinite(values) | np.isnan(values)).all()
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert finite.min() >= -1.0
    assert finite.max() <= 1.0
    assert extended[features].notna().any().all()


def test_future_market_perturbation_does_not_change_prior_factor_values():
    frame = _daily_frame()
    cutoff = pd.Timestamp("2023-06-30")
    original, features = add_barra_residual_factors(frame)

    changed_frame = frame.copy(deep=True)
    future = changed_frame["date"].gt(cutoff)
    changed_frame.loc[future, ["open", "high", "low", "close"]] *= 7.0
    changed_frame.loc[future, ["volume", "amount"]] *= 11.0
    changed, changed_features = add_barra_residual_factors(changed_frame)

    assert features == changed_features
    left = original[original["date"].le(cutoff)].sort_values(
        ["code", "date"], kind="mergesort"
    )
    right = changed[changed["date"].le(cutoff)].sort_values(
        ["code", "date"], kind="mergesort"
    )
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_invalid_values_are_safe_and_missing_columns_are_reported():
    frame = _daily_frame(days=90, symbols=6)
    frame.loc[0, ["close", "amount"]] = [np.inf, 0.0]
    frame.loc[1, ["open", "high", "low", "volume"]] = [np.nan, np.inf, -1.0, 0.0]
    extended, features = add_barra_residual_factors(frame)
    values = extended[features].to_numpy(dtype=float)
    assert (np.isfinite(values) | np.isnan(values)).all()

    with pytest.raises(ValueError, match="amount"):
        add_barra_residual_factors(frame.drop(columns=["amount"]))

