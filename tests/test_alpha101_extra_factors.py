from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.alpha101_extra_factors import (
    ALPHA101_EXTRA_FACTOR_SPECS,
    SPECS,
    add_alpha101_extra_factors,
    alpha101_extra_factor_metadata,
)
from app.framework_factors import FRAMEWORK_FACTOR_SPECS
from app.qlib_extra_factors import QLIB_EXTRA_FACTOR_SPECS
from app.risk_liquidity_factors import RISK_LIQUIDITY_FACTOR_SPECS


def _input_frame(days: int = 140, symbols: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-03", periods=days)
    rows: list[dict[str, object]] = []
    for code_index in range(symbols):
        close = 18.0 + code_index * 2.5
        for day_index, date in enumerate(dates):
            daily_return = (
                0.0012 * np.sin(day_index / 8.0 + code_index / 4.0)
                + 0.0004 * (code_index - symbols / 2.0)
            )
            # The cross-sectional ordering changes over time, so correlation
            # factors have non-degenerate windows in this fixture.
            open_price = close * (
                1.0
                + 0.20 * np.sin(day_index / 6.0 + code_index * 1.7)
                + 0.003 * np.cos(day_index / 4.0 + code_index)
            )
            close = close * (1.0 + daily_return)
            high = max(open_price, close) * (1.0 + 0.006 + 0.0005 * code_index)
            low = min(open_price, close) * (1.0 - 0.006 - 0.0003 * code_index)
            volume = (900_000.0 + 14_000.0 * day_index) * (
                1.0 + 0.08 * code_index + 0.12 * np.sin(day_index / 10.0 + code_index)
            )
            amount = volume * close * (1.0 + 0.01 * np.cos(day_index + code_index))
            rows.append(
                {
                    "code": f"600{code_index:03d}",
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "untouched": f"{code_index}-{day_index}",
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=23).reset_index(drop=True)


def test_alpha101_specs_are_auditable_and_unique_from_existing_layers():
    assert SPECS is ALPHA101_EXTRA_FACTOR_SPECS
    assert 8 <= len(SPECS) <= 15
    assert len(SPECS) == 12
    assert len(set(SPECS)) == len(SPECS)
    existing = (
        set(FRAMEWORK_FACTOR_SPECS)
        | set(QLIB_EXTRA_FACTOR_SPECS)
        | set(RISK_LIQUIDITY_FACTOR_SPECS)
    )
    assert set(SPECS).isdisjoint(existing)
    assert all(name.startswith("x_wq_alpha") for name in SPECS)
    for name, metadata in SPECS.items():
        assert set(metadata) == {"family", "origin", "formula"}
        assert metadata["origin"].startswith("worldquant_alpha101_")
        assert metadata["formula"]
        assert alpha101_extra_factor_metadata(name) == metadata
    assert alpha101_extra_factor_metadata("x_wq_alpha999") is None


def test_alpha101_factors_are_bounded_ranked_and_input_is_unchanged():
    source = _input_frame()
    frozen = source.copy(deep=True)
    result, features = add_alpha101_extra_factors(source)

    assert features == list(SPECS)
    assert set(features).issubset(result.columns)
    assert list(result.columns[: source.shape[1]]) == list(source.columns)
    assert source.equals(frozen)
    assert result["code"].tolist() == sorted(result["code"].tolist())
    assert result[["code", "date"]].equals(
        result[["code", "date"]].sort_values(["code", "date"], kind="mergesort")
    )
    values = result[features].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert np.isfinite(values).sum() + np.isnan(values).sum() == values.size
    assert finite.min() >= -1.0 - 1e-12
    assert finite.max() <= 1.0 + 1e-12
    assert result[features].notna().any().all()


def test_future_market_perturbation_does_not_change_prior_values():
    source = _input_frame()
    cutoff = pd.Timestamp("2023-06-30")
    original, features = add_alpha101_extra_factors(source)

    changed_source = source.copy(deep=True)
    future = changed_source["date"].gt(cutoff)
    changed_source.loc[future, ["open", "high", "low", "close"]] *= 7.0
    changed_source.loc[future, ["volume", "amount"]] *= 11.0
    changed, changed_features = add_alpha101_extra_factors(changed_source)

    assert features == changed_features
    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_nan_and_inf_inputs_become_missing_outputs_instead_of_inf():
    frame = _input_frame(days=100)
    frame.loc[0, "open"] = np.inf
    frame.loc[1, "close"] = -np.inf
    frame.loc[2, "volume"] = np.nan
    frame.loc[3, "amount"] = np.inf

    result, features = add_alpha101_extra_factors(frame)
    values = result[features].to_numpy(dtype=float)

    assert not np.isinf(values).any()
    assert np.isnan(values).any()


def test_missing_daily_ohlcv_amount_columns_are_rejected():
    frame = _input_frame().drop(columns=["amount"])
    with pytest.raises(ValueError, match="amount"):
        add_alpha101_extra_factors(frame)
