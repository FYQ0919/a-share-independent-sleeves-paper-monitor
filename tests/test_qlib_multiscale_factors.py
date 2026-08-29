from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.qlib_multiscale_factors import (
    QLIB_MULTISCALE_FACTOR_SPECS,
    SPECS,
    add,
    add_qlib_multiscale_factors,
    metadata,
    qlib_multiscale_factor_metadata,
)


def _input_frame(days: int = 130, codes: tuple[str, ...] = ("A", "B", "C", "D")) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        base = 18.0 + code_index * 6.0
        for day_index, date in enumerate(dates):
            trend = (0.0012 + code_index * 0.00035) * day_index
            cycle = 0.018 * np.sin(day_index / (3.0 + code_index))
            close = base * (1.0 + trend + cycle)
            open_price = close * (1.0 + 0.004 * np.cos(day_index + code_index))
            high = max(open_price, close) * (1.0 + 0.012 + 0.001 * code_index)
            low = min(open_price, close) * (1.0 - 0.011 - 0.001 * code_index)
            volume = (900_000.0 + 15_000.0 * day_index) * (1.0 + 0.11 * code_index)
            amount = volume * close * (1.0 + 0.01 * np.sin(day_index / 5.0))
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "untouched": f"{code}-{day_index}",
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=17).reset_index(drop=True)


def test_multiscale_specs_are_auditable_and_have_expected_count():
    assert SPECS is QLIB_MULTISCALE_FACTOR_SPECS
    assert 8 <= len(QLIB_MULTISCALE_FACTOR_SPECS) <= 15
    assert len(QLIB_MULTISCALE_FACTOR_SPECS) == 14
    assert list(QLIB_MULTISCALE_FACTOR_SPECS) == [
        "x_beta_20",
        "x_beta_60",
        "x_rsqr_20",
        "x_rsqr_60",
        "x_resi_20",
        "x_resi_60",
        "x_volatility_regime_20",
        "x_volatility_regime_60",
        "x_vwap_gap_20",
        "x_vwap_gap_60",
        "x_money_flow_imbalance_20",
        "x_money_flow_imbalance_60",
        "x_volume_price_slope_gap_20",
        "x_volume_price_slope_gap_60",
    ]
    for name, spec in SPECS.items():
        assert set(spec) == {"family", "origin", "formula"}
        assert spec["family"] in {"price_trend", "risk_tail", "microstructure_proxy"}
        assert spec["origin"]
        assert spec["formula"]
        assert qlib_multiscale_factor_metadata(name) == spec
        assert metadata(name) == spec
    assert qlib_multiscale_factor_metadata("x_missing") is None


def test_add_returns_sorted_copy_ranked_to_bounds_without_mutating_input():
    source = _input_frame()
    frozen = source.copy(deep=True)
    result, features = add_qlib_multiscale_factors(source)

    assert features == list(SPECS)
    assert add is add_qlib_multiscale_factors
    assert set(features).issubset(result.columns)
    assert list(result.columns[: source.shape[1]]) == list(source.columns)
    assert source.equals(frozen)
    assert result[["code", "date"]].equals(
        result[["code", "date"]].sort_values(["code", "date"], kind="mergesort")
    )

    values = result[features].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert finite.min() >= -1.0 - 1e-12
    assert finite.max() <= 1.0 + 1e-12
    assert result[features].notna().any().all()
    assert result.loc[result["code"].eq("A"), "x_rsqr_60"].iloc[:59].isna().all()


def test_future_market_perturbation_cannot_change_prior_factor_values():
    source = _input_frame()
    cutoff = pd.Timestamp("2024-05-31")
    original, features = add_qlib_multiscale_factors(source)

    changed_source = source.copy(deep=True)
    future = changed_source["date"].gt(cutoff)
    changed_source.loc[future, ["open", "high", "low", "close", "volume", "amount"]] *= 9.0
    changed, changed_features = add_qlib_multiscale_factors(changed_source)

    assert features == changed_features
    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_nan_inf_and_non_numeric_market_values_are_safe():
    source = _input_frame()
    source.loc[0, "close"] = np.inf
    source.loc[1, "volume"] = 0.0
    source["amount"] = source["amount"].astype(object)
    source.loc[2, "amount"] = "not-a-number"
    source.loc[3, "high"] = np.nan
    result, features = add_qlib_multiscale_factors(source)

    values = result[features].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert result[features].notna().any().all()


def test_missing_daily_columns_are_reported():
    with pytest.raises(ValueError, match="amount"):
        add_qlib_multiscale_factors(
            pd.DataFrame({"code": ["A"], "date": ["2024-01-01"]})
        )
