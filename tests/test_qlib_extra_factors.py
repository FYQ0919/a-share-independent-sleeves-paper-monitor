from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.qlib_extra_factors import (
    QLIB_EXTRA_FACTOR_SPECS,
    add_qlib_extra_factors,
    qlib_extra_factor_metadata,
)


def _input_frame(days: int = 90, codes: tuple[str, ...] = ("A", "B", "C", "D")) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    rows: list[dict] = []
    for code_index, code in enumerate(codes):
        base = 20.0 + code_index * 7.0
        for day_index, date in enumerate(dates):
            trend = 0.0015 * day_index
            cycle = 0.025 * np.sin(day_index / (3.0 + code_index))
            close = base * (1.0 + trend + cycle)
            open_price = close * (1.0 + 0.004 * np.cos(day_index + code_index))
            high = max(open_price, close) * (1.0 + 0.012 + 0.001 * code_index)
            low = min(open_price, close) * (1.0 - 0.011 - 0.001 * code_index)
            volume = (1_000_000.0 + 20_000.0 * day_index) * (1.0 + 0.1 * code_index)
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "untouched": f"{code}-{day_index}",
                }
            )
    # Deliberately make input order unlike the calculation order.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=17).reset_index(drop=True)


def test_extra_layer_has_auditable_factor_contract_and_expected_count():
    assert 10 <= len(QLIB_EXTRA_FACTOR_SPECS) <= 16
    assert len(QLIB_EXTRA_FACTOR_SPECS) == 15
    assert list(QLIB_EXTRA_FACTOR_SPECS) == [
        "x_imax_20",
        "x_imin_20",
        "x_imxd_20",
        "x_imax_60",
        "x_imin_60",
        "x_imxd_60",
        "x_cntn_20",
        "x_cntn_60",
        "x_sump_20",
        "x_sumn_20",
        "x_sumd_20",
        "x_corr_20",
        "x_cord_20",
        "x_interval_drawdown_60",
        "x_interval_recovery_60",
    ]
    for name, metadata in QLIB_EXTRA_FACTOR_SPECS.items():
        assert set(metadata) == {"family", "origin", "formula"}
        assert metadata["origin"]
        assert metadata["formula"]
        assert metadata["family"] in {"price_trend", "microstructure_proxy", "risk_tail"}
        assert qlib_extra_factor_metadata(name) == metadata
    assert qlib_extra_factor_metadata("x_not_registered") is None


def test_extra_factors_are_ranked_to_minus_one_one_and_do_not_mutate_input():
    source = _input_frame()
    frozen = source.copy(deep=True)
    result, features = add_qlib_extra_factors(source)

    assert features == list(QLIB_EXTRA_FACTOR_SPECS)
    assert set(features).issubset(result.columns)
    assert list(result.columns[: source.shape[1]]) == list(source.columns)
    assert source.equals(frozen)
    assert result[["code", "date"]].equals(
        result[["code", "date"]].sort_values(["code", "date"], kind="mergesort")
    )
    values = result[features].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    assert finite.min() >= -1.0 - 1e-12
    assert finite.max() <= 1.0 + 1e-12
    assert result[features].notna().any().all()


def test_future_market_perturbation_cannot_change_prior_extra_factor_values():
    source = _input_frame()
    cutoff = pd.Timestamp("2024-04-30")
    original, features = add_qlib_extra_factors(source)

    changed_source = source.copy(deep=True)
    future = changed_source["date"].gt(cutoff)
    changed_source.loc[future, ["open", "high", "low", "close", "volume"]] *= 9.0
    changed, changed_features = add_qlib_extra_factors(changed_source)

    assert features == changed_features
    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_invalid_input_reports_all_missing_ohlcv_columns():
    with pytest.raises(ValueError, match="close"):
        add_qlib_extra_factors(pd.DataFrame({"code": ["A"], "date": ["2024-01-01"]}))

