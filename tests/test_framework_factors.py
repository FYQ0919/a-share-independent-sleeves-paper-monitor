import numpy as np
import pandas as pd

from app.backtest_data import DemoHistoryProvider
from app.framework_factors import (
    FRAMEWORK_FACTOR_SPECS,
    add_framework_factors,
    framework_factor_metadata,
)
from app.lgbm_features import build_alpha158_lite


def _research_frame():
    codes = [f"{600000 + index:06d}" for index in range(12)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, pd.Timestamp("2023-01-01").date(), pd.Timestamp("2025-12-31").date()
    )
    return history, build_alpha158_lite(history)


def test_framework_layer_is_separate_from_frozen_alpha158_schema():
    _, (frame, alpha_features) = _research_frame()
    extended, framework_features = add_framework_factors(frame)

    assert len(alpha_features) == 85
    assert len(framework_features) == 31
    assert set(alpha_features).isdisjoint(framework_features)
    assert set(framework_features) == set(FRAMEWORK_FACTOR_SPECS)
    assert set(framework_features).issubset(extended.columns)


def test_framework_factor_metadata_is_auditable():
    assert framework_factor_metadata("x_rsv_20") == {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "(close - min(low, 20)) / (max(high, 20) - min(low, 20))",
    }
    assert framework_factor_metadata("x_unknown") is None


def test_framework_factors_before_cutoff_ignore_future_market_changes():
    history, (frame, _) = _research_frame()
    cutoff = pd.Timestamp("2025-06-30")
    original, features = add_framework_factors(frame)

    perturbed_history = history.copy()
    future = perturbed_history["date"].gt(cutoff)
    perturbed_history.loc[future, ["open", "high", "low", "close", "volume", "amount"]] *= 7.0
    changed_base, _ = build_alpha158_lite(perturbed_history)
    changed, changed_features = add_framework_factors(changed_base)

    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])
    assert features == changed_features
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )
