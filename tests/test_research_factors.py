import numpy as np
import pandas as pd

from app.backtest_data import DemoHistoryProvider
from app.lgbm_features import build_alpha158_lite
from app.research_factors import (
    RESEARCH_FACTOR_BATCHES,
    add_research_factors,
    research_factor_metadata,
)


def _history():
    codes = [f"{600000 + index:06d}" for index in range(12)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, pd.Timestamp("2023-01-01").date(), pd.Timestamp("2025-12-31").date()
    )
    return history


def test_research_factor_batches_are_unique_and_auditable():
    names = [name for specs in RESEARCH_FACTOR_BATCHES.values() for name in specs]
    assert {batch: len(specs) for batch, specs in RESEARCH_FACTOR_BATCHES.items()} == {
        "framework_v1": 31,
        "qlib_extra_v1": 15,
        "risk_liquidity_v1": 14,
        "alpha101_extra_v1": 12,
        "qlib_multiscale_v1": 14,
        "barra_residual_v1": 15,
    }
    assert len(names) == len(set(names)) == 101
    assert all(research_factor_metadata(name) for name in names)


def test_research_layer_preserves_frozen_lgbm_schema_and_is_causal():
    history = _history()
    base, alpha_features = build_alpha158_lite(history)
    original, features, batches = add_research_factors(base)
    cutoff = pd.Timestamp("2025-06-30")

    changed_history = history.copy()
    future = changed_history["date"].gt(cutoff)
    changed_history.loc[future, ["open", "high", "low", "close", "volume", "amount"]] *= 6.0
    changed_base, changed_alpha_features = build_alpha158_lite(changed_history)
    changed, changed_features, changed_batches = add_research_factors(changed_base)

    assert len(alpha_features) == 85
    assert alpha_features == changed_alpha_features
    assert features == changed_features
    assert batches == changed_batches
    assert len(features) == 101
    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )
