from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.backtest_data import DemoHistoryProvider
from scripts.research_lgbm_new_factors_v1 import (
    FEATURE_BATCH_NAMES,
    blend_model_predictions,
    build_extended_feature_frame,
    candidate_rank,
    promotion_checks,
)


def test_feature_batch_manifest_is_fixed_and_extends_frozen_schema():
    codes = [f"{600000 + index:06d}" for index in range(12)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, date(2023, 1, 1), date(2025, 12, 31)
    )

    frame, feature_sets = build_extended_feature_frame(history)

    assert tuple(feature_sets) == FEATURE_BATCH_NAMES
    assert {name: len(features) for name, features in feature_sets.items()} == {
        "alpha158_baseline": 85,
        "alpha158_plus_alpha101": 97,
        "alpha158_plus_qlib_multiscale": 99,
        "alpha158_plus_barra": 100,
        "alpha158_plus_all_new": 126,
    }
    assert all(len(features) == len(set(features)) for features in feature_sets.values())
    assert set(feature_sets["alpha158_plus_all_new"]).issubset(frame.columns)


def test_extended_features_ignore_future_market_perturbations():
    codes = [f"{600000 + index:06d}" for index in range(12)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, date(2023, 1, 1), date(2025, 12, 31)
    )
    cutoff = pd.Timestamp("2025-06-30")
    original, feature_sets = build_extended_feature_frame(history)

    changed_history = history.copy()
    future = changed_history["date"].gt(cutoff)
    changed_history.loc[
        future, ["open", "high", "low", "close", "volume", "amount"]
    ] *= 5.0
    changed, changed_sets = build_extended_feature_frame(changed_history)

    features = feature_sets["alpha158_plus_all_new"]
    assert feature_sets == changed_sets
    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_candidate_rank_prefers_sharpe_before_return():
    high_sharpe = {
        "selection": {
            "performance": {
                "sharpe": 1.2,
                "annual_return": 0.20,
                "max_drawdown": -0.15,
                "information_ratio": 0.5,
            },
            "activity": {"annual_turnover": 8.0},
        }
    }
    high_return = {
        "selection": {
            "performance": {
                "sharpe": 1.1,
                "annual_return": 0.30,
                "max_drawdown": -0.12,
                "information_ratio": 0.6,
            },
            "activity": {"annual_turnover": 7.0},
        }
    }

    assert candidate_rank(high_sharpe) > candidate_rank(high_return)


def test_model_prediction_blend_uses_daily_ranks_and_bounded_weight():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 3),
        "code": ["A", "B", "C"],
    })
    baseline = pd.Series([3.0, 2.0, 1.0])
    candidate = pd.Series([1.0, 2.0, 3.0])

    baseline_only = blend_model_predictions(frame, baseline, candidate, 0.0)
    balanced = blend_model_predictions(frame, baseline, candidate, 0.5)

    assert baseline_only.iloc[0] > baseline_only.iloc[-1]
    assert balanced.nunique() == 1

    try:
        blend_model_predictions(frame, baseline, candidate, 1.1)
    except ValueError as error:
        assert "融合权重" in str(error)
    else:
        raise AssertionError("out-of-range model blend weight should be rejected")


def test_promotion_gate_requires_return_risk_and_turnover_improvement():
    baseline = {
        "performance": {"annual_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.20},
        "activity": {"annual_turnover": 10.0},
    }
    better = {
        "performance": {"annual_return": 0.22, "sharpe": 1.1, "max_drawdown": -0.18},
        "activity": {"annual_turnover": 10.5},
    }
    worse_turnover = {
        **better,
        "activity": {"annual_turnover": 11.1},
    }

    assert all(promotion_checks(better, baseline).values())
    assert promotion_checks(worse_turnover, baseline)[
        "annual_turnover_not_higher_10pct"
    ] is False
