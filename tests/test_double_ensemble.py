from datetime import date

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("lightgbm")

from app.double_ensemble import (
    DoubleEnsembleConfig,
    cross_sectional_regression_target,
    residual_feature_scores,
    residual_sample_weights,
    selected_feature_indices,
)


def test_cross_sectional_target_is_date_local():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3),
        "label_return": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })
    target = cross_sectional_regression_target(frame)
    np.testing.assert_allclose(target[:3], target[3:])


def test_residual_weights_balance_dates_and_emphasize_hard_samples():
    dates = pd.to_datetime(["2020-01-02"] * 2 + ["2020-01-03"] * 4)
    residuals = np.array([0.1, 2.0, 0.1, 0.2, 0.3, 3.0])
    weights = residual_sample_weights(dates, residuals)

    assert weights[1] > weights[0]
    assert weights[-1] > weights[-2]
    assert weights[:2].sum() == pytest.approx(weights[2:].sum(), rel=0.30)
    assert weights.mean() == pytest.approx(1.0)


def test_dynamic_feature_selection_prefers_residual_signal():
    residual = np.linspace(-1.0, 1.0, 100)
    matrix = np.column_stack([
        residual,
        np.sin(np.arange(100)),
        np.zeros(100),
        -residual,
    ])
    scores = residual_feature_scores(matrix, residual, [1], np.array([1.0]))
    selected = selected_feature_indices(scores, ratio=0.5, minimum=1)

    assert 0 in selected or 3 in selected
    assert len(selected) == 2


def test_config_is_fixed_and_capacity_limited():
    config = DoubleEnsembleConfig()
    config.validate()

    assert config.num_models == 4
    assert config.max_depth <= 4
    assert config.min_child_samples >= 300
    assert config.feature_ratios[-1] == pytest.approx(0.40)
