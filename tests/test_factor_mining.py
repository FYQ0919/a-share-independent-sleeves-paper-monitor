from datetime import date

import numpy as np
import pandas as pd

from app.backtest_data import DemoHistoryProvider
from app.factor_mining import (
    FactorMiningConfig,
    MiningWindow,
    expanding_factor_scores,
    factor_diagnostics,
    factor_family,
    select_stable_nonredundant_factors,
    window_ic_rows,
)
from app.lgbm_features import build_alpha158_lite


def _config(**overrides):
    values = {
        "discovery": MiningWindow("discovery", date(2020, 1, 1), date(2020, 1, 10), "discovery"),
        "selection": MiningWindow("selection", date(2020, 1, 11), date(2020, 1, 20), "selection"),
        "diagnostic": MiningWindow("diagnostic", date(2020, 1, 21), date(2020, 1, 31), "diagnostic"),
        "min_cross_section": 2,
        "min_ic_days": 3,
        "min_abs_discovery_ic": 0.01,
        "max_factors": 5,
        "ic_lookback_days": 5,
    }
    values.update(overrides)
    return FactorMiningConfig(**values)


def test_window_ic_rows_excludes_labels_that_mature_after_window_end():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-08", "2020-01-09"]),
            "label_end_date": pd.to_datetime(["2020-01-10", "2020-01-21"]),
            "x_factor": [0.1, 0.2],
        }
    )

    selected = window_ic_rows(rows, _config().discovery)

    assert selected["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-08"]


def test_diagnostic_period_cannot_change_factor_selection():
    dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
    features = [f"x_factor_{index}" for index in range(6)]
    rows = pd.DataFrame(
        {
            "date": dates,
            "label_end_date": dates,
            **{
                feature: np.r_[
                    np.full(10, 0.10 - index * 0.005),
                    np.full(10, 0.08 - index * 0.004),
                    np.full(11, -0.90),
                ]
                for index, feature in enumerate(features)
            },
        }
    )
    correlation = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    config = _config()

    first = factor_diagnostics(rows, features, config)
    _, selected_first = select_stable_nonredundant_factors(first, correlation, config)
    rows.loc[rows["date"].ge("2020-01-21"), features] *= -100
    second = factor_diagnostics(rows, features, config)
    _, selected_second = select_stable_nonredundant_factors(second, correlation, config)

    assert selected_first == selected_second
    assert len(selected_first) == 5


def test_correlation_deduplication_keeps_only_stronger_factor():
    features = ["x_strong", "x_duplicate", "x_independent", "x_four", "x_five", "x_six"]
    diagnostics = pd.DataFrame(
        {
            "feature": features,
            "family": ["other"] * 6,
            "discovery_ic_days": [10] * 6,
            "selection_ic_days": [10] * 6,
            "discovery_abs_mean_rank_ic": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05],
            "discovery_direction_hit_rate": [0.7] * 6,
            "selection_direction_hit_rate": [0.7] * 6,
            "selection_retention": [0.8] * 6,
            "stability_score": [0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
        }
    )
    correlation = pd.DataFrame(np.eye(6), index=features, columns=features)
    correlation.loc["x_strong", "x_duplicate"] = 0.95
    correlation.loc["x_duplicate", "x_strong"] = 0.95

    output, selected = select_stable_nonredundant_factors(
        diagnostics, correlation, _config(max_factors=5)
    )

    assert "x_strong" in selected
    assert "x_duplicate" not in selected
    assert output.set_index("feature").loc["x_duplicate", "selection_reason"] == "redundant"


def test_monthly_factor_weights_only_use_strictly_mature_labels():
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "x_a": np.tile([0.2, 0.8], len(dates)),
            "x_b": np.tile([0.7, 0.3], len(dates)),
        }
    )
    ic_rows = pd.DataFrame(
        {
            "date": dates,
            "label_end_date": dates + pd.Timedelta(days=2),
            "x_a": 0.1,
            "x_b": -0.05,
        }
    )

    scores, audit = expanding_factor_scores(
        frame,
        ic_rows,
        ["x_a", "x_b"],
        date(2020, 2, 1),
        date(2020, 3, 31),
        _config(),
    )

    assert audit
    assert all(row["strictly_mature"] for row in audit)
    assert all(row["max_training_label_end_date"] < row["prediction_start"] for row in audit)
    assert scores.notna().any()


def test_alpha_features_before_cutoff_ignore_future_price_perturbation():
    codes = [f"{600000 + index:06d}" for index in range(12)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, date(2023, 1, 1), date(2025, 12, 31)
    )
    cutoff = pd.Timestamp("2025-06-30")
    original, features = build_alpha158_lite(history)
    perturbed = history.copy()
    future = perturbed["date"].gt(cutoff)
    perturbed.loc[future, ["open", "high", "low", "close"]] *= 7.0
    changed, changed_features = build_alpha158_lite(perturbed)
    left = original[original["date"].le(cutoff)].sort_values(["code", "date"])
    right = changed[changed["date"].le(cutoff)].sort_values(["code", "date"])

    assert features == changed_features
    np.testing.assert_allclose(
        left[features].to_numpy(dtype=float),
        right[features].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_factor_family_catalog_is_explicit():
    assert factor_family("x_pb") == "valuation"
    assert factor_family("x_downside_vol_60") == "risk_tail"
    assert factor_family("x_turnover_ratio_20") == "liquidity"
