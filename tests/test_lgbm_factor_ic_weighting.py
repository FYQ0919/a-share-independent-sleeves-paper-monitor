import numpy as np
import pandas as pd
import pytest

from scripts.research_lgbm_factor_ic_weighting_v1 import (
    daily_rank_ic,
    expanding_factor_scores,
    normalized_ic_weights,
    select_nonredundant_factors,
)


def test_daily_rank_ic_recovers_positive_and_negative_factor_direction():
    rows = []
    for signal_date in pd.to_datetime(["2017-01-03", "2017-01-04"]):
        for value in range(6):
            rows.append({
                "date": signal_date,
                "code": f"60000{value}",
                "label_return": float(value),
                "label_end_date": signal_date + pd.Timedelta(days=15),
                "x_positive": float(value),
                "x_negative": float(-value),
            })
    result = daily_rank_ic(
        pd.DataFrame(rows), ["x_positive", "x_negative"]
    )

    assert result["x_positive"].tolist() == pytest.approx([1.0, 1.0])
    assert result["x_negative"].tolist() == pytest.approx([-1.0, -1.0])


def test_nonredundant_selection_removes_highly_correlated_factor():
    statistics = pd.DataFrame([
        {
            "feature": "x_a",
            "ic_days": 200,
            "direction_hit_rate": 0.65,
            "selection_strength": 0.08,
            "abs_mean_rank_ic": 0.12,
        },
        {
            "feature": "x_b",
            "ic_days": 200,
            "direction_hit_rate": 0.63,
            "selection_strength": 0.07,
            "abs_mean_rank_ic": 0.11,
        },
        {
            "feature": "x_c",
            "ic_days": 200,
            "direction_hit_rate": 0.61,
            "selection_strength": 0.06,
            "abs_mean_rank_ic": 0.10,
        },
        {
            "feature": "x_d",
            "ic_days": 200,
            "direction_hit_rate": 0.60,
            "selection_strength": 0.05,
            "abs_mean_rank_ic": 0.09,
        },
        {
            "feature": "x_e",
            "ic_days": 200,
            "direction_hit_rate": 0.59,
            "selection_strength": 0.04,
            "abs_mean_rank_ic": 0.08,
        },
        {
            "feature": "x_f",
            "ic_days": 200,
            "direction_hit_rate": 0.58,
            "selection_strength": 0.03,
            "abs_mean_rank_ic": 0.07,
        },
    ])
    names = statistics["feature"].tolist()
    correlation = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    correlation.loc["x_a", "x_b"] = correlation.loc["x_b", "x_a"] = 0.95

    selected, redundancy = select_nonredundant_factors(
        statistics, correlation, max_factors=5, max_pair_correlation=0.80
    )

    assert selected == ["x_a", "x_c", "x_d", "x_e", "x_f"]
    assert "x_b" not in selected
    assert redundancy["x_c"] == pytest.approx(0.0)


def test_ic_weights_preserve_direction_and_normalize_absolute_exposure():
    history = pd.DataFrame({
        "x_positive": [0.10, 0.20, 0.15, 0.12],
        "x_negative": [-0.20, -0.10, -0.15, -0.12],
    })

    weights = normalized_ic_weights(history, ["x_positive", "x_negative"])

    assert weights["x_positive"] > 0
    assert weights["x_negative"] < 0
    assert weights.abs().sum() == pytest.approx(1.0)


def test_expanding_factor_score_excludes_unmatured_label_rows():
    dates = pd.to_datetime(["2019-01-02", "2019-01-03"])
    frame = pd.DataFrame([
        {"date": signal_date, "x_a": value / 10, "x_b": -value / 10}
        for signal_date in dates
        for value in range(6)
    ])
    mature = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=126, freq="D"),
        "label_end_date": pd.Timestamp("2018-12-31"),
        "x_a": 0.10,
        "x_b": -0.05,
    })
    future = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=10, freq="D"),
        "label_end_date": pd.Timestamp("2019-12-31"),
        "x_a": -1.0,
        "x_b": 1.0,
    })

    scores, audit = expanding_factor_scores(
        frame,
        pd.concat([mature, future], ignore_index=True),
        ["x_a", "x_b"],
        dates[0].date(),
        dates[-1].date(),
    )

    assert scores.notna().all()
    assert audit["refit_months"] == 1
    assert audit["maturity_audit"][0]["max_training_label_end_date"] == "2018-12-31"
    assert audit["maturity_audit"][0]["max_training_label_end_date"] < "2019-01-02"
