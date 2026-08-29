from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from research_qlib_lgbm_ranker_v1 import (
    CANDIDATES,
    build_ranker_training_set,
    eligible_training_mask,
    lgbm_model_params,
    validate_candidates,
)


def _ranker_frame() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2019-01-02", "2019-01-03"])
    for date_index, signal_date in enumerate(dates):
        for code_index in range(6):
            rows.append({
                "date": signal_date,
                "code": f"{600000 + code_index:06d}",
                "label_return": float(code_index + date_index),
                "label_end_date": pd.Timestamp("2019-01-18"),
                "tradestatus": 1,
                "is_st": 0,
                "close": 10.0,
                "amount_20": 50_000_000.0,
                "universe_member": True,
                "x_a": code_index / 10,
                "x_b": np.nan if code_index == 0 else code_index / 20,
            })
    return pd.DataFrame(rows)


def test_candidate_grid_is_small_and_capacity_limited():
    manifest = validate_candidates()

    assert len(manifest) == 64
    assert len(CANDIDATES) == 3
    assert all(config["max_depth"] <= 4 for config in CANDIDATES.values())
    assert all(config["min_child_samples"] >= 300 for config in CANDIDATES.values())


def test_ranker_training_groups_and_relevance_are_cross_sectional():
    frame = _ranker_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)

    train, relevance, groups = build_ranker_training_set(
        frame, ["x_a", "x_b"], pd.Timestamp("2019-02-01")
    )

    assert groups == [6, 6]
    assert sum(groups) == len(train) == len(relevance)
    assert train[["date", "code"]].equals(
        train[["date", "code"]].sort_values(["date", "code"])
    )
    assert relevance.dtype == np.int32
    assert relevance.min() == 0
    assert relevance.max() == 4
    assert relevance[:6].tolist() == relevance[6:].tolist()


def test_immature_peer_is_removed_before_relevance_is_recomputed():
    frame = _ranker_frame().iloc[:6].copy()
    frame.loc[frame["code"].eq("600005"), "label_end_date"] = pd.Timestamp("2019-03-01")

    mask = eligible_training_mask(frame, pd.Timestamp("2019-02-01"))
    train, relevance, groups = build_ranker_training_set(
        frame, ["x_a", "x_b"], pd.Timestamp("2019-02-01")
    )

    assert mask.sum() == 5
    assert "600005" not in train["code"].tolist()
    assert groups == [5]
    assert relevance.tolist() == [0, 1, 2, 3, 4]


def test_training_boundary_excludes_labels_ending_on_prediction_date():
    frame = _ranker_frame().iloc[:6].copy()
    frame["label_end_date"] = pd.Timestamp("2019-02-01")

    mask = eligible_training_mask(frame, pd.Timestamp("2019-02-01"))

    assert not mask.any()


def test_model_parameters_are_deterministic_and_regularized():
    params = lgbm_model_params(CANDIDATES["lgbm_l7_blend40"])

    assert params["objective"] == "lambdarank"
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert params["num_threads"] == 1
    assert params["bagging_freq"] == 0
    assert params["lambda_l1"] > 0
    assert params["lambda_l2"] > 0
