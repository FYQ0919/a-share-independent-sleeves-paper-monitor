from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


# The research script is intentionally executable from scripts/ as well as
# importable by tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app.backtest_data import DemoHistoryProvider
from research_qlib_ridge_topk_v1 import (
    build_alpha158_lite,
    mature_training_mask,
    topk_dropout_backtest,
)


def _demo_history() -> pd.DataFrame:
    codes = [f"{600000 + index:06d}" for index in range(20)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, date(2024, 1, 1), date(2025, 12, 31)
    )
    return history


def test_alpha158_lite_group_alignment_is_invariant_to_input_order():
    history = _demo_history()
    ordered, feature_columns = build_alpha158_lite(history)
    shuffled, shuffled_features = build_alpha158_lite(
        history.sample(frac=1.0, random_state=17).reset_index(drop=True)
    )

    assert feature_columns == shuffled_features
    columns = ["date", "code"] + feature_columns
    left = ordered[columns].sort_values(["code", "date"]).reset_index(drop=True)
    right = shuffled[columns].sort_values(["code", "date"]).reset_index(drop=True)
    for column in feature_columns:
        assert np.allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_alpha158_lite_features_are_prefix_invariant_to_future_rows():
    history = _demo_history()
    cutoff = pd.Timestamp("2025-06-30")
    original, feature_columns = build_alpha158_lite(history)

    perturbed_history = history.copy()
    future = perturbed_history["date"] > cutoff
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover", "pe", "pb"]:
        perturbed_history.loc[future, column] = (
            pd.to_numeric(perturbed_history.loc[future, column], errors="coerce") * 1.7
            + 3.0
        )
    perturbed, _ = build_alpha158_lite(perturbed_history)

    columns = ["date", "code"] + feature_columns
    left = (
        original.loc[original["date"] <= cutoff, columns]
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    right = (
        perturbed.loc[perturbed["date"] <= cutoff, columns]
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    for column in feature_columns:
        assert np.allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_maturity_mask_uses_per_row_label_endpoint():
    frame = pd.DataFrame(
        {
            "label_return": [0.1, 0.2, np.nan, 0.4],
            "label_end_date": pd.to_datetime(
                ["2020-01-09", "2020-01-11", "2020-01-10", None]
            ),
        }
    )

    mask = mature_training_mask(frame, pd.Timestamp("2020-01-10"))

    assert mask.tolist() == [True, False, False, False]


def test_topk_dropout_executes_signal_at_next_open_and_maps_returns_correctly():
    dates = pd.bdate_range("2020-01-01", periods=8)
    codes = ["A", "B", "C", "D"]
    history_rows = []
    score_rows = []
    for index, trade_date in enumerate(dates):
        for code in codes:
            price = 100.0 * (1.1**index) if code == "A" else 100.0
            history_rows.append({"date": trade_date, "code": code, "open": price})
        ordering = ["A", "B", "C", "D"] if index < 4 else ["A", "C", "B", "D"]
        for rank, code in enumerate(ordering):
            score_rows.append({"date": trade_date, "code": code, "score": 4 - rank})

    result = topk_dropout_backtest(
        pd.DataFrame(history_rows),
        pd.DataFrame(score_rows),
        date(2020, 1, 1),
        date(2020, 1, 10),
        top_k=2,
        n_drop=1,
        rebalance_days=4,
        cost_bps=0,
    )

    assert result["rebalances"][0]["signal_date"] == "2020-01-01"
    assert result["rebalances"][0]["execution_date"] == "2020-01-02"
    # No position exists over the signal-to-execution interval. The first
    # realized holding return starts at the execution open on Jan 2.
    assert result["return_periods"][0]["strategy_return"] == 0.0
    assert result["return_periods"][1]["period_start"] == "2020-01-02"
    assert result["return_periods"][1]["strategy_return"] == pytest.approx(0.05)
