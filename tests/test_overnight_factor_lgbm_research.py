import numpy as np
import pandas as pd

from scripts.research_overnight_factor_lgbm_2020_2026 import (
    fixed_factor_overlay,
    maturity_checks,
)


def test_fixed_overlay_preserves_invalid_rows_and_uses_frozen_weight():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 3),
        "x_overnight_momentum_5": [-1.0, 0.0, 1.0],
    })
    scores = pd.DataFrame({"score": [0.20, np.nan, 0.80]})

    result = fixed_factor_overlay(frame, scores, weight=0.15)

    assert np.isclose(result.loc[0, "score"], 0.85 * 0.20 + 0.15 * (1 / 3))
    assert np.isnan(result.loc[1, "score"])
    assert np.isclose(result.loc[2, "score"], 0.85 * 0.80 + 0.15)


def test_maturity_checks_require_label_end_before_prediction_start():
    audit = {
        "maturity_audit": [
            {
                "month": "2020-01",
                "prediction_start": "2020-01-02",
                "max_training_label_end_date": "2019-12-31",
            },
            {
                "month": "2020-02",
                "prediction_start": "2020-02-03",
                "max_training_label_end_date": "2020-02-03",
            },
        ]
    }

    assert maturity_checks(audit) == {"2020-01": True, "2020-02": False}
