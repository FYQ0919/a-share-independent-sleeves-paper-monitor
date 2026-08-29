import numpy as np
import pandas as pd
import pytest

from scripts.research_lgbm_risk_exposure_v1 import (
    RISK_WEIGHTS,
    blend_risk_scores,
    build_causal_risk_exposures,
    validate_research_config,
)


def synthetic_risk_frame(days=180, codes=6):
    dates = pd.bdate_range("2018-01-02", periods=days)
    market_returns = 0.008 * np.sin(np.arange(days) / 4) - 0.001
    rows = []
    for code_index in range(codes):
        beta = 0.6 + code_index * 0.18
        residual = 0.002 * np.cos(np.arange(days) / (3 + code_index))
        returns = beta * market_returns + residual
        close = 10 * np.cumprod(1 + returns)
        previous = np.r_[close[0], close[:-1]]
        open_price = previous * (1 + 0.001 * np.sin(np.arange(days) + code_index))
        for index, signal_date in enumerate(dates):
            rows.append({
                "date": signal_date,
                "code": f"6000{code_index:02d}",
                "close": close[index],
                "open": open_price[index],
                "amount": 100_000_000 * (1 + code_index / 10),
            })
    return pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)


def test_risk_configuration_is_small_and_normalized():
    manifest = validate_research_config()

    assert len(manifest) == 64
    assert sum(RISK_WEIGHTS.values()) == pytest.approx(1.0)
    assert len(RISK_WEIGHTS) == 8


def test_future_price_changes_do_not_change_current_risk_exposure_or_regime():
    frame = synthetic_risk_frame()
    cutoff = pd.Timestamp(frame["date"].drop_duplicates().iloc[145])
    original_risk, original_regime = build_causal_risk_exposures(frame)

    changed = frame.copy()
    future = changed["date"].gt(cutoff)
    changed.loc[future, "close"] *= 1.8
    changed.loc[future, "open"] *= 1.6
    changed_risk, changed_regime = build_causal_risk_exposures(changed)

    risk_columns = list(RISK_WEIGHTS) + ["defensive_score", "market_stress"]
    current = original_risk["date"].eq(cutoff)
    pd.testing.assert_frame_equal(
        original_risk.loc[current, risk_columns].reset_index(drop=True),
        changed_risk.loc[current, risk_columns].reset_index(drop=True),
    )
    assert original_regime.loc[cutoff, "market_stress"] == changed_regime.loc[
        cutoff, "market_stress"
    ]


def test_risk_score_only_changes_rank_on_stress_dates():
    current_scores = pd.DataFrame({"score": [0.8, 0.2]})
    risk_frame = pd.DataFrame({
        "defensive_score": [0.1, 0.9],
        "market_stress": [False, True],
    })

    blended = blend_risk_scores(current_scores, risk_frame, 0.50)

    assert blended.loc[0, "score"] == pytest.approx(0.8)
    assert blended.loc[1, "score"] == pytest.approx(0.55)
