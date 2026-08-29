from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from research_tail_board_factor_v1 import (  # noqa: E402
    TailBoardResearchEngine,
    validate_candidates,
)


def _history(code="300001", periods=90):
    dates = pd.bdate_range("2020-06-01", periods=periods)
    close = np.linspace(10.0, 12.0, periods)
    return pd.DataFrame({
        "date": dates,
        "code": code,
        "name": "测试股票",
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": 1_000_000.0,
        "amount": 100_000_000.0,
        "turnover": 2.0,
        "pct_change": 5.0,
        "tradestatus": 1,
        "is_st": 0,
        "pe": 20.0,
        "pb": 2.0,
    })


def test_candidate_manifest_is_valid_and_frozen():
    assert len(validate_candidates()) == 64


def test_chinext_limit_regime_and_listing_age_filter():
    features = TailBoardResearchEngine()._features(_history())
    before = features[features["date"] < pd.Timestamp("2020-08-24")]
    after = features[features["date"] >= pd.Timestamp("2020-08-24")]

    assert before["limit_pct"].eq(10.0).all()
    assert after["limit_pct"].eq(20.0).all()
    assert features.iloc[:59]["tail_board_raw"].isna().all()
    assert features.iloc[59:]["tail_board_raw"].notna().all()


def test_research_engine_does_not_change_production_strategy_weights():
    engine = TailBoardResearchEngine()
    assert "tail_board" not in engine.STRATEGIES["contrarian"]["weights"]
