from datetime import date

import numpy as np
import pandas as pd

from scripts.backtest_lgbm_dynamic_top50_2y import (
    TOP50_SIZE,
    attach_historical_membership,
    blend_scores_historical,
    build_membership_intervals,
    historical_selection_dates,
    select_historical_top50,
)


def test_selection_grid_is_aligned_and_has_warmup_dates():
    dates = pd.bdate_range("2024-01-02", periods=35)
    selected = historical_selection_dates(dates, date(2024, 1, 22), step=10)

    assert selected[0] == dates[4]
    assert selected[-1] == dates[-1]
    assert list(selected) == [dates[4], dates[14], dates[24], dates[34]]


def test_full_market_cross_section_selects_top50_as_of_each_date():
    trade_date = pd.Timestamp("2024-09-23")
    rows = []
    for index in range(60):
        rows.append({
            "date": trade_date,
            "code": f"{600000 + index:06d}",
            "name": f"股票{index}",
            "tradestatus": 1,
            "is_st": 0,
            "close": 10.0,
            "amount_20": 20_000_000.0,
            "amount_60": 20_000_000.0,
            "ret_5": 0.01,
            "ret_20": 0.02,
            "ret_60": 0.03,
            "ret_120": 0.04,
            "volatility_20": 0.01,
            "volatility_60": 0.01,
            "downside_volatility_60": 0.01,
            "absolute_return_60": 0.20,
            "absolute_return_120": 0.40,
            "ma_20": 10.0,
            "drawdown_60": -0.01,
            "drawdown_120": -0.02,
            "intraday_strength_20": 0.01,
            "pe": 10.0,
            "pb": 1.0,
        })
    frame = pd.DataFrame(rows)
    frame.loc[frame["code"] == "600059", "ret_60"] = 1.0

    top50, audits = select_historical_top50(
        frame, pd.DatetimeIndex([trade_date]), top_n=TOP50_SIZE
    )

    assert len(top50) == TOP50_SIZE
    assert top50["selection_date"].eq(trade_date).all()
    assert top50["universe_rank"].tolist() == list(range(1, TOP50_SIZE + 1))
    assert audits[0]["eligible_count"] == 60
    assert audits[0]["selected_count"] == TOP50_SIZE


def test_membership_and_intervals_are_historical_and_auditable():
    dates = pd.bdate_range("2024-01-02", periods=25)
    selection_dates = pd.DatetimeIndex([dates[0], dates[10], dates[20]])
    top50 = pd.DataFrame([
        {
            "selection_date": dates[0],
            "available_date": dates[0],
            "code": "000001",
            "name": "A",
            "universe_rank": 1,
            "universe_selection_score": 90.0,
        },
        {
            "selection_date": dates[10],
            "available_date": dates[10],
            "code": "000002",
            "name": "B",
            "universe_rank": 1,
            "universe_selection_score": 91.0,
        },
        {
            "selection_date": dates[20],
            "available_date": dates[20],
            "code": "000001",
            "name": "A",
            "universe_rank": 2,
            "universe_selection_score": 89.0,
        },
    ])
    history = pd.DataFrame([
        {"date": trade_date, "code": code, "open": 10.0, "close": 10.0}
        for trade_date in dates
        for code in ["000001", "000002"]
    ])

    marked = attach_historical_membership(history, top50, selection_dates)
    intervals = build_membership_intervals(top50, selection_dates, dates)

    first_period = marked[marked["date"].eq(dates[0])]
    second_period = marked[marked["date"].eq(dates[10])]
    assert set(first_period.loc[first_period["universe_member"], "code"]) == {"000001"}
    assert set(second_period.loc[second_period["universe_member"], "code"]) == {"000002"}
    assert (marked.loc[marked["universe_member"], "universe_selection_date"] <= marked.loc[marked["universe_member"], "date"]).all()
    code_a_intervals = intervals[intervals["code"].eq("000001")]
    assert len(code_a_intervals) == 2
    assert code_a_intervals["entry_date"].iloc[0] == dates[0].date().isoformat()
    assert code_a_intervals["exit_date"].iloc[0] == dates[9].date().isoformat()
    assert code_a_intervals["exit_date"].iloc[1] == dates[24].date().isoformat()


def test_scores_are_ranked_only_inside_active_top50():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-09-23"] * 3),
        "code": ["A", "B", "C"],
        "tradestatus": [1, 1, 1],
        "is_st": [0, 0, 0],
        "close": [10.0, 10.0, 10.0],
        "amount_20": [20_000_000.0] * 3,
        "universe_member": [True, True, False],
        "universe_selection_date": pd.to_datetime(["2024-09-23"] * 3),
        "available_date": pd.to_datetime(["2024-09-23"] * 3),
        "universe_rank": [1.0, 2.0, np.nan],
    })
    scores = blend_scores_historical(
        frame,
        pd.Series([1.0, 2.0, 100.0]),
        pd.Series([0.5, 1.0, np.nan]),
        baseline_weight=0.4,
    )

    assert scores.loc[0, "score"] < scores.loc[1, "score"]
    assert scores.loc[2, "score"] != scores.loc[2, "score"]


def test_scores_fall_back_to_rules_before_first_mature_lgbm_refit():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-09-23"] * 2),
        "code": ["A", "B"],
        "tradestatus": [1, 1],
        "is_st": [0, 0],
        "close": [10.0, 10.0],
        "amount_20": [20_000_000.0] * 2,
        "universe_member": [True, True],
        "universe_selection_date": pd.to_datetime(["2024-09-23"] * 2),
        "available_date": pd.to_datetime(["2024-09-23"] * 2),
        "universe_rank": [1.0, 2.0],
    })
    scores = blend_scores_historical(
        frame,
        pd.Series([np.nan, np.nan]),
        pd.Series([0.5, 1.0]),
        baseline_weight=0.4,
    )

    assert scores["score"].notna().all()
    assert scores.loc[1, "score"] > scores.loc[0, "score"]
