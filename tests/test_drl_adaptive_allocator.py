from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.research_drl_adaptive_allocator_v1 import (
    FEATURE_COLUMNS,
    HEDGE_RATIO_MAX,
    SIGNAL_TO_RETURN_LAG,
    SharpePolicy,
    _episode_reward,
    build_causal_features,
    expand_actions,
    map_and_smooth_action,
    seed_everything,
    simulate_returns,
)


def test_action_mapping_respects_allocation_and_hedge_bounds() -> None:
    latent = torch.tensor(
        [[-100.0, -100.0], [0.0, 0.0], [100.0, 100.0]], dtype=torch.float32
    )
    previous = torch.tensor([[0.75, 0.40]] * 3, dtype=torch.float32)
    actions = map_and_smooth_action(latent, previous)

    assert torch.all(actions[:, 0] >= 0.55)
    assert torch.all(actions[:, 0] <= 0.90)
    assert torch.all(1.0 - actions[:, 0] >= 0.10)
    assert torch.all(1.0 - actions[:, 0] <= 0.45)
    assert torch.all(actions[:, 1] >= 0.0)
    assert torch.all(actions[:, 1] <= HEDGE_RATIO_MAX)


def test_t_close_action_cannot_affect_t_or_t_plus_one_curve_return() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    records = pd.DataFrame(
        {"trend_weight": [0.90], "hedge_ratio": [0.0]}, index=[dates[1]]
    )
    expanded = expand_actions(dates, records)

    assert SIGNAL_TO_RETURN_LAG == 2
    assert expanded.loc[dates[1], "trend_weight"] == pytest.approx(0.75)
    assert expanded.loc[dates[2], "trend_weight"] == pytest.approx(0.75)
    assert expanded.loc[dates[3], "trend_weight"] == pytest.approx(0.90)


def test_future_perturbation_does_not_change_feature_prefix() -> None:
    dates = pd.bdate_range("2022-01-03", periods=260)
    base = np.linspace(1.0, 1.8, len(dates))
    trend = pd.Series(base * (1.0 + 0.01 * np.sin(np.arange(len(dates)))), index=dates)
    lgbm = pd.Series(base * (1.0 + 0.02 * np.cos(np.arange(len(dates)))), index=dates)
    index = pd.DataFrame(
        {"open": 4_000 * base, "close": 4_010 * base}, index=dates
    )
    cutoff = dates[210]
    original = build_causal_features(trend, lgbm, index)

    changed_trend = trend.copy()
    changed_lgbm = lgbm.copy()
    changed_index = index.copy()
    changed_trend.loc[changed_trend.index > cutoff] *= 8.0
    changed_lgbm.loc[changed_lgbm.index > cutoff] *= 0.1
    changed_index.loc[changed_index.index > cutoff, ["open", "close"]] *= 4.0
    perturbed = build_causal_features(changed_trend, changed_lgbm, changed_index)

    pd.testing.assert_frame_equal(original.loc[:cutoff], perturbed.loc[:cutoff])
    assert list(original.columns) == list(FEATURE_COLUMNS)


def test_weight_and_hedge_changes_charge_costs_only_when_they_change() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    daily = pd.DataFrame(
        {
            "trend_return": 0.0,
            "lgbm_return": 0.0,
            "csi300_open_return": 0.0,
            "risk_off_effective": [0.0, 0.0, 1.0, 1.0, 1.0],
        },
        index=dates,
    )
    unchanged = pd.DataFrame(
        {"trend_weight": 0.75, "hedge_ratio": 0.75}, index=dates
    )
    changed = unchanged.copy()
    changed.loc[dates[2]:, "trend_weight"] = 0.60
    unchanged_result = simulate_returns(daily, unchanged)
    changed_result = simulate_returns(daily, changed)

    assert unchanged_result["allocation_cost"].sum() == 0.0
    assert unchanged_result.loc[dates[0]:dates[1], "hedge_cost"].sum() == 0.0
    assert unchanged_result.loc[dates[2], "hedge_cost"] > 0.0
    assert changed_result.loc[dates[2], "allocation_cost"] > 0.0
    assert changed_result.loc[dates[2], "hedge_cost"] > 0.0


def test_fixed_seed_reproduces_policy_parameters_and_actions() -> None:
    seed_everything(1234)
    first = SharpePolicy(len(FEATURE_COLUMNS))
    state = torch.linspace(-1.0, 1.0, len(FEATURE_COLUMNS) + 2)
    first_output = first(state).detach().clone()
    first_parameters = [parameter.detach().clone() for parameter in first.parameters()]

    seed_everything(1234)
    second = SharpePolicy(len(FEATURE_COLUMNS))
    second_output = second(state).detach().clone()

    assert torch.equal(first_output, second_output)
    for expected, actual in zip(first_parameters, second.parameters()):
        assert torch.equal(expected, actual.detach())


def test_training_reward_is_invariant_to_returns_after_training_cutoff() -> None:
    seed_everything(4321)
    model = SharpePolicy(len(FEATURE_COLUMNS))
    feature_values = torch.zeros((2, len(FEATURE_COLUMNS)), dtype=torch.float32)
    event_positions = np.array([0, 4])
    trend = torch.linspace(-0.01, 0.02, 12)
    lgbm = torch.linspace(0.015, -0.005, 12)
    index_return = torch.linspace(-0.02, 0.01, 12)
    risk_off = torch.ones(12)
    event_risk_on = torch.ones(2, dtype=torch.bool)
    cutoff = 7
    expected = _episode_reward(
        model,
        feature_values,
        event_positions,
        0,
        2,
        trend,
        lgbm,
        index_return,
        risk_off,
        event_risk_on,
        stochastic=False,
        daily_stop_position=cutoff,
    )
    changed_trend = trend.clone()
    changed_lgbm = lgbm.clone()
    changed_index = index_return.clone()
    changed_trend[cutoff:] = 10.0
    changed_lgbm[cutoff:] = -0.9
    changed_index[cutoff:] = 5.0
    actual = _episode_reward(
        model,
        feature_values,
        event_positions,
        0,
        2,
        changed_trend,
        changed_lgbm,
        changed_index,
        risk_off,
        event_risk_on,
        stochastic=False,
        daily_stop_position=cutoff,
    )

    assert torch.equal(expected, actual)


def test_saved_training_metadata_is_strictly_pre_prediction_year() -> None:
    payload_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "data"
        / "drl_adaptive_allocator_v1.json"
    )
    if not payload_path.exists():
        pytest.skip("research artifact is generated by the DRL backtest script")
    payload = __import__("json").loads(payload_path.read_text(encoding="utf-8"))
    for row in payload["training_audit"]:
        assert pd.Timestamp(row["training_end"]) < pd.Timestamp(
            f"{row['prediction_year']}-01-01"
        )
        assert pd.Timestamp(row["maximum_training_return_date"]) < pd.Timestamp(
            f"{row['prediction_year']}-01-01"
        )
    assert payload["production_change"] is False
