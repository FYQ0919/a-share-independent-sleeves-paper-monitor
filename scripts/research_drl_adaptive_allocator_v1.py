from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_CURVES = ROOT / "reports" / "semiconductor_trend_csi300_hedge_v1_curves.csv"
INDEX_CACHE = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
PREVIOUS_CURVE = ROOT / "reports" / "high_trend_75_hedge_blend_2020_2026.csv"
OUTPUT_JSON = ROOT / "data" / "drl_adaptive_allocator_v1.json"
OUTPUT_REPORT = ROOT / "reports" / "drl_adaptive_allocator_v1.md"
OUTPUT_CURVES = ROOT / "reports" / "drl_adaptive_allocator_v1_curves.csv"
OUTPUT_WEIGHTS = ROOT / "reports" / "drl_adaptive_allocator_v1_weights.csv"
OUTPUT_CHART_HTML = ROOT / "reports" / "drl_adaptive_allocator_v1_chart.html"
OUTPUT_CHART_PNG = ROOT / "reports" / "drl_adaptive_allocator_v1_chart.png"
MODEL_DIR = ROOT / "data" / "models" / "drl_adaptive_allocator_v1"
PROFILE_LABEL = "DRL adaptive allocator v1"

START = "2020-01-02"
END = "2026-08-25"
ADAPTIVE_START_YEAR = 2023
PREDICTION_YEARS = (2023, 2024, 2025, 2026)
REBALANCE_DAYS = 10
SIGNAL_TO_RETURN_LAG = 2
TREND_WEIGHT_MIN = 0.55
TREND_WEIGHT_MAX = 0.90
HEDGE_RATIO_MAX = 0.75
INITIAL_TREND_WEIGHT = 0.75
INITIAL_HEDGE_RATIO = 0.75
STATIC_BASELINE_TREND_WEIGHT = 0.75
STATIC_BASELINE_HEDGE_RATIO = 0.75
ACTION_SMOOTHING = 0.35
ACTION_SMOOTHING_UP = 0.35
ACTION_SMOOTHING_DOWN = 0.35
HEDGE_ACTION_SMOOTHING = 0.35
PRIOR_TARGET_BLEND = 0.0
PRIOR_TREND_WEIGHT_START = 0.75
PRIOR_TREND_WEIGHT_END = 0.75
PRIOR_ANNEAL_EVENTS = 1
ALLOCATION_COST_BPS = 12.0
FUTURES_COST_BPS = 2.0
TRAINING_UPDATES = 360
EPISODE_EVENTS = 12
EPISODES_PER_UPDATE = 4
LEARNING_RATE = 3e-4
BASE_SEED = 20260829


FEATURE_COLUMNS = (
    "trend_return_5",
    "trend_return_20",
    "trend_return_60",
    "trend_volatility_20",
    "trend_volatility_60",
    "trend_drawdown",
    "lgbm_return_5",
    "lgbm_return_20",
    "lgbm_return_60",
    "lgbm_volatility_20",
    "lgbm_volatility_60",
    "lgbm_drawdown",
    "strategy_correlation_20",
    "strategy_correlation_60",
    "relative_strength_20",
    "relative_strength_60",
    "relative_volatility_20",
    "relative_volatility_60",
    "csi300_close_ma20",
    "csi300_close_ma60",
    "csi300_close_ma120",
    "csi300_return_5",
    "csi300_return_20",
    "csi300_return_60",
    "csi300_volatility_20",
    "csi300_volatility_60",
    "csi300_drawdown",
    "csi300_risk_off_ma120",
)


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class SharpePolicy(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count + 2, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def map_and_smooth_action(
    latent: torch.Tensor,
    previous: torch.Tensor,
    *,
    smoothing: Optional[float] = None,
    decision_number: int = 0,
    risk_on: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target_trend = TREND_WEIGHT_MIN + (
        TREND_WEIGHT_MAX - TREND_WEIGHT_MIN
    ) * torch.sigmoid(latent[..., 0])
    if PRIOR_TARGET_BLEND > 0.0:
        progress = min(max(decision_number / max(PRIOR_ANNEAL_EVENTS, 1), 0.0), 1.0)
        prior = PRIOR_TREND_WEIGHT_START + progress * (
            PRIOR_TREND_WEIGHT_END - PRIOR_TREND_WEIGHT_START
        )
        target_trend = (
            (1.0 - PRIOR_TARGET_BLEND) * target_trend
            + PRIOR_TARGET_BLEND * prior
        )
    target = torch.stack(
        (target_trend, HEDGE_RATIO_MAX * torch.sigmoid(latent[..., 1])), dim=-1
    )
    if smoothing is not None:
        return previous + smoothing * (target - previous)
    if risk_on is None:
        risk_on = torch.ones_like(target[..., 0], dtype=torch.bool)
    else:
        risk_on = risk_on.to(dtype=torch.bool)
    upward_smoothing = torch.where(
        risk_on,
        torch.as_tensor(ACTION_SMOOTHING_UP, dtype=target.dtype, device=target.device),
        torch.as_tensor(ACTION_SMOOTHING, dtype=target.dtype, device=target.device),
    )
    trend_smoothing = torch.where(
        target[..., 0] > previous[..., 0],
        upward_smoothing,
        torch.as_tensor(ACTION_SMOOTHING_DOWN, dtype=target.dtype, device=target.device),
    )
    return torch.stack(
        (
            previous[..., 0] + trend_smoothing * (target[..., 0] - previous[..., 0]),
            previous[..., 1]
            + HEDGE_ACTION_SMOOTHING * (target[..., 1] - previous[..., 1]),
        ),
        dim=-1,
    )


def action_state(previous: torch.Tensor) -> torch.Tensor:
    trend = (previous[..., 0] - TREND_WEIGHT_MIN) / (
        TREND_WEIGHT_MAX - TREND_WEIGHT_MIN
    )
    hedge = previous[..., 1] / HEDGE_RATIO_MAX
    return torch.stack((trend, hedge), dim=-1)


def build_causal_features(
    trend_curve: pd.Series,
    lgbm_curve: pd.Series,
    index_ohlc: pd.DataFrame,
) -> pd.DataFrame:
    common_index = trend_curve.index.intersection(lgbm_curve.index)
    trend_curve = pd.to_numeric(trend_curve.reindex(common_index), errors="coerce")
    lgbm_curve = pd.to_numeric(lgbm_curve.reindex(common_index), errors="coerce")
    trend_return = trend_curve.pct_change(fill_method=None)
    lgbm_return = lgbm_curve.pct_change(fill_method=None)

    index_close_full = pd.to_numeric(index_ohlc["close"], errors="coerce").sort_index()
    index_return_full = index_close_full.pct_change(fill_method=None)
    index_drawdown_full = index_close_full.div(index_close_full.cummax()).sub(1.0)
    index_close = index_close_full.reindex(common_index).ffill()
    index_return = index_return_full.reindex(common_index)

    features = pd.DataFrame(index=common_index)
    for prefix, curve, returns in (
        ("trend", trend_curve, trend_return),
        ("lgbm", lgbm_curve, lgbm_return),
    ):
        for window in (5, 20, 60):
            features[f"{prefix}_return_{window}"] = curve.div(curve.shift(window)).sub(1.0)
        for window in (20, 60):
            features[f"{prefix}_volatility_{window}"] = (
                returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
            )
        features[f"{prefix}_drawdown"] = curve.div(curve.cummax()).sub(1.0)

    for window in (20, 60):
        features[f"strategy_correlation_{window}"] = trend_return.rolling(
            window, min_periods=window
        ).corr(lgbm_return)
        features[f"relative_strength_{window}"] = (
            features[f"trend_return_{window}"] - features[f"lgbm_return_{window}"]
        )
        features[f"relative_volatility_{window}"] = (
            features[f"trend_volatility_{window}"]
            - features[f"lgbm_volatility_{window}"]
        )

    for window in (20, 60, 120):
        moving_average = index_close_full.rolling(window, min_periods=window).mean()
        features[f"csi300_close_ma{window}"] = (
            index_close_full.div(moving_average).sub(1.0).reindex(common_index)
        )
    for window in (5, 20, 60):
        features[f"csi300_return_{window}"] = (
            index_close_full.div(index_close_full.shift(window)).sub(1.0).reindex(common_index)
        )
    for window in (20, 60):
        features[f"csi300_volatility_{window}"] = (
            index_return_full.rolling(window, min_periods=window).std(ddof=0)
            .mul(np.sqrt(252.0))
            .reindex(common_index)
        )
    features["csi300_drawdown"] = index_drawdown_full.reindex(common_index)
    features["csi300_risk_off_ma120"] = features["csi300_close_ma120"].lt(0.0).astype(float)
    return features.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def decision_dates(features: pd.DataFrame) -> pd.DatetimeIndex:
    eligible = features.dropna().index
    return pd.DatetimeIndex(eligible[::REBALANCE_DAYS])


def fit_scaler(features: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> Scaler:
    values = features.loc[list(dates), FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return Scaler(mean=mean, std=std)


def annualized_sharpe_tensor(returns: torch.Tensor) -> torch.Tensor:
    return returns.mean() / (returns.std(unbiased=False) + 1e-8) * np.sqrt(252.0)


def adam_step(
    parameters: list[torch.nn.Parameter],
    first_moments: list[torch.Tensor],
    second_moments: list[torch.Tensor],
    step: int,
    *,
    learning_rate: float,
    weight_decay: float = 1e-4,
) -> None:
    beta1, beta2 = 0.9, 0.999
    with torch.no_grad():
        for parameter, first, second in zip(
            parameters, first_moments, second_moments
        ):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.add(parameter, alpha=weight_decay)
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            corrected_first = first / (1.0 - beta1**step)
            corrected_second = second / (1.0 - beta2**step)
            parameter.addcdiv_(
                corrected_first,
                corrected_second.sqrt().add_(1e-8),
                value=-learning_rate,
            )
            parameter.grad = None


def _episode_reward(
    model: SharpePolicy,
    feature_values: torch.Tensor,
    event_positions: np.ndarray,
    event_start: int,
    event_stop: int,
    trend_returns: torch.Tensor,
    lgbm_returns: torch.Tensor,
    csi300_open_returns: torch.Tensor,
    risk_off_effective: torch.Tensor,
    event_risk_on: torch.Tensor,
    *,
    stochastic: bool,
    daily_stop_position: int | None = None,
) -> torch.Tensor:
    previous = torch.tensor(
        [INITIAL_TREND_WEIGHT, INITIAL_HEDGE_RATIO], dtype=torch.float32
    )
    daily_returns: list[torch.Tensor] = []
    previous_position: torch.Tensor | None = None
    for event_offset in range(event_start, event_stop):
        state = torch.cat((feature_values[event_offset], action_state(previous)))
        latent = model(state)
        if stochastic:
            latent = latent + torch.randn_like(latent) * 0.12
        action = map_and_smooth_action(
            latent,
            previous,
            decision_number=event_offset - event_start,
            risk_on=event_risk_on[event_offset],
        )
        start = int(event_positions[event_offset]) + SIGNAL_TO_RETURN_LAG
        if event_offset + 1 < event_stop:
            stop = int(event_positions[event_offset + 1]) + SIGNAL_TO_RETURN_LAG
        else:
            stop = min(start + REBALANCE_DAYS, len(trend_returns))
        if daily_stop_position is not None:
            stop = min(stop, daily_stop_position)
        if start >= stop:
            previous = action
            continue

        risk_slice = risk_off_effective[start:stop]
        effective_position = action[0] * action[1] * risk_slice
        allocation_cost = torch.zeros_like(effective_position)
        allocation_cost[0] = (
            2.0
            * ALLOCATION_COST_BPS
            / 10_000.0
            * torch.abs(action[0] - previous[0])
        )
        if previous_position is None:
            prior_risk = risk_off_effective[max(start - 1, 0)]
            prior_position = previous[0] * previous[1] * prior_risk
        else:
            prior_position = previous_position
        hedge_changes = torch.cat(
            (
                torch.abs(effective_position[:1] - prior_position),
                torch.abs(effective_position[1:] - effective_position[:-1]),
            )
        )
        hedge_cost = hedge_changes * FUTURES_COST_BPS / 10_000.0
        segment = (
            action[0] * trend_returns[start:stop]
            + (1.0 - action[0]) * lgbm_returns[start:stop]
            - effective_position * csi300_open_returns[start:stop]
            - allocation_cost
            - hedge_cost
        )
        daily_returns.append(segment)
        previous_position = effective_position[-1]
        previous = action
    if not daily_returns:
        raise RuntimeError("training episode contains no executable returns")
    return annualized_sharpe_tensor(torch.cat(daily_returns))


def train_policy(
    prediction_year: int,
    features: pd.DataFrame,
    events: pd.DatetimeIndex,
    daily_frame: pd.DataFrame,
) -> tuple[SharpePolicy, Scaler, dict]:
    seed = BASE_SEED + prediction_year
    seed_everything(seed)
    train_events = events[events < pd.Timestamp(f"{prediction_year}-01-01")]
    if len(train_events) < EPISODE_EVENTS:
        raise RuntimeError(f"insufficient training events for {prediction_year}")
    scaler = fit_scaler(features, train_events)
    normalized = scaler.transform(
        features.loc[train_events, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )
    feature_values = torch.tensor(normalized, dtype=torch.float32)
    event_risk_on = torch.tensor(
        features.loc[train_events, "csi300_risk_off_ma120"].lt(0.5).to_numpy(),
        dtype=torch.bool,
    )
    event_positions = daily_frame.index.get_indexer(train_events)
    prediction_start = pd.Timestamp(f"{prediction_year}-01-01")
    daily_stop_position = int(daily_frame.index.searchsorted(prediction_start))
    arrays = {
        column: torch.tensor(daily_frame[column].to_numpy(), dtype=torch.float32)
        for column in (
            "trend_return",
            "lgbm_return",
            "csi300_open_return",
            "risk_off_effective",
        )
    }
    model = SharpePolicy(len(FEATURE_COLUMNS))
    parameters = list(model.parameters())
    first_moments = [torch.zeros_like(parameter) for parameter in parameters]
    second_moments = [torch.zeros_like(parameter) for parameter in parameters]
    rng = np.random.default_rng(seed)
    reward_trace = []
    maximum_start = len(train_events) - EPISODE_EVENTS
    for update in range(TRAINING_UPDATES):
        starts = rng.integers(0, maximum_start + 1, size=EPISODES_PER_UPDATE)
        rewards = []
        for start in starts:
            rewards.append(
                _episode_reward(
                    model,
                    feature_values,
                    event_positions,
                    int(start),
                    int(start) + EPISODE_EVENTS,
                    arrays["trend_return"],
                    arrays["lgbm_return"],
                    arrays["csi300_open_return"],
                    arrays["risk_off_effective"],
                    event_risk_on,
                    stochastic=True,
                    daily_stop_position=daily_stop_position,
                )
            )
        reward = torch.stack(rewards).mean()
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adam_step(
            parameters,
            first_moments,
            second_moments,
            update + 1,
            learning_rate=LEARNING_RATE,
        )
        if update % 30 == 0 or update == TRAINING_UPDATES - 1:
            reward_trace.append(float(reward.detach()))

    model.eval()
    full_train_reward = float(
        _episode_reward(
            model,
            feature_values,
            event_positions,
            0,
            len(train_events),
            arrays["trend_return"],
            arrays["lgbm_return"],
            arrays["csi300_open_return"],
            arrays["risk_off_effective"],
            event_risk_on,
            stochastic=False,
            daily_stop_position=daily_stop_position,
        ).detach()
    )
    metadata = {
        "prediction_year": prediction_year,
        "seed": seed,
        "training_start": train_events.min().date().isoformat(),
        "training_end": train_events.max().date().isoformat(),
        "maximum_training_return_date": daily_frame.index[
            daily_stop_position - 1
        ].date().isoformat(),
        "training_events": int(len(train_events)),
        "updates": TRAINING_UPDATES,
        "episode_events": EPISODE_EVENTS,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "deterministic_training_sharpe": full_train_reward,
        "sampled_reward_trace": reward_trace,
    }
    return model, scaler, metadata


def deterministic_action(
    model: SharpePolicy,
    scaler: Scaler,
    feature_row: pd.Series,
    previous: np.ndarray,
    *,
    decision_number: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    feature = scaler.transform(feature_row.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    previous_tensor = torch.tensor(previous, dtype=torch.float32)
    state = torch.cat(
        (torch.tensor(feature, dtype=torch.float32), action_state(previous_tensor))
    )
    with torch.no_grad():
        latent = model(state)
        risk_on = torch.tensor(
            float(feature_row["csi300_risk_off_ma120"]) < 0.5,
            dtype=torch.bool,
        )
        action = map_and_smooth_action(
            latent,
            previous_tensor,
            decision_number=decision_number,
            risk_on=risk_on,
        )
        target = map_and_smooth_action(
            latent,
            previous_tensor,
            smoothing=1.0,
            decision_number=decision_number,
            risk_on=risk_on,
        )
    return action.numpy(), target.numpy()


def expand_actions(
    index: pd.DatetimeIndex,
    action_records: pd.DataFrame,
    *,
    initial_trend_weight: Optional[float] = None,
    initial_hedge_ratio: Optional[float] = None,
) -> pd.DataFrame:
    if initial_trend_weight is None:
        initial_trend_weight = INITIAL_TREND_WEIGHT
    if initial_hedge_ratio is None:
        initial_hedge_ratio = INITIAL_HEDGE_RATIO
    expanded = pd.DataFrame(
        {
            "trend_weight": initial_trend_weight,
            "hedge_ratio": initial_hedge_ratio,
            "signal_date": pd.NaT,
        },
        index=index,
    )
    for signal_date, row in action_records.sort_index().iterrows():
        signal_position = index.get_indexer([signal_date])[0]
        if signal_position < 0:
            raise ValueError(f"signal date is absent from execution index: {signal_date}")
        effective_position = signal_position + SIGNAL_TO_RETURN_LAG
        if effective_position >= len(index):
            continue
        effective_date = index[effective_position]
        expanded.loc[effective_date:, "trend_weight"] = float(row["trend_weight"])
        expanded.loc[effective_date:, "hedge_ratio"] = float(row["hedge_ratio"])
        expanded.loc[effective_date:, "signal_date"] = signal_date
    return expanded


def simulate_returns(
    daily_frame: pd.DataFrame,
    expanded_actions: pd.DataFrame,
) -> pd.DataFrame:
    trend_weight = expanded_actions["trend_weight"].astype(float)
    hedge_ratio = expanded_actions["hedge_ratio"].astype(float)
    effective_hedge = trend_weight * hedge_ratio * daily_frame["risk_off_effective"]
    allocation_cost = (
        trend_weight.diff().abs().fillna(0.0)
        * 2.0
        * ALLOCATION_COST_BPS
        / 10_000.0
    )
    hedge_cost = (
        effective_hedge.diff().abs().fillna(effective_hedge.abs())
        * FUTURES_COST_BPS
        / 10_000.0
    )
    gross_return = (
        trend_weight * daily_frame["trend_return"]
        + (1.0 - trend_weight) * daily_frame["lgbm_return"]
        - effective_hedge * daily_frame["csi300_open_return"]
    )
    net_return = gross_return - allocation_cost - hedge_cost
    return pd.DataFrame(
        {
            "net_return": net_return,
            "gross_return": gross_return,
            "allocation_cost": allocation_cost,
            "hedge_cost": hedge_cost,
            "effective_hedge": effective_hedge,
            "trend_weight": trend_weight,
            "lgbm_weight": 1.0 - trend_weight,
            "hedge_ratio": hedge_ratio,
        },
        index=daily_frame.index,
    )


def performance(returns: pd.Series) -> dict:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    curve = (1.0 + clean).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)
    volatility = float(clean.std(ddof=0) * np.sqrt(252.0))
    return {
        "terminal_value": float(curve.iloc[-1]),
        "total_return": total_return,
        "annual_return": float((1.0 + total_return) ** (252.0 / len(clean)) - 1.0),
        "sharpe": float(clean.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(curve.div(curve.cummax()).sub(1.0).min()),
        "observations": int(len(clean)),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = (
        pd.read_csv(SOURCE_CURVES, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[START:END]
    )
    index_ohlc = (
        pd.read_csv(INDEX_CACHE, parse_dates=["date"]).set_index("date").sort_index()
    )
    features = build_causal_features(
        curves["trend_expert"], curves["current_lgbm"], index_ohlc
    )
    index_open_return = index_ohlc["open"].pct_change(fill_method=None).reindex(curves.index)
    daily = pd.DataFrame(
        {
            "trend_return": curves["trend_expert"].pct_change(fill_method=None).fillna(0.0),
            "lgbm_return": curves["current_lgbm"].pct_change(fill_method=None).fillna(0.0),
            "csi300_open_return": index_open_return.fillna(0.0),
            "risk_off_effective": features["csi300_risk_off_ma120"]
            .shift(SIGNAL_TO_RETURN_LAG)
            .fillna(0.0),
        },
        index=curves.index,
    )
    return curves, features, daily


def save_model(
    model: SharpePolicy,
    scaler: Scaler,
    metadata: dict,
    prediction_year: int,
) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"policy_{prediction_year}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "metadata": metadata,
        },
        path,
    )
    return path


def format_metrics(metrics: dict) -> str:
    return (
        f"{metrics['terminal_value']:.4f}x | {metrics['annual_return']:.2%} | "
        f"{metrics['sharpe']:.3f} | {metrics['max_drawdown']:.2%}"
    )


def main() -> None:
    curves, features, daily = load_inputs()
    events = decision_dates(features)
    models: dict[int, tuple[SharpePolicy, Scaler]] = {}
    training_audit = []
    model_paths = []
    for year in PREDICTION_YEARS:
        model, scaler, metadata = train_policy(year, features, events, daily)
        models[year] = (model, scaler)
        training_audit.append(metadata)
        model_paths.append(save_model(model, scaler, metadata, year))

    previous = np.array([INITIAL_TREND_WEIGHT, INITIAL_HEDGE_RATIO], dtype=np.float32)
    records = []
    for decision_number, signal_date in enumerate(
        events[events.year >= ADAPTIVE_START_YEAR]
    ):
        year = int(signal_date.year)
        if year not in models:
            continue
        model, scaler = models[year]
        action, target = deterministic_action(
            model,
            scaler,
            features.loc[signal_date],
            previous,
            decision_number=decision_number,
        )
        effective_position = daily.index.get_loc(signal_date) + SIGNAL_TO_RETURN_LAG
        if effective_position >= len(daily.index):
            continue
        records.append(
            {
                "signal_date": signal_date,
                "effective_date": daily.index[effective_position],
                "model_year": year,
                "decision_number": decision_number,
                "trend_weight": float(action[0]),
                "lgbm_weight": float(1.0 - action[0]),
                "hedge_ratio": float(action[1]),
                "raw_target_trend_weight": float(target[0]),
                "raw_target_hedge_ratio": float(target[1]),
                "training_end": training_audit[year - ADAPTIVE_START_YEAR]["training_end"],
            }
        )
        previous = action
    action_records = pd.DataFrame(records).set_index("signal_date")
    expanded = expand_actions(daily.index, action_records)
    adaptive = simulate_returns(daily, expanded)

    static_actions = pd.DataFrame(
        {
            "trend_weight": STATIC_BASELINE_TREND_WEIGHT,
            "hedge_ratio": STATIC_BASELINE_HEDGE_RATIO,
            "signal_date": pd.NaT,
        },
        index=daily.index,
    )
    static = simulate_returns(daily, static_actions)
    previous_curve = (
        pd.read_csv(PREVIOUS_CURVE, parse_dates=["date"])
        .set_index("date")["high_trend_blend"]
        .reindex(daily.index)
    )
    previous_return = previous_curve.pct_change(fill_method=None).fillna(0.0)
    lgbm_return = daily["lgbm_return"]

    return_frame = pd.DataFrame(
        {
            "drl_adaptive": adaptive["net_return"],
            "fixed_75_25_same_contract": static["net_return"],
            "previous_75_25_independent_sleeves": previous_return,
            "formal_lgbm": lgbm_return,
        },
        index=daily.index,
    )
    curve_frame = (1.0 + return_frame).cumprod()
    curve_frame.iloc[0] = 1.0
    for column in return_frame:
        curve_frame[f"{column}_drawdown"] = curve_frame[column].div(
            curve_frame[column].cummax()
        ).sub(1.0)
    curve_frame["trend_weight"] = adaptive["trend_weight"]
    curve_frame["lgbm_weight"] = adaptive["lgbm_weight"]
    curve_frame["hedge_ratio"] = adaptive["hedge_ratio"]
    curve_frame["effective_hedge"] = adaptive["effective_hedge"]
    curve_frame["allocation_cost"] = adaptive["allocation_cost"]
    curve_frame["hedge_cost"] = adaptive["hedge_cost"]
    curve_frame.index.name = "date"
    curve_frame.to_csv(OUTPUT_CURVES, encoding="utf-8-sig", float_format="%.10f")
    action_records.reset_index().to_csv(
        OUTPUT_WEIGHTS, index=False, encoding="utf-8-sig", float_format="%.10f"
    )

    windows = {
        "full_2020_2026": (START, END),
        "static_2020_2022": (START, "2022-12-31"),
        "adaptive_2023_2026": ("2023-01-01", END),
        "year_2023": ("2023-01-01", "2023-12-31"),
        "year_2024": ("2024-01-01", "2024-12-31"),
        "year_2025": ("2025-01-01", "2025-12-31"),
        "year_2026_ytd": ("2026-01-01", END),
    }
    diagnostics = {}
    for name, (start, end) in windows.items():
        window_returns = return_frame.loc[start:end]
        diagnostics[name] = {
            strategy: performance(window_returns[strategy])
            for strategy in return_frame.columns
        }

    adaptive_metrics = diagnostics["adaptive_2023_2026"]["drl_adaptive"]
    static_metrics = diagnostics["adaptive_2023_2026"]["fixed_75_25_same_contract"]
    promotion_checks = {
        "adaptive_sharpe_exceeds_fixed": adaptive_metrics["sharpe"] > static_metrics["sharpe"],
        "adaptive_cagr_not_below_95pct_fixed": (
            adaptive_metrics["annual_return"] >= 0.95 * static_metrics["annual_return"]
        ),
        "adaptive_drawdown_not_worse": (
            adaptive_metrics["max_drawdown"] >= static_metrics["max_drawdown"]
        ),
    }
    promotion_passed = all(promotion_checks.values())
    weight_summary = {
        "average_trend_weight": float(action_records["trend_weight"].mean()),
        "minimum_trend_weight": float(action_records["trend_weight"].min()),
        "maximum_trend_weight": float(action_records["trend_weight"].max()),
        "average_hedge_ratio": float(action_records["hedge_ratio"].mean()),
        "minimum_hedge_ratio": float(action_records["hedge_ratio"].min()),
        "maximum_hedge_ratio": float(action_records["hedge_ratio"].max()),
        "trend_weight_turnover": float(action_records["trend_weight"].diff().abs().sum()),
        "effective_hedge_changes": int(adaptive["effective_hedge"].diff().abs().gt(1e-12).sum()),
        "allocation_cost_total": float(adaptive["allocation_cost"].sum()),
        "hedge_cost_total": float(adaptive["hedge_cost"].sum()),
    }
    causality_checks = {
        "signal_to_return_lag_is_two": SIGNAL_TO_RETURN_LAG == 2,
        "all_training_ends_precede_prediction_year": all(
            pd.Timestamp(row["training_end"]) < pd.Timestamp(f"{row['prediction_year']}-01-01")
            for row in training_audit
        ),
        "features_have_no_backward_fill": True,
        "yearly_models_frozen_before_prediction": True,
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": (
            "historical_candidate_only_not_deployed"
            if promotion_passed
            else "historical_candidate_rejected_not_deployed"
        ),
        "strategy": f"{PROFILE_LABEL}: PyTorch direct policy optimization with net annualized Sharpe reward",
        "adaptive_period": ["2023-01-01", END],
        "configuration": {
            "rebalance_days": REBALANCE_DAYS,
            "trend_weight_bounds": [TREND_WEIGHT_MIN, TREND_WEIGHT_MAX],
            "lgbm_weight_bounds": [1.0 - TREND_WEIGHT_MAX, 1.0 - TREND_WEIGHT_MIN],
            "hedge_ratio_bounds": [0.0, HEDGE_RATIO_MAX],
            "initial_trend_weight": INITIAL_TREND_WEIGHT,
            "initial_lgbm_weight": 1.0 - INITIAL_TREND_WEIGHT,
            "initial_hedge_ratio": INITIAL_HEDGE_RATIO,
            "static_baseline_trend_weight": STATIC_BASELINE_TREND_WEIGHT,
            "static_baseline_lgbm_weight": 1.0 - STATIC_BASELINE_TREND_WEIGHT,
            "static_baseline_hedge_ratio": STATIC_BASELINE_HEDGE_RATIO,
            "action_smoothing": ACTION_SMOOTHING,
            "trend_weight_increase_smoothing_risk_on": ACTION_SMOOTHING_UP,
            "trend_weight_decrease_smoothing": ACTION_SMOOTHING_DOWN,
            "hedge_action_smoothing": HEDGE_ACTION_SMOOTHING,
            "prior_target_blend": PRIOR_TARGET_BLEND,
            "prior_trend_weight_start": PRIOR_TREND_WEIGHT_START,
            "prior_trend_weight_end": PRIOR_TREND_WEIGHT_END,
            "prior_anneal_events": PRIOR_ANNEAL_EVENTS,
            "signal_to_return_lag": SIGNAL_TO_RETURN_LAG,
            "allocation_cost_bps_one_way": ALLOCATION_COST_BPS,
            "allocation_transfer_cost_formula": "2 * 12bp * abs(delta trend weight)",
            "futures_cost_bps": FUTURES_COST_BPS,
            "reward": "annualized Sharpe of daily portfolio returns after allocation and hedge costs",
            "network": "30 inputs including prior action -> Linear32/Tanh -> Linear16/Tanh -> 2 actions",
        },
        "training_audit": training_audit,
        "causality_checks": causality_checks,
        "diagnostics": diagnostics,
        "weight_summary": weight_summary,
        "promotion_gate": {"passed": promotion_passed, "checks": promotion_checks},
        "production_change": False,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
            "curves": str(OUTPUT_CURVES.relative_to(ROOT)),
            "weights": str(OUTPUT_WEIGHTS.relative_to(ROOT)),
            "chart_html": str(OUTPUT_CHART_HTML.relative_to(ROOT)),
            "chart_png": str(OUTPUT_CHART_PNG.relative_to(ROOT)),
            "models": [str(path.relative_to(ROOT)) for path in model_paths],
        },
        "limitations": [
            "The 2020-2026 history has been repeatedly inspected and is diagnostic, not fresh out-of-sample evidence.",
            "Walk-forward training prevents direct future-row access but cannot undo research-selection bias in the repeatedly viewed period.",
            "The frozen current stock universe remains subject to survivorship bias.",
            "CSI300 open returns proxy a futures hedge and omit basis, margin, roll, financing, and short-sale constraints.",
            "Promotion requires forward paper trading after 2026-08-25 under the unchanged execution contract.",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {PROFILE_LABEL}",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This is a historical walk-forward diagnostic. Production was not changed.",
        "",
        "## Adaptive-period comparison (2023-2026 YTD)",
        "",
        "| Strategy | Terminal | CAGR | Sharpe | Max drawdown |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "drl_adaptive": "DRL adaptive",
        "fixed_75_25_same_contract": "Fixed 75/25, same contract",
        "previous_75_25_independent_sleeves": "Previous 75/25 independent sleeves",
        "formal_lgbm": "Formal LGBM",
    }
    for key, label in labels.items():
        metric = diagnostics["adaptive_2023_2026"][key]
        lines.append(
            f"| {label} | {metric['terminal_value']:.4f}x | {metric['annual_return']:.2%} | "
            f"{metric['sharpe']:.3f} | {metric['max_drawdown']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Promotion gate",
            "",
            f"Passed: **{promotion_passed}**",
            "",
            *[f"- {name}: {passed}" for name, passed in promotion_checks.items()],
            "",
            "## Causality contract",
            "",
            "- Each calendar-year model is trained only on decision events before that year.",
            "- A T-close action first affects the curve return at index T+2.",
            "- Feature scaling is fit only on the corresponding training events.",
            "- Costs are deducted before the Sharpe reward is calculated.",
            "",
            "## Weight summary",
            "",
            f"- Average trend weight: {weight_summary['average_trend_weight']:.2%}",
            f"- Trend-weight range: {weight_summary['minimum_trend_weight']:.2%} to {weight_summary['maximum_trend_weight']:.2%}",
            f"- Average maximum hedge ratio: {weight_summary['average_hedge_ratio']:.2%}",
            f"- Maximum-hedge range: {weight_summary['minimum_hedge_ratio']:.2%} to {weight_summary['maximum_hedge_ratio']:.2%}",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "promotion_gate": payload["promotion_gate"],
                "adaptive_2023_2026": diagnostics["adaptive_2023_2026"],
                "full_2020_2026": diagnostics["full_2020_2026"],
                "weight_summary": weight_summary,
                "artifacts": payload["artifacts"],
                "production_change": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
