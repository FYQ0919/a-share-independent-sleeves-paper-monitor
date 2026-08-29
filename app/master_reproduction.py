from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Iterable

import numpy as np
import pandas as pd

from app.lgbm_features import build_alpha158_lite


LOOKBACK = 8
MARKET_WINDOWS = (5, 10, 20, 30, 60)
MARKET_PROXY_NAMES = ("broad", "liquid_two_thirds", "liquid_one_third")


def _weighted_return(group: pd.DataFrame) -> float:
    values = pd.to_numeric(group["ret_1"], errors="coerce")
    weights = pd.to_numeric(group["amount"], errors="coerce").clip(lower=0.0)
    valid = values.notna() & weights.notna()
    if not valid.any():
        return np.nan
    if float(weights.loc[valid].sum()) <= 0.0:
        return float(values.loc[valid].mean())
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def build_market_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build three causal market proxies analogous to the paper's three indices."""
    required = {"date", "code", "ret_1", "amount", "amount_60"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"market proxy input is missing columns: {', '.join(missing)}")

    data = frame[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["liquidity_rank"] = data.groupby("date")["amount_60"].rank(pct=True)
    proxy_thresholds = {
        "broad": 0.0,
        "liquid_two_thirds": 1.0 / 3.0,
        "liquid_one_third": 2.0 / 3.0,
    }
    daily_parts = []
    for proxy_name, threshold in proxy_thresholds.items():
        selected = data[data["liquidity_rank"].ge(threshold)].copy()
        daily = selected.groupby("date", sort=True).apply(
            lambda group: pd.Series(
                {
                    "return": _weighted_return(group),
                    "amount": float(
                        pd.to_numeric(group["amount"], errors="coerce")
                        .clip(lower=0.0)
                        .sum()
                    ),
                }
            ),
            include_groups=False,
        )
        daily.columns = [f"{proxy_name}_{column}" for column in daily.columns]
        daily_parts.append(daily)

    raw = pd.concat(daily_parts, axis=1).sort_index()
    features: dict[str, pd.Series] = {}
    for proxy_name in MARKET_PROXY_NAMES:
        returns = raw[f"{proxy_name}_return"]
        amount = raw[f"{proxy_name}_amount"].replace(0.0, np.nan)
        features[f"m_{proxy_name}_return_1"] = returns
        for window in MARKET_WINDOWS:
            features[f"m_{proxy_name}_return_mean_{window}"] = returns.rolling(
                window, min_periods=window
            ).mean()
            features[f"m_{proxy_name}_return_std_{window}"] = returns.rolling(
                window, min_periods=window
            ).std(ddof=0)
            features[f"m_{proxy_name}_amount_mean_ratio_{window}"] = (
                amount.rolling(window, min_periods=window).mean().div(amount)
            )
            features[f"m_{proxy_name}_amount_std_ratio_{window}"] = (
                amount.rolling(window, min_periods=window).std(ddof=0).div(amount)
            )
    output = pd.DataFrame(features, index=raw.index)
    output.index.name = "date"
    if output.shape[1] != 63:
        raise RuntimeError(f"expected 63 market features, got {output.shape[1]}")
    return output.replace([np.inf, -np.inf], np.nan)


def robust_scale_market_features(
    market: pd.DataFrame, training_end_exclusive: date | pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    cutoff = pd.Timestamp(training_end_exclusive)
    training = market.loc[market.index < cutoff]
    if training.empty:
        raise ValueError("market scaler has no training rows")
    median = training.median(skipna=True).fillna(0.0)
    mad = training.sub(median).abs().median(skipna=True).mul(1.4826)
    scale = mad.where(mad.gt(1e-8), 1.0).fillna(1.0)
    scaled = market.sub(median).div(scale).clip(-3.0, 3.0).fillna(0.0)
    audit = {
        column: {"median": float(median[column]), "scale": float(scale[column])}
        for column in market.columns
    }
    return scaled, audit


@dataclass
class MasterPanel:
    frame: pd.DataFrame
    feature_columns: list[str]
    market_columns: list[str]
    dates: pd.DatetimeIndex
    codes: list[str]
    feature_values: np.ndarray
    market_values: np.ndarray
    labels: np.ndarray
    label_end_dates: np.ndarray
    eligible: np.ndarray
    row_lookup: np.ndarray
    lookback: int = LOOKBACK

    def date_indices(self, start: date, end: date) -> list[int]:
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return np.flatnonzero(mask).astype(int).tolist()

    def training_date_indices(self, cutoff: date | pd.Timestamp) -> list[int]:
        boundary = np.datetime64(pd.Timestamp(cutoff).to_datetime64())
        indices = []
        for day_index in range(self.lookback - 1, len(self.dates)):
            mature = (
                self.eligible[day_index]
                & np.isfinite(self.labels[day_index])
                & ~np.isnat(self.label_end_dates[day_index])
                & (self.label_end_dates[day_index] < boundary)
            )
            if mature.sum() >= 10 and self.dates[day_index] < pd.Timestamp(cutoff):
                indices.append(day_index)
        return indices

    def batch(
        self,
        day_index: int,
        require_label: bool = False,
        label_cutoff: date | pd.Timestamp | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if day_index < self.lookback - 1:
            raise ValueError("not enough historical rows for MASTER lookback")
        mask = self.eligible[day_index].copy()
        if require_label:
            mask &= np.isfinite(self.labels[day_index])
        if label_cutoff is not None:
            boundary = np.datetime64(pd.Timestamp(label_cutoff).to_datetime64())
            mask &= ~np.isnat(self.label_end_dates[day_index])
            mask &= self.label_end_dates[day_index] < boundary
        stock_indices = np.flatnonzero(mask)
        sequence = self.feature_values[
            day_index - self.lookback + 1 : day_index + 1, stock_indices, :
        ].transpose(1, 0, 2)
        labels = self.labels[day_index, stock_indices]
        market = self.market_values[day_index]
        return sequence, market, labels, stock_indices


def build_master_panel(
    history: pd.DataFrame,
    training_end_exclusive: date | pd.Timestamp,
    lookback: int = LOOKBACK,
) -> tuple[MasterPanel, dict[str, dict[str, float]]]:
    frame, feature_columns = build_alpha158_lite(history)
    return build_master_panel_from_frame(
        frame, feature_columns, training_end_exclusive, lookback=lookback
    )


def build_master_panel_from_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    training_end_exclusive: date | pd.Timestamp,
    lookback: int = LOOKBACK,
) -> tuple[MasterPanel, dict[str, dict[str, float]]]:
    data = frame.copy().sort_values(["date", "code"]).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data["code"] = data["code"].astype(str)
    if data.duplicated(["date", "code"]).any():
        raise ValueError("MASTER panel requires unique date+code rows")
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    codes = sorted(data["code"].unique().tolist())
    full_index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    indexed = data.set_index(["date", "code"])

    feature_values = (
        indexed[feature_columns]
        .reindex(full_index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        .reshape(len(dates), len(codes), len(feature_columns))
    )
    labels = (
        pd.to_numeric(indexed["label_return"], errors="coerce")
        .reindex(full_index)
        .to_numpy(dtype=np.float32)
        .reshape(len(dates), len(codes))
    )
    label_end_dates = (
        pd.to_datetime(indexed["label_end_date"], errors="coerce")
        .reindex(full_index)
        .to_numpy(dtype="datetime64[ns]")
        .reshape(len(dates), len(codes))
    )

    required_signal_columns = [
        "ret_5",
        "ret_20",
        "ret_60",
        "volatility_20",
        "amount_20",
        "amount_60",
        "absolute_return_60",
        "ma_20",
    ]
    signal = indexed.reindex(full_index)
    eligible_series = (
        signal["tradestatus"].eq(1)
        & signal["is_st"].eq(0)
        & signal["close"].gt(1)
        & signal["amount_20"].ge(10_000_000)
        & signal[required_signal_columns].notna().all(axis=1)
    )
    if "universe_member" in signal.columns:
        eligible_series &= signal["universe_member"].fillna(False).astype(bool)
    eligible = eligible_series.fillna(False).to_numpy().reshape(len(dates), len(codes))

    row_series = pd.Series(np.arange(len(data), dtype=np.int64), index=pd.MultiIndex.from_frame(data[["date", "code"]]))
    row_lookup = (
        row_series.reindex(full_index)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
        .reshape(len(dates), len(codes))
    )
    market = build_market_proxy_features(data)
    market, market_scaler = robust_scale_market_features(
        market, training_end_exclusive
    )
    market = market.reindex(dates).fillna(0.0)
    panel = MasterPanel(
        frame=data,
        feature_columns=list(feature_columns),
        market_columns=market.columns.tolist(),
        dates=dates,
        codes=codes,
        feature_values=feature_values,
        market_values=market.to_numpy(dtype=np.float32),
        labels=labels,
        label_end_dates=label_end_dates,
        eligible=eligible,
        row_lookup=row_lookup,
        lookback=lookback,
    )
    return panel, market_scaler


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("MASTER reproduction requires PyTorch") from exc
    return torch, nn


torch, nn = _torch_modules()


class LightweightAdam:
    """Adam update without importing torch.optim's optional compiler stack."""

    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.first_moments = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second_moments = [torch.zeros_like(parameter) for parameter in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        self.step_count += 1
        first_correction = 1.0 - self.beta1**self.step_count
        second_correction = 1.0 - self.beta2**self.step_count
        with torch.no_grad():
            for parameter, first, second in zip(
                self.parameters, self.first_moments, self.second_moments
            ):
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
                second.mul_(self.beta2).addcmul_(
                    gradient, gradient, value=1.0 - self.beta2
                )
                denominator = second.sqrt().div_(math.sqrt(second_correction)).add_(
                    self.eps
                )
                parameter.addcdiv_(
                    first,
                    denominator,
                    value=-self.lr / first_correction,
                )


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = values + attended
        return hidden + self.ffn(self.norm2(hidden))


class TemporalAggregation(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.transform = nn.Linear(d_model, d_model, bias=False)

    def forward(self, values):
        hidden = self.transform(values)
        query = hidden[:, -1, :].unsqueeze(-1)
        weights = torch.softmax(torch.matmul(hidden, query).squeeze(-1), dim=1)
        return torch.matmul(weights.unsqueeze(1), values).squeeze(1)


class MasterReproductionModel(nn.Module):
    """Scaled structural reproduction of the public AAAI-2024 MASTER model."""

    def __init__(
        self,
        feature_dim: int,
        market_dim: int,
        d_model: int = 64,
        temporal_heads: int = 4,
        stock_heads: int = 2,
        dropout: float = 0.5,
        beta: float = 5.0,
        max_lookback: int = 32,
    ):
        super().__init__()
        if d_model % temporal_heads or d_model % stock_heads:
            raise ValueError("d_model must be divisible by both attention head counts")
        self.feature_dim = feature_dim
        self.market_dim = market_dim
        self.beta = beta
        self.market_gate = nn.Linear(market_dim, feature_dim)
        self.feature_projection = nn.Linear(feature_dim, d_model)
        position = torch.zeros(max_lookback, d_model)
        positions = torch.arange(max_lookback, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        position[:, 0::2] = torch.sin(positions * divisor)
        position[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("position", position)
        self.intra_stock = AttentionBlock(d_model, temporal_heads, dropout)
        self.inter_stock = AttentionBlock(d_model, stock_heads, dropout)
        self.temporal_aggregation = TemporalAggregation(d_model)
        self.decoder = nn.Linear(d_model, 1)

    def gate_weights(self, market):
        return self.feature_dim * torch.softmax(
            self.market_gate(market) / self.beta, dim=-1
        )

    def forward(self, stock_sequences, market):
        if market.ndim == 2:
            market = market[0]
        gated = stock_sequences * self.gate_weights(market).view(1, 1, -1)
        hidden = self.feature_projection(gated)
        hidden = hidden + self.position[: hidden.shape[1]].unsqueeze(0)
        hidden = self.intra_stock(hidden)
        by_time = hidden.transpose(0, 1)
        by_time = self.inter_stock(by_time)
        hidden = by_time.transpose(0, 1)
        return self.decoder(self.temporal_aggregation(hidden)).squeeze(-1)


def make_master_model(
    feature_dim: int,
    market_dim: int,
    seed: int,
    d_model: int = 64,
    dropout: float = 0.5,
    beta: float = 5.0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return MasterReproductionModel(
        feature_dim=feature_dim,
        market_dim=market_dim,
        d_model=d_model,
        dropout=dropout,
        beta=beta,
    )


def train_master_epoch(
    model,
    panel: MasterPanel,
    day_indices: Iterable[int],
    optimizer,
    label_cutoff: date | pd.Timestamp,
    seed: int,
) -> float:
    model.train()
    order = np.asarray(list(day_indices), dtype=int)
    np.random.default_rng(seed).shuffle(order)
    losses = []
    for day_index in order:
        sequence, market, labels, _ = panel.batch(
            int(day_index), require_label=True, label_cutoff=label_cutoff
        )
        if len(labels) < 10:
            continue
        x = torch.from_numpy(sequence)
        m = torch.from_numpy(market)
        y = torch.from_numpy(labels)
        sorted_indices = torch.argsort(y)
        trim = max(1, int(len(y) * 0.025)) if len(y) >= 20 else 0
        if trim:
            keep = sorted_indices[trim:-trim]
            x = x[keep]
            y = y[keep]
        y = (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-6)
        prediction = model(x, m)
        loss = torch.mean((prediction - y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else np.nan


def predict_master(
    model,
    panel: MasterPanel,
    day_indices: Iterable[int],
) -> pd.Series:
    predictions = pd.Series(np.nan, index=panel.frame.index, dtype=float)
    model.eval()
    with torch.no_grad():
        for day_index in day_indices:
            sequence, market, _, stock_indices = panel.batch(int(day_index))
            if len(stock_indices) < 2:
                continue
            values = model(
                torch.from_numpy(sequence), torch.from_numpy(market)
            ).detach().cpu().numpy()
            rows = panel.row_lookup[int(day_index), stock_indices]
            valid = rows >= 0
            predictions.iloc[rows[valid]] = values[valid]
    return predictions


def prediction_metrics(
    frame: pd.DataFrame,
    predictions: pd.Series,
    start: date,
    end: date,
    label_end_before: date | pd.Timestamp | None = None,
) -> dict[str, float | int]:
    data = frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end)),
        ["date", "label_return", "label_end_date"],
    ].copy()
    if label_end_before is not None:
        data = data[
            pd.to_datetime(data["label_end_date"], errors="coerce").lt(
                pd.Timestamp(label_end_before)
            )
        ]
    data["prediction"] = predictions.reindex(data.index)
    data = data.dropna(subset=["prediction", "label_return"])
    daily = data.groupby("date").apply(
        lambda group: pd.Series(
            {
                "ic": group["prediction"].corr(group["label_return"]),
                "rank_ic": group["prediction"].corr(
                    group["label_return"], method="spearman"
                ),
            }
        ),
        include_groups=False,
    ).dropna(how="all")
    return {
        "days": int(len(daily)),
        "mean_ic": float(daily["ic"].mean()),
        "ic_ir": float(daily["ic"].mean() / daily["ic"].std(ddof=0))
        if float(daily["ic"].std(ddof=0)) > 0
        else 0.0,
        "mean_rank_ic": float(daily["rank_ic"].mean()),
        "rank_ic_ir": float(
            daily["rank_ic"].mean() / daily["rank_ic"].std(ddof=0)
        )
        if float(daily["rank_ic"].std(ddof=0)) > 0
        else 0.0,
    }
