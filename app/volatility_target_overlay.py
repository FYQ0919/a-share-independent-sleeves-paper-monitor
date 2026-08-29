from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityTargetConfig:
    """Frozen risk-budget parameters for the high-Sharpe research candidate."""

    window: int = 60
    min_observations: int = 30
    target_volatility: float = 0.22
    minimum_exposure: float = 0.65
    maximum_exposure: float = 1.0
    annualization_periods: int = 252
    exposure_change_cost_bps: float = 12.0

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least two sessions")
        if not 2 <= self.min_observations <= self.window:
            raise ValueError("min_observations must be between two and window")
        if self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if not 0 <= self.minimum_exposure <= self.maximum_exposure <= 1:
            raise ValueError("exposure bounds must satisfy 0 <= minimum <= maximum <= 1")
        if self.annualization_periods <= 0:
            raise ValueError("annualization_periods must be positive")
        if self.exposure_change_cost_bps < 0:
            raise ValueError("exposure_change_cost_bps cannot be negative")


class VolatilityTargetOverlay:
    """Causal cash overlay that scales aggregate long exposure without leverage."""

    def __init__(self, config: VolatilityTargetConfig | None = None):
        self.config = config or VolatilityTargetConfig()

    def exposure_for_realized_returns(self, returns: pd.Series) -> pd.Series:
        """Return exposure aligned with each realized return using prior data only."""

        clean = self._clean_returns(returns)
        observed_volatility = (
            clean.shift(1)
            .rolling(
                self.config.window,
                min_periods=self.config.min_observations,
            )
            .std(ddof=0)
            .mul(np.sqrt(self.config.annualization_periods))
        )
        raw = self.config.target_volatility / observed_volatility.replace(0.0, np.nan)
        return raw.clip(
            lower=self.config.minimum_exposure,
            upper=self.config.maximum_exposure,
        ).fillna(self.config.maximum_exposure)

    def next_session_exposure(self, realized_returns: pd.Series) -> float:
        """Calculate the next-session target using returns known through today."""

        clean = self._clean_returns(realized_returns)
        sample = clean.iloc[-self.config.window :]
        if len(sample) < self.config.min_observations:
            return self.config.maximum_exposure
        volatility = float(
            sample.std(ddof=0) * np.sqrt(self.config.annualization_periods)
        )
        if not np.isfinite(volatility) or volatility <= 0:
            return self.config.maximum_exposure
        return float(
            np.clip(
                self.config.target_volatility / volatility,
                self.config.minimum_exposure,
                self.config.maximum_exposure,
            )
        )

    def apply(self, returns: pd.Series) -> pd.DataFrame:
        """Apply lagged exposure and charge cost on every exposure change."""

        clean = self._clean_returns(returns)
        exposure = self.exposure_for_realized_returns(clean)
        turnover = exposure.diff().abs().fillna(exposure.abs())
        cost = turnover.mul(self.config.exposure_change_cost_bps / 10_000.0)
        managed_return = exposure.mul(clean).sub(cost)
        equity = (1.0 + managed_return).cumprod()
        return pd.DataFrame(
            {
                "base_return": clean,
                "risk_exposure": exposure,
                "exposure_turnover": turnover,
                "overlay_cost": cost,
                "managed_return": managed_return,
                "managed_equity": equity,
            },
            index=clean.index,
        )

    @staticmethod
    def scale_target_weights(
        target_weights: pd.Series | Iterable[float], exposure: float
    ) -> pd.Series:
        """Scale stock weights to the risk budget; the residual stays in cash."""

        values = (
            target_weights.astype(float).copy()
            if isinstance(target_weights, pd.Series)
            else pd.Series(list(target_weights), dtype=float)
        )
        if values.empty or values.lt(0).any() or not np.isfinite(values).all():
            raise ValueError("target weights must be finite, non-negative and non-empty")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("target weights must have a positive sum")
        if not 0 <= exposure <= 1:
            raise ValueError("exposure must be between zero and one")
        return values.div(total).mul(float(exposure))

    @staticmethod
    def _clean_returns(returns: pd.Series) -> pd.Series:
        clean = pd.to_numeric(returns, errors="coerce")
        if clean.isna().any() or not np.isfinite(clean).all():
            raise ValueError("returns must be finite and contain no missing values")
        if clean.empty or clean.le(-1.0).any():
            raise ValueError("returns must be non-empty and greater than -100%")
        return clean.astype(float)
