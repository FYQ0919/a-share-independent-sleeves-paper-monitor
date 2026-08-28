"""Causal risk and liquidity factors based on daily OHLCV data.

The module is intentionally separate from :mod:`app.framework_factors`.  It
contains range-volatility estimators and liquidity diagnostics which are not
the same statistics as the existing Parkinson, Amihud, downside-volatility,
or amount-ratio features.

All raw statistics are calculated after sorting by ``code`` and ``date``.
Rolling windows therefore use the current observation and observations that
precede it for the same security only.  The returned ``x_*`` columns are
cross-sectional percentile ranks in ``[-1, 1]`` for each date.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


WINDOWS = (20, 60)


RISK_LIQUIDITY_FACTOR_SPECS: dict[str, dict[str, str]] = {}

for _window in WINDOWS:
    RISK_LIQUIDITY_FACTOR_SPECS.update(
        {
            f"x_rogers_satchell_vol_{_window}": {
                "family": "risk_range_volatility",
                "origin": "rogers_satchell_1991",
                "formula": (
                    f"sqrt(mean(log(high/open)*(log(high/open)-log(close/open)) "
                    f"+ log(low/open)*(log(low/open)-log(close/open)), {_window}))"
                ),
            },
            f"x_garman_klass_vol_{_window}": {
                "family": "risk_range_volatility",
                "origin": "garman_klass_1980",
                "formula": (
                    f"sqrt(mean(0.5*log(high/low)^2 "
                    f"- (2*log(2)-1)*log(close/open)^2, {_window}))"
                ),
            },
            f"x_downside_upside_vol_ratio_{_window}": {
                "family": "risk_asymmetry",
                "origin": "semi_variance_ratio",
                "formula": (
                    f"sqrt(mean(min(return_1,0)^2, {_window})) / "
                    f"sqrt(mean(max(return_1,0)^2, {_window}))"
                ),
            },
            f"x_tail_loss_frequency_{_window}": {
                "family": "risk_tail",
                "origin": "empirical_lower_tail_frequency",
                "formula": (
                    f"mean(return_1 <= rolling_quantile(return_1, 5%, {_window}), "
                    f"{_window})"
                ),
            },
            f"x_corwin_schultz_spread_{_window}": {
                "family": "liquidity_spread",
                "origin": "corwin_schultz_2012",
                "formula": (
                    f"mean(2*(exp(alpha)-1)/(1+exp(alpha)), {_window}); "
                    "alpha=(sqrt(2*beta)-sqrt(beta))/(3-2*sqrt(2)) "
                    "- sqrt(gamma/(3-2*sqrt(2)))"
                ),
            },
            f"x_zero_return_ratio_{_window}": {
                "family": "liquidity_trading_friction",
                "origin": "zero_return_illiquidity_proxy",
                "formula": f"mean(abs(return_1) <= 1e-12, {_window})",
            },
            f"x_amount_shock_stability_{_window}": {
                "family": "liquidity_stability",
                "origin": "liquidity_shock_dispersion",
                "formula": (
                    f"1 / (1 + std(log(amount / lag(amount)), {_window}))"
                ),
            },
        }
    )


def risk_liquidity_factor_metadata(feature: str) -> dict[str, str] | None:
    """Return an auditable copy of metadata for one factor name."""

    metadata = RISK_LIQUIDITY_FACTOR_SPECS.get(feature)
    return dict(metadata) if metadata else None


def _rolling_mean(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    return values.groupby(codes, sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).mean()
    )


def _rolling_std(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    return values.groupby(codes, sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).std(ddof=0)
    )


def _rolling_quantile(
    values: pd.Series, codes: pd.Series, window: int, quantile: float
) -> pd.Series:
    return values.groupby(codes, sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).quantile(quantile)
    )


def _cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    ranked = numeric.groupby(dates, sort=False).rank(pct=True)
    return ranked.sub(0.5).mul(2).clip(-1.0, 1.0)


def add_risk_liquidity_factors(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Append causal range-risk and liquidity factors to a daily OHLCV frame.

    Required columns are ``code``, ``date``, ``open``, ``high``, ``low``,
    ``close``, ``volume`` and ``amount``.  Prices and amounts must be positive
    for a row to contribute to a statistic.  A full rolling window is required
    so a partially observed history cannot silently produce a factor value.
    """

    required = {
        "code",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"风险流动性因子缺少输入列: {', '.join(missing)}")

    output = frame.copy().sort_values(
        ["code", "date"], kind="mergesort"
    ).reset_index(drop=True)
    codes = output["code"]
    grouped = output.groupby("code", sort=False, group_keys=False)

    numeric = {
        column: pd.to_numeric(output[column], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount")
    }
    open_price = numeric["open"].where(numeric["open"].gt(0))
    high_price = numeric["high"].where(numeric["high"].gt(0))
    low_price = numeric["low"].where(numeric["low"].gt(0))
    close_price = numeric["close"].where(numeric["close"].gt(0))
    previous_close = numeric["close"].groupby(codes, sort=False).shift(1)
    previous_close = previous_close.where(previous_close.gt(0))
    returns = close_price.div(previous_close).sub(1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_high_open = np.log(high_price.div(open_price))
        log_low_open = np.log(low_price.div(open_price))
        log_close_open = np.log(close_price.div(open_price))
        log_high_low = np.log(high_price.div(low_price))

    # Rogers-Satchell is directional and uses open, high, low and close.  The
    # non-negative clamp is only a numerical/data-quality guard; valid OHLC
    # observations have a non-negative RS contribution.
    rogers_satchell_daily = (
        log_high_open.mul(log_high_open.sub(log_close_open))
        + log_low_open.mul(log_low_open.sub(log_close_open))
    )
    # Garman-Klass can have a negative single-day estimate.  As with the usual
    # volatility estimator, take the non-negative part after aggregation.
    garman_klass_daily = log_high_low.pow(2).mul(0.5).sub(
        log_close_open.pow(2).mul(2.0 * math.log(2.0) - 1.0)
    )

    upside = returns.clip(lower=0.0)
    downside = returns.clip(upper=0.0)
    amount = numeric["amount"].where(numeric["amount"].gt(0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_amount = np.log(amount)
    amount_shock = log_amount.groupby(codes, sort=False).diff()

    # Corwin-Schultz estimates a spread from two-day high-low ranges.  Its
    # alpha-to-spread transform may be slightly negative on noisy OHLC data;
    # the published quantity is a non-negative spread estimate.
    previous_log_high_low = log_high_low.groupby(codes, sort=False).shift(1)
    previous_high = numeric["high"].groupby(codes, sort=False).shift(1)
    previous_low = numeric["low"].groupby(codes, sort=False).shift(1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        beta = log_high_low.pow(2).add(previous_log_high_low.pow(2))
        two_day_high = pd.concat([high_price, previous_high], axis=1).max(axis=1)
        two_day_low = pd.concat([low_price, previous_low], axis=1).min(axis=1)
        gamma = np.log(two_day_high.div(two_day_low)).pow(2)
    corwin_denominator = 3.0 - 2.0 * math.sqrt(2.0)
    sqrt_beta = beta.clip(lower=0.0).pow(0.5)
    alpha = sqrt_beta.mul(math.sqrt(2.0) - 1.0).div(corwin_denominator).sub(
        gamma.div(corwin_denominator).clip(lower=0.0).pow(0.5)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exp_alpha = np.exp(alpha.clip(-50.0, 50.0))
        corwin_schultz_daily = exp_alpha.sub(1.0).mul(2.0).div(exp_alpha.add(1.0))
    corwin_schultz_daily = corwin_schultz_daily.clip(lower=0.0, upper=2.0)

    raw: dict[str, pd.Series] = {}
    for window in WINDOWS:
        rs_mean = _rolling_mean(rogers_satchell_daily, codes, window)
        raw[f"rogers_satchell_vol_{window}"] = rs_mean.clip(lower=0.0).pow(0.5)

        gk_mean = _rolling_mean(garman_klass_daily, codes, window)
        raw[f"garman_klass_vol_{window}"] = gk_mean.clip(lower=0.0).pow(0.5)

        downside_rms = _rolling_mean(downside.pow(2), codes, window).pow(0.5)
        upside_rms = _rolling_mean(upside.pow(2), codes, window).pow(0.5)
        raw[f"downside_upside_vol_ratio_{window}"] = downside_rms.div(
            upside_rms.where(upside_rms.gt(0))
        )

        tail_cutoff = _rolling_quantile(returns, codes, window, 0.05)
        tail_events = returns.le(tail_cutoff).astype(float).where(
            returns.notna() & tail_cutoff.notna()
        )
        raw[f"tail_loss_frequency_{window}"] = _rolling_mean(
            tail_events, codes, window
        )

        raw[f"corwin_schultz_spread_{window}"] = _rolling_mean(
            corwin_schultz_daily, codes, window
        )

        zero_return = returns.abs().le(1e-12).astype(float).where(returns.notna())
        raw[f"zero_return_ratio_{window}"] = _rolling_mean(
            zero_return, codes, window
        )

        shock_dispersion = _rolling_std(amount_shock, codes, window)
        raw[f"amount_shock_stability_{window}"] = 1.0 / (
            1.0 + shock_dispersion
        )

    feature_columns: list[str] = []
    for raw_name, values in raw.items():
        feature_name = f"x_{raw_name}"
        if feature_name not in RISK_LIQUIDITY_FACTOR_SPECS:
            raise RuntimeError(f"风险流动性因子缺少元数据: {feature_name}")
        output[feature_name] = _cross_section_rank(values, output["date"])
        feature_columns.append(feature_name)
    return output, feature_columns
