"""Barra-style residual, downside, impact and market-state factors.

This layer is deliberately independent from the existing Alpha158/Qlib and
``risk_liquidity_factors`` layers.  It only consumes daily OHLCV plus dollar
amount and recomputes returns from close prices.  The market return used by the
residual and conditional-beta factors is the equal-weight cross-sectional
mean of the same-date security returns.  A returned row is therefore a
post-close signal for the next tradable session; no observations after that
date enter its calculation.

The raw statistics are calculated in stable ``code``/``date`` order.  Every
``x_*`` output is then ranked cross-sectionally by signal date to the common
``[-1, 1]`` scale.  A complete rolling window is required for all factors so
that a short history is not silently treated as a valid risk estimate.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import numpy as np
import pandas as pd


WINDOWS = (20, 60)


# These statistics are intentionally distinct from the existing layers:
# residual risk is a rolling market regression residual, downside factors use
# loss magnitude rather than only a downside/upside ratio, and impact factors
# use signed-flow slope or an outlier frequency rather than an Amihud mean.
BARRA_RESIDUAL_FACTOR_SPECS: dict[str, dict[str, str]] = {
    "x_residual_vol_20": {
        "family": "residual_risk",
        "origin": "barra_style_market_residual",
        "formula": "sqrt(var(ret_i) - cov(ret_i, mkt)^2 / var(mkt), 20)",
    },
    "x_residual_vol_60": {
        "family": "residual_risk",
        "origin": "barra_style_market_residual",
        "formula": "sqrt(var(ret_i) - cov(ret_i, mkt)^2 / var(mkt), 60)",
    },
    "x_residual_downside_vol_20": {
        "family": "residual_risk",
        "origin": "barra_style_market_residual",
        "formula": "sqrt(mean(min(residual_20, 0)^2, 20))",
    },
    "x_residual_downside_vol_60": {
        "family": "residual_risk",
        "origin": "barra_style_market_residual",
        "formula": "sqrt(mean(min(residual_60, 0)^2, 60))",
    },
    "x_downside_deviation_20": {
        "family": "downside_risk",
        "origin": "semi_deviation",
        "formula": "sqrt(mean(min(ret_i, 0)^2, 20))",
    },
    "x_downside_deviation_60": {
        "family": "downside_risk",
        "origin": "semi_deviation",
        "formula": "sqrt(mean(min(ret_i, 0)^2, 60))",
    },
    "x_expected_shortfall_20": {
        "family": "downside_risk",
        "origin": "historical_expected_shortfall",
        "formula": "-mean(ret_i | ret_i <= quantile(ret_i, 20%), 20)",
    },
    "x_expected_shortfall_60": {
        "family": "downside_risk",
        "origin": "historical_expected_shortfall",
        "formula": "-mean(ret_i | ret_i <= quantile(ret_i, 20%), 60)",
    },
    "x_kyle_lambda_20": {
        "family": "transaction_impact",
        "origin": "kyle_lambda_1985_proxy",
        "formula": (
            "cov(ret_i, sign(ret_i)*amount/1e6) / "
            "var(sign(ret_i)*amount/1e6), 20"
        ),
    },
    "x_kyle_lambda_60": {
        "family": "transaction_impact",
        "origin": "kyle_lambda_1985_proxy",
        "formula": (
            "cov(ret_i, sign(ret_i)*amount/1e6) / "
            "var(sign(ret_i)*amount/1e6), 60"
        ),
    },
    "x_impact_shock_frequency_20": {
        "family": "transaction_impact",
        "origin": "liquidity_shock_proxy",
        "formula": (
            "mean(impact_i > 2*median(impact_i, 20), 20); "
            "impact=abs(ret_i)*sqrt(1e6/amount)"
        ),
    },
    "x_impact_shock_frequency_60": {
        "family": "transaction_impact",
        "origin": "liquidity_shock_proxy",
        "formula": (
            "mean(impact_i > 2*median(impact_i, 60), 60); "
            "impact=abs(ret_i)*sqrt(1e6/amount)"
        ),
    },
    "x_market_corr_60": {
        "family": "market_state_sensitivity",
        "origin": "barra_style_market_exposure",
        "formula": "corr(ret_i, mkt_ret_equal_weight, 60)",
    },
    "x_market_down_beta_60": {
        "family": "market_state_sensitivity",
        "origin": "conditional_market_beta",
        "formula": "cov(ret_i, mkt | mkt <= 0) / var(mkt | mkt <= 0), 60",
    },
    "x_market_up_beta_60": {
        "family": "market_state_sensitivity",
        "origin": "conditional_market_beta",
        "formula": "cov(ret_i, mkt | mkt > 0) / var(mkt | mkt > 0), 60",
    },
}

# ``SPECS`` is a small convenience alias used by some factor-batch tooling;
# retain the descriptive constant as the primary public name.
SPECS = BARRA_RESIDUAL_FACTOR_SPECS

_REQUIRED_COLUMNS = {
    "code",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}


def barra_residual_factor_metadata(feature: str) -> dict[str, str] | None:
    """Return a copy of one factor's auditable metadata, if registered."""

    metadata = BARRA_RESIDUAL_FACTOR_SPECS.get(feature)
    return dict(metadata) if metadata else None


# Keep a short generic spelling available for callers that use the same
# ``metadata`` convention as their dynamically loaded factor modules.
metadata = barra_residual_factor_metadata


def _group_indices(codes: pd.Series) -> Iterable[pd.Index]:
    """Yield row indexes for each instrument in the already sorted frame."""

    # Missing identifiers cannot form a causal per-security history.  Exclude
    # them from grouped calculations; their output rows remain NaN instead of
    # causing pandas' null-category grouping error.
    return codes.groupby(codes, sort=False, dropna=True).groups.values()


def _rolling_apply(
    values: pd.Series,
    codes: pd.Series,
    window: int,
    function: Callable[[np.ndarray], float],
) -> pd.Series:
    """Apply a complete-window statistic independently for each security."""

    result = pd.Series(np.nan, index=values.index, dtype=float)
    for indices in _group_indices(codes):
        group_values = values.loc[indices]
        result.loc[indices] = group_values.rolling(
            window, min_periods=window
        ).apply(function, raw=True)
    return result


def _rolling_mean(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    return _rolling_apply(values, codes, window, lambda array: float(np.mean(array)))


def _rolling_covariance(
    left: pd.Series,
    right: pd.Series,
    codes: pd.Series,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for indices in _group_indices(codes):
        left_group = left.loc[indices]
        right_group = right.loc[indices]
        left_mean = left_group.rolling(window, min_periods=window).mean()
        right_mean = right_group.rolling(window, min_periods=window).mean()
        product_mean = (left_group * right_group).rolling(
            window, min_periods=window
        ).mean()
        result.loc[indices] = product_mean - left_mean * right_mean
    return result


def _rolling_variance(
    values: pd.Series,
    codes: pd.Series,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for indices in _group_indices(codes):
        group_values = values.loc[indices]
        result.loc[indices] = group_values.rolling(
            window, min_periods=window
        ).var(ddof=0)
    return result


def _rolling_corr(
    left: pd.Series,
    right: pd.Series,
    codes: pd.Series,
    window: int,
) -> pd.Series:
    covariance = _rolling_covariance(left, right, codes, window)
    left_variance = _rolling_variance(left, codes, window)
    right_variance = _rolling_variance(right, codes, window)
    denominator = left_variance.mul(right_variance).clip(lower=0.0).pow(0.5)
    return covariance.div(denominator.where(denominator.gt(0)))


def _rolling_residual_stats(
    returns: pd.Series,
    market_returns: pd.Series,
    codes: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Fit an intercept/beta within each window and return two residual risks.

    For an endpoint ``t``, the regression uses ``[t-window+1, t]`` only.  The
    resulting residuals are the in-window observations against that endpoint's
    fitted coefficients.  This is the usual rolling idiosyncratic-volatility
    estimate and keeps the residual downside measure on the same regression
    basis as residual volatility.
    """

    residual_vol = pd.Series(np.nan, index=returns.index, dtype=float)
    residual_downside = pd.Series(np.nan, index=returns.index, dtype=float)
    for indices in _group_indices(codes):
        return_values = returns.loc[indices].to_numpy(dtype=float)
        market_values = market_returns.loc[indices].to_numpy(dtype=float)
        vol_values = np.full(len(indices), np.nan, dtype=float)
        downside_values = np.full(len(indices), np.nan, dtype=float)
        for endpoint in range(window - 1, len(indices)):
            start = endpoint - window + 1
            stock_window = return_values[start : endpoint + 1]
            market_window = market_values[start : endpoint + 1]
            if not (
                np.isfinite(stock_window).all()
                and np.isfinite(market_window).all()
            ):
                continue
            market_centered = market_window - np.mean(market_window)
            market_variance = float(np.mean(market_centered * market_centered))
            if not np.isfinite(market_variance) or market_variance <= 1e-18:
                continue
            stock_centered = stock_window - np.mean(stock_window)
            beta = float(np.mean(stock_centered * market_centered) / market_variance)
            alpha = float(np.mean(stock_window) - beta * np.mean(market_window))
            residual = stock_window - alpha - beta * market_window
            residual = residual[np.isfinite(residual)]
            if len(residual) != window:
                continue
            vol_values[endpoint] = float(np.sqrt(np.mean(residual * residual)))
            negative = np.minimum(residual, 0.0)
            downside_values[endpoint] = float(np.sqrt(np.mean(negative * negative)))
        residual_vol.loc[indices] = vol_values
        residual_downside.loc[indices] = downside_values
    return residual_vol, residual_downside


def _rolling_expected_shortfall(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    """Historical lower-tail mean, expressed as a positive loss magnitude."""

    def expected_shortfall(array: np.ndarray) -> float:
        if not np.isfinite(array).all():
            return np.nan
        cutoff = float(np.quantile(array, 0.20))
        tail = array[array <= cutoff]
        # A positive-only window has no downside loss.  Returning zero keeps
        # the statistic defined while still ranking real losses higher.
        losses = tail[tail < 0.0]
        return float(-np.mean(losses)) if len(losses) else 0.0

    return _rolling_apply(values, codes, window, expected_shortfall)


def _rolling_state_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    codes: pd.Series,
    window: int,
    downside: bool,
) -> pd.Series:
    """Estimate beta only on down or up market observations in each window."""

    result = pd.Series(np.nan, index=returns.index, dtype=float)
    required_state_observations = max(3, int(np.ceil(window * 0.20)))
    for indices in _group_indices(codes):
        return_values = returns.loc[indices].to_numpy(dtype=float)
        market_values = market_returns.loc[indices].to_numpy(dtype=float)
        beta_values = np.full(len(indices), np.nan, dtype=float)
        for endpoint in range(window - 1, len(indices)):
            start = endpoint - window + 1
            stock_window = return_values[start : endpoint + 1]
            market_window = market_values[start : endpoint + 1]
            if not (
                np.isfinite(stock_window).all()
                and np.isfinite(market_window).all()
            ):
                continue
            state = market_window <= 0.0 if downside else market_window > 0.0
            if int(state.sum()) < required_state_observations:
                continue
            stock_state = stock_window[state]
            market_state = market_window[state]
            market_centered = market_state - np.mean(market_state)
            market_variance = float(np.mean(market_centered * market_centered))
            if not np.isfinite(market_variance) or market_variance <= 1e-18:
                continue
            stock_centered = stock_state - np.mean(stock_state)
            beta_values[endpoint] = float(
                np.mean(stock_centered * market_centered) / market_variance
            )
        result.loc[indices] = beta_values
    return result


def _rolling_impact_shock_frequency(
    impact: pd.Series,
    codes: pd.Series,
    window: int,
) -> pd.Series:
    """Count impact outliers relative to the current causal window median."""

    def shock_frequency(array: np.ndarray) -> float:
        if not np.isfinite(array).all():
            return np.nan
        baseline = float(np.median(array))
        if not np.isfinite(baseline) or baseline < 0.0:
            return np.nan
        return float(np.mean(array > 2.0 * baseline))

    return _rolling_apply(impact, codes, window, shock_frequency)


def _cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    ranked = numeric.groupby(dates, sort=False, dropna=True).rank(pct=True)
    return ranked.sub(0.5).mul(2.0).clip(-1.0, 1.0)


def add_barra_residual_factors(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Append causal Barra-style residual and impact factors.

    Parameters
    ----------
    frame:
        Daily rows with ``code``, ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume`` and ``amount``.  The OHLCV/amount columns are
        converted numerically inside this function; precomputed return or
        factor columns are ignored so the layer cannot inherit look-ahead
        values from another feature builder.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        A stable ``code``/``date`` sorted copy and the 15 new feature names.
        Each feature is a date-level cross-sectional rank in ``[-1, 1]``;
        unavailable windows remain ``NaN``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("因子输入必须是 pandas.DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Barra 残差因子缺少输入列: {', '.join(missing)}")

    # Normalize only the working copy so duplicate caller index labels cannot
    # make label-based reindexing select a row more than once.
    source = frame.copy(deep=True).reset_index(drop=True)
    date_key = pd.to_datetime(source["date"], errors="coerce")
    order_frame = pd.DataFrame(
        {
            "_code": source["code"],
            "_date_key": date_key,
            "_input_order": np.arange(len(source), dtype=np.int64),
        },
        index=source.index,
    )
    order = order_frame.sort_values(
        ["_code", "_date_key", "_input_order"], kind="mergesort"
    ).index.to_numpy()
    output = source.iloc[order].reset_index(drop=True)
    date_key = date_key.iloc[order].reset_index(drop=True)
    codes = output["code"]

    numeric = {
        column: pd.to_numeric(output[column], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount")
    }
    close = numeric["close"].where(numeric["close"].gt(0.0))
    previous_close = close.groupby(codes, sort=False, dropna=False).shift(1)
    previous_close = previous_close.where(previous_close.gt(0.0))
    returns = close.div(previous_close).sub(1.0)

    # This is explicitly an equal-weight same-date market return.  Because
    # returns use each security's previous close, it contains no future date;
    # using it here means the factor is a post-close signal for next open.
    market_returns = returns.groupby(date_key, sort=False, dropna=True).transform("mean")

    market_corr = _rolling_corr(returns, market_returns, codes, 60)
    raw: dict[str, pd.Series] = {"market_corr_60": market_corr}

    for window in WINDOWS:
        residual_vol, residual_downside = _rolling_residual_stats(
            returns, market_returns, codes, window
        )
        raw[f"residual_vol_{window}"] = residual_vol
        raw[f"residual_downside_vol_{window}"] = residual_downside

        downside = returns.clip(upper=0.0)
        raw[f"downside_deviation_{window}"] = _rolling_mean(
            downside.pow(2), codes, window
        ).pow(0.5)
        raw[f"expected_shortfall_{window}"] = _rolling_expected_shortfall(
            returns, codes, window
        )

        amount = numeric["amount"].where(numeric["amount"].gt(0.0))
        # Scale dollar flow only to keep rolling covariance numerically well
        # conditioned; it does not change the economic ordering of lambda.
        signed_flow = np.sign(returns).mul(amount).div(1_000_000.0)
        signed_flow_variance = _rolling_variance(signed_flow, codes, window)
        raw[f"kyle_lambda_{window}"] = _rolling_covariance(
            returns, signed_flow, codes, window
        ).div(
            signed_flow_variance.where(signed_flow_variance.gt(1e-18))
        )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            impact = returns.abs().mul(
                (1_000_000.0 / amount).pow(0.5)
            )
        raw[f"impact_shock_frequency_{window}"] = (
            _rolling_impact_shock_frequency(impact, codes, window)
        )

    raw["market_down_beta_60"] = _rolling_state_beta(
        returns, market_returns, codes, 60, downside=True
    )
    raw["market_up_beta_60"] = _rolling_state_beta(
        returns, market_returns, codes, 60, downside=False
    )

    expected_raw_names = [name.removeprefix("x_") for name in SPECS]
    if set(raw) != set(expected_raw_names) or len(raw) != len(expected_raw_names):
        raise RuntimeError("Barra 残差因子计算结果与元数据不一致")

    feature_columns: list[str] = []
    for raw_name in expected_raw_names:
        feature_name = f"x_{raw_name}"
        output[feature_name] = _cross_section_rank(raw[raw_name], date_key)
        feature_columns.append(feature_name)
    return output, feature_columns


# Generic alias for dynamic batch loaders while preserving the explicit name.
add = add_barra_residual_factors
