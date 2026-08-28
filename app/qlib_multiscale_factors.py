"""Causal, multi-scale Qlib-style daily price/volume factors.

This layer is intentionally independent from the frozen Alpha158-lite and
the existing research-factor batches.  It fills several useful Qlib
Alpha158/Alpha360 families that are not present there: rolling regression
trend (BETA/RSQR/RESI), a short-versus-long volatility regime, rolling VWAP
dislocation, money-flow imbalance, and volume/price trend divergence.

Only daily ``code/date/open/high/low/close/volume/amount`` observations are
required.  Every statistic is computed after sorting by instrument and date,
uses the current row plus preceding rows for that instrument, and is ranked
cross-sectionally by date into the project's ``[-1, 1]`` score convention.
The first incomplete windows intentionally remain ``NaN``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


WINDOWS = (20, 60)


# ``origin`` and ``formula`` are kept human-readable so the layer can be
# audited without importing Qlib or relying on an expression evaluator.
QLIB_MULTISCALE_FACTOR_SPECS: dict[str, dict[str, str]] = {
    "x_beta_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_beta",
        "formula": "OLS_slope(log(close), time, 20)",
    },
    "x_beta_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_beta",
        "formula": "OLS_slope(log(close), time, 60)",
    },
    "x_rsqr_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_rsqr",
        "formula": "R^2(log(close) ~ time, 20)",
    },
    "x_rsqr_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_rsqr",
        "formula": "R^2(log(close) ~ time, 60)",
    },
    "x_resi_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_resi",
        "formula": "log(close_t) - fitted_log(close_t; OLS_20)",
    },
    "x_resi_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_resi",
        "formula": "log(close_t) - fitted_log(close_t; OLS_60)",
    },
    "x_volatility_regime_20": {
        "family": "risk_tail",
        "origin": "multiscale_realized_volatility",
        "formula": "std(ret_1, 5) / std(ret_1, 20)",
    },
    "x_volatility_regime_60": {
        "family": "risk_tail",
        "origin": "multiscale_realized_volatility",
        "formula": "std(ret_1, 15) / std(ret_1, 60)",
    },
    "x_vwap_gap_20": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha360_vwap_style",
        "formula": "close / (sum(amount, 20) / sum(volume, 20)) - 1",
    },
    "x_vwap_gap_60": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha360_vwap_style",
        "formula": "close / (sum(amount, 60) / sum(volume, 60)) - 1",
    },
    "x_money_flow_imbalance_20": {
        "family": "microstructure_proxy",
        "origin": "clv_money_flow_imbalance",
        "formula": "sum(CLV * amount, 20) / sum(amount, 20); CLV=(2*close-high-low)/(high-low)",
    },
    "x_money_flow_imbalance_60": {
        "family": "microstructure_proxy",
        "origin": "clv_money_flow_imbalance",
        "formula": "sum(CLV * amount, 60) / sum(amount, 60); CLV=(2*close-high-low)/(high-low)",
    },
    "x_volume_price_slope_gap_20": {
        "family": "microstructure_proxy",
        "origin": "multiscale_volume_price_divergence",
        "formula": "OLS_slope(log(volume), time, 20) - OLS_slope(log(close), time, 20)",
    },
    "x_volume_price_slope_gap_60": {
        "family": "microstructure_proxy",
        "origin": "multiscale_volume_price_divergence",
        "formula": "OLS_slope(log(volume), time, 60) - OLS_slope(log(close), time, 60)",
    },
}


SPECS = QLIB_MULTISCALE_FACTOR_SPECS

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


def qlib_multiscale_factor_metadata(feature: str) -> dict[str, str] | None:
    """Return a defensive copy of one factor's audit metadata."""

    metadata_value = QLIB_MULTISCALE_FACTOR_SPECS.get(feature)
    return dict(metadata_value) if metadata_value else None


def _group_indices(frame: pd.DataFrame) -> list[pd.Index]:
    """Return instrument row indexes in the already sorted output frame."""

    return list(frame.groupby("code", sort=False, dropna=True).groups.values())


def _rolling_sum_by_code(
    values: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for indices in groups:
        result.loc[indices] = values.loc[indices].rolling(
            window, min_periods=window
        ).sum()
    return result


def _rolling_std_by_code(
    values: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for indices in groups:
        result.loc[indices] = values.loc[indices].rolling(
            window, min_periods=window
        ).std(ddof=0)
    return result


def _rolling_regression_by_code(
    values: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute causal rolling OLS slope, R-squared, and endpoint residual.

    A global row-position variable is sufficient for each window because
    shifting the time origin does not change slope or fitted residuals.  The
    rolling-sum form avoids a Python-level loop for every individual window.
    Missing/non-finite observations make the corresponding complete window
    invalid through pandas' ``min_periods=window`` behavior.
    """

    beta = pd.Series(np.nan, index=values.index, dtype=float)
    rsqr = pd.Series(np.nan, index=values.index, dtype=float)
    residual = pd.Series(np.nan, index=values.index, dtype=float)

    for indices in groups:
        y = values.loc[indices].astype(float)
        t = pd.Series(np.arange(len(y), dtype=float), index=indices)
        sum_y = y.rolling(window, min_periods=window).sum()
        sum_y2 = y.pow(2).rolling(window, min_periods=window).sum()
        sum_t = t.rolling(window, min_periods=window).sum()
        sum_t2 = t.pow(2).rolling(window, min_periods=window).sum()
        sum_ty = (t * y).rolling(window, min_periods=window).sum()

        denominator_t = float(window) * sum_t2.sub(sum_t.pow(2).div(window))
        covariance_numerator = float(window) * sum_ty.sub(
            sum_t.mul(sum_y).div(window)
        )
        slope = covariance_numerator.div(denominator_t)
        mean_t = sum_t.div(window)
        mean_y = sum_y.div(window)
        fitted_endpoint = mean_y.add(slope.mul(t.sub(mean_t)))
        sum_squared_total = sum_y2.sub(sum_y.pow(2).div(window))

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r_squared = covariance_numerator.pow(2).div(
                denominator_t.mul(sum_squared_total)
            )

        # A flat log-price path has no explanatory variation.  Its slope and
        # endpoint residual are well-defined, while R^2 is conventionally 0.
        flat_path = sum_squared_total.abs().le(1e-14)
        slope = slope.where(denominator_t.gt(0))
        valid_window = denominator_t.gt(0) & sum_squared_total.notna()
        r_squared = r_squared.where(valid_window)
        r_squared = r_squared.where(
            ~valid_window | sum_squared_total.gt(1e-14), 0.0
        )
        r_squared = r_squared.clip(lower=0.0, upper=1.0)
        fitted_endpoint = fitted_endpoint.where(denominator_t.gt(0))
        endpoint_residual = y.sub(fitted_endpoint)
        endpoint_residual = endpoint_residual.where(~flat_path | y.isna(), 0.0)

        beta.loc[indices] = slope
        rsqr.loc[indices] = r_squared
        residual.loc[indices] = endpoint_residual

    return beta, rsqr, residual


def _cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    ranked = numeric.groupby(dates, sort=False, dropna=True).rank(pct=True)
    return ranked.sub(0.5).mul(2.0).clip(-1.0, 1.0)


def _sorted_copy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sort without assuming a unique input index or mutating ``frame``."""

    output = frame.copy(deep=True)
    date_key = pd.to_datetime(output["date"], errors="coerce")
    order_frame = pd.DataFrame(
        {
            "_code_key": output["code"].astype("string").fillna("").to_numpy(),
            "_date_key": date_key.to_numpy(),
            "_row_key": np.arange(len(output), dtype=np.int64),
        }
    )
    order = order_frame.sort_values(
        ["_code_key", "_date_key", "_row_key"], kind="mergesort"
    ).index.to_numpy()
    output = output.iloc[order].reset_index(drop=True)
    return output, date_key.iloc[order].reset_index(drop=True)


def add_qlib_multiscale_factors(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Append 14 causal multi-scale factors and return their feature names.

    The raw statistics are never taken from precomputed factor columns; all
    returns, logs, rolling VWAPs, and regression terms are rebuilt from the
    supplied daily OHLCV/amount fields.  This keeps the layer composable and
    prevents a future-looking upstream column from leaking into the result.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("多尺度因子输入必须是 pandas.DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Qlib 多尺度因子缺少输入列: {', '.join(missing)}")

    output, date_key = _sorted_copy(frame)
    codes = output["code"]
    groups = _group_indices(output)

    numeric = {
        column: pd.to_numeric(output[column], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount")
    }
    close = numeric["close"].where(numeric["close"].gt(0))
    high = numeric["high"].where(numeric["high"].gt(0))
    low = numeric["low"].where(numeric["low"].gt(0))
    volume = numeric["volume"].where(numeric["volume"].gt(0))
    amount = numeric["amount"].where(numeric["amount"].gt(0))

    previous_close = close.groupby(codes, sort=False).shift(1)
    returns = close.div(previous_close.where(previous_close.gt(0))).sub(1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_close = np.log(close)
        log_volume = np.log(volume)

    raw: dict[str, pd.Series] = {}
    regression: dict[int, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for window in WINDOWS:
        regression[window] = _rolling_regression_by_code(
            log_close, groups, window
        )

    for window in WINDOWS:
        beta, _, _ = regression[window]
        raw[f"beta_{window}"] = beta

    for window in WINDOWS:
        _, rsqr, _ = regression[window]
        raw[f"rsqr_{window}"] = rsqr

    for window in WINDOWS:
        _, _, residual = regression[window]
        raw[f"resi_{window}"] = residual

    volatility_regime: dict[int, pd.Series] = {}
    vwap_gap: dict[int, pd.Series] = {}
    money_flow_imbalance: dict[int, pd.Series] = {}
    volume_price_slope_gap: dict[int, pd.Series] = {}
    for window in WINDOWS:
        short_window = max(5, window // 4)
        short_volatility = _rolling_std_by_code(returns, groups, short_window)
        long_volatility = _rolling_std_by_code(returns, groups, window)
        volatility_regime[window] = short_volatility.div(
            long_volatility.where(long_volatility.gt(0))
        )

        amount_sum = _rolling_sum_by_code(amount, groups, window)
        volume_sum = _rolling_sum_by_code(volume, groups, window)
        rolling_vwap = amount_sum.div(volume_sum.where(volume_sum.gt(0)))
        vwap_gap[window] = close.div(rolling_vwap).sub(1.0)

        daily_range = high.sub(low)
        clv = close.mul(2.0).sub(high).sub(low).div(
            daily_range.where(daily_range.gt(0))
        )
        flow = clv.mul(amount)
        money_flow_imbalance[window] = _rolling_sum_by_code(
            flow, groups, window
        ).div(amount_sum.where(amount_sum.gt(0)))

        volume_beta, _, _ = _rolling_regression_by_code(
            log_volume, groups, window
        )
        close_beta, _, _ = regression[window]
        volume_price_slope_gap[window] = volume_beta.sub(
            close_beta
        )

    for window in WINDOWS:
        raw[f"volatility_regime_{window}"] = volatility_regime[window]
    for window in WINDOWS:
        raw[f"vwap_gap_{window}"] = vwap_gap[window]
    for window in WINDOWS:
        raw[f"money_flow_imbalance_{window}"] = money_flow_imbalance[window]
    for window in WINDOWS:
        raw[f"volume_price_slope_gap_{window}"] = volume_price_slope_gap[window]

    expected_raw_names = [name.removeprefix("x_") for name in SPECS]
    if list(raw) != expected_raw_names:
        raise RuntimeError("Qlib 多尺度因子计算顺序与元数据不一致")

    feature_columns: list[str] = []
    for raw_name, values in raw.items():
        feature_name = f"x_{raw_name}"
        output[feature_name] = _cross_section_rank(values, date_key)
        feature_columns.append(feature_name)
    return output, feature_columns


# Short aliases are useful to small research scripts while the explicit names
# above remain consistent with the other app factor modules.
metadata = qlib_multiscale_factor_metadata
add = add_qlib_multiscale_factors


__all__ = [
    "WINDOWS",
    "SPECS",
    "QLIB_MULTISCALE_FACTOR_SPECS",
    "qlib_multiscale_factor_metadata",
    "add_qlib_multiscale_factors",
    "metadata",
    "add",
]
