from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


RETURN_DECOMPOSITION_FACTOR_SPECS: dict[str, dict[str, str]] = {
    **{
        f"x_idio_momentum_{window}": {
            "family": "residual_momentum",
            "origin": "market_neutral_return_decomposition",
            "formula": f"sum(ret_i - equal_weight_market_ret, {window})",
        }
        for window in (5, 20, 60)
    },
    **{
        f"x_idio_trend_tstat_{window}": {
            "family": "residual_momentum",
            "origin": "market_neutral_return_decomposition",
            "formula": f"mean(idio_ret, {window}) / std(idio_ret, {window})",
        }
        for window in (20, 60)
    },
    **{
        f"x_market_beta_{window}": {
            "family": "market_exposure",
            "origin": "rolling_market_model",
            "formula": f"cov(ret_i, market_ret, {window}) / var(market_ret, {window})",
        }
        for window in (20, 60)
    },
    "x_market_beta_change_20_60": {
        "family": "market_exposure",
        "origin": "rolling_market_model",
        "formula": "beta_20 - beta_60",
    },
    **{
        f"x_intraday_momentum_{window}": {
            "family": "return_decomposition",
            "origin": "overnight_intraday_decomposition",
            "formula": f"sum(log(close/open), {window})",
        }
        for window in (5, 20, 60)
    },
    **{
        f"x_overnight_momentum_{window}": {
            "family": "return_decomposition",
            "origin": "overnight_intraday_decomposition",
            "formula": f"sum(log(open/previous_close), {window})",
        }
        for window in (5, 20, 60)
    },
    **{
        f"x_intraday_overnight_spread_{window}": {
            "family": "return_decomposition",
            "origin": "overnight_intraday_decomposition",
            "formula": f"intraday_momentum_{window} - overnight_momentum_{window}",
        }
        for window in (20, 60)
    },
    **{
        f"x_high_volume_return_{window}": {
            "family": "volume_conditioned_return",
            "origin": "volume_conditioned_momentum",
            "formula": f"mean(ret_i where volume > rolling_median_20, {window})",
        }
        for window in (20, 60)
    },
    **{
        f"x_low_volume_return_{window}": {
            "family": "volume_conditioned_return",
            "origin": "volume_conditioned_momentum",
            "formula": f"mean(ret_i where volume <= rolling_median_20, {window})",
        }
        for window in (20, 60)
    },
    **{
        f"x_return_autocorr_{window}": {
            "family": "return_persistence",
            "origin": "serial_dependence",
            "formula": f"corr(ret_i, lag(ret_i, 1), {window})",
        }
        for window in (20, 60)
    },
    **{
        f"x_range_adjusted_momentum_{window}": {
            "family": "risk_adjusted_momentum",
            "origin": "range_normalized_return",
            "formula": f"sum(ret_i, {window}) / mean((high-low)/previous_close, {window})",
        }
        for window in (20, 60)
    },
}

SPECS = RETURN_DECOMPOSITION_FACTOR_SPECS
_REQUIRED = {"code", "date", "open", "high", "low", "close", "volume"}


def return_decomposition_factor_metadata(feature: str) -> dict[str, str] | None:
    metadata = SPECS.get(feature)
    return dict(metadata) if metadata else None


def _transform(
    values: pd.Series,
    codes: pd.Series,
    window: int,
    operation: str | Callable,
) -> pd.Series:
    return values.groupby(codes, sort=False, dropna=True).transform(
        lambda group: group.rolling(window, min_periods=window).agg(operation)
    )


def _rolling_corr(
    left: pd.Series,
    right: pd.Series,
    codes: pd.Series,
    window: int,
) -> pd.Series:
    output = pd.Series(np.nan, index=left.index, dtype=float)
    for indices in left.groupby(codes, sort=False, dropna=True).groups.values():
        output.loc[indices] = left.loc[indices].rolling(
            window, min_periods=window
        ).corr(right.loc[indices])
    return output


def _rank_by_date(values: pd.Series, dates: pd.Series) -> pd.Series:
    ranked = values.groupby(dates, sort=False, dropna=True).rank(pct=True)
    return ranked.mul(2.0).sub(1.0)


def add_return_decomposition_factors(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(
            "return decomposition factors missing columns: " + ", ".join(missing)
        )
    source = frame.copy(deep=True).reset_index(drop=True)
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["_input_order"] = np.arange(len(source), dtype=np.int64)
    source = source.sort_values(["code", "date", "_input_order"], kind="mergesort")
    codes, dates = source["code"], source["date"]
    close = pd.to_numeric(source["close"], errors="coerce")
    open_price = pd.to_numeric(source["open"], errors="coerce")
    high = pd.to_numeric(source["high"], errors="coerce")
    low = pd.to_numeric(source["low"], errors="coerce")
    volume = pd.to_numeric(source["volume"], errors="coerce")
    previous_close = close.groupby(codes, sort=False).shift(1)
    returns = close.div(previous_close).sub(1.0)
    market_return = returns.groupby(dates, sort=False).transform("mean")
    idio_return = returns.sub(market_return)
    intraday = np.log(close.where(close > 0).div(open_price.where(open_price > 0)))
    overnight = np.log(
        open_price.where(open_price > 0).div(previous_close.where(previous_close > 0))
    )
    range_ratio = high.sub(low).div(previous_close.where(previous_close > 0))
    volume_median = _transform(volume, codes, 20, "median")
    high_volume = volume.gt(volume_median)

    raw: dict[str, pd.Series] = {}
    for window in (5, 20, 60):
        raw[f"x_idio_momentum_{window}"] = _transform(
            idio_return, codes, window, "sum"
        )
        raw[f"x_intraday_momentum_{window}"] = _transform(
            intraday, codes, window, "sum"
        )
        raw[f"x_overnight_momentum_{window}"] = _transform(
            overnight, codes, window, "sum"
        )
    betas = {}
    for window in (20, 60):
        idio_mean = _transform(idio_return, codes, window, "mean")
        idio_std = _transform(idio_return, codes, window, "std")
        raw[f"x_idio_trend_tstat_{window}"] = idio_mean.div(
            idio_std.replace(0.0, np.nan)
        )
        ret_mean = _transform(returns, codes, window, "mean")
        market_mean = _transform(market_return, codes, window, "mean")
        covariance = _transform(
            returns.mul(market_return), codes, window, "mean"
        ).sub(ret_mean.mul(market_mean))
        market_variance = _transform(
            market_return.mul(market_return), codes, window, "mean"
        ).sub(market_mean.pow(2))
        betas[window] = covariance.div(market_variance.replace(0.0, np.nan))
        raw[f"x_market_beta_{window}"] = betas[window]
        raw[f"x_intraday_overnight_spread_{window}"] = raw[
            f"x_intraday_momentum_{window}"
        ].sub(raw[f"x_overnight_momentum_{window}"])
        high_count = _transform(high_volume.astype(float), codes, window, "sum")
        low_count = _transform((~high_volume).astype(float), codes, window, "sum")
        raw[f"x_high_volume_return_{window}"] = _transform(
            returns.where(high_volume, 0.0), codes, window, "sum"
        ).div(high_count.replace(0.0, np.nan))
        raw[f"x_low_volume_return_{window}"] = _transform(
            returns.where(~high_volume, 0.0), codes, window, "sum"
        ).div(low_count.replace(0.0, np.nan))
        raw[f"x_return_autocorr_{window}"] = _rolling_corr(
            returns, returns.groupby(codes, sort=False).shift(1), codes, window
        )
        raw[f"x_range_adjusted_momentum_{window}"] = _transform(
            returns, codes, window, "sum"
        ).div(_transform(range_ratio, codes, window, "mean").replace(0.0, np.nan))
    raw["x_market_beta_change_20_60"] = betas[20].sub(betas[60])

    for feature in SPECS:
        source[feature] = _rank_by_date(
            raw[feature].replace([np.inf, -np.inf], np.nan), dates
        )
    source = source.sort_values("_input_order", kind="mergesort").drop(
        columns="_input_order"
    )
    source.index = frame.index
    return source, list(SPECS)
