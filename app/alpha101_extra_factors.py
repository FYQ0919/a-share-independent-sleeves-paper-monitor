"""Causal WorldQuant Alpha101-style factors from daily OHLCV data.

This module is an independent research layer.  It deliberately computes all
intermediate expressions from ``code``, ``date``, ``open``, ``high``, ``low``,
``close``, ``volume`` and ``amount`` so it can be evaluated without depending
on another feature builder.  Rolling expressions are evaluated in
``code/date`` order with complete windows only.  The returned features are
cross-sectional percentile ranks on each date, mapped to ``[-1, 1]``.

The expressions are faithful, low-complexity adaptations of the named
WorldQuant Alpha101 formulas.  In particular, ``adv20`` means the causal
20-day mean of volume and ``vwap`` is the daily amount/volume proxy.  A final
cross-sectional rank makes the explicit outer ``rank`` in some source
expressions equivalent while keeping one stable output contract for this
project.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# The dictionary order is also the feature output order and is kept stable for
# model schemas and factor-batch comparisons.
ALPHA101_EXTRA_FACTOR_SPECS: dict[str, dict[str, str]] = {
    "x_wq_alpha003": {
        "family": "price_volume_correlation",
        "origin": "worldquant_alpha101_003",
        "formula": "-correlation(rank(open), rank(volume), 10)",
    },
    "x_wq_alpha004": {
        "family": "price_trend",
        "origin": "worldquant_alpha101_004",
        "formula": "-ts_rank(rank(low), 9)",
    },
    "x_wq_alpha006": {
        "family": "price_volume_correlation",
        "origin": "worldquant_alpha101_006",
        "formula": "-correlation(open, volume, 10)",
    },
    "x_wq_alpha007": {
        "family": "conditional_momentum",
        "origin": "worldquant_alpha101_007",
        "formula": (
            "if(volume > adv20, -ts_rank(abs(delta(close, 7)), 60) "
            "* sign(delta(close, 7)), -1); adv20=mean(volume, 20)"
        ),
    },
    "x_wq_alpha012": {
        "family": "price_volume_reversal",
        "origin": "worldquant_alpha101_012",
        "formula": "-sign(delta(volume, 1)) * delta(close, 1)",
    },
    "x_wq_alpha016": {
        "family": "price_volume_covariance",
        "origin": "worldquant_alpha101_016",
        "formula": "-covariance(rank(high), rank(volume), 5)",
    },
    "x_wq_alpha018": {
        "family": "intraday_reversal",
        "origin": "worldquant_alpha101_018",
        "formula": (
            "-rank(stddev(abs(close-open), 5) + (close-open) "
            "+ correlation(close, open, 10))"
        ),
    },
    "x_wq_alpha020": {
        "family": "overnight_gap",
        "origin": "worldquant_alpha101_020",
        "formula": (
            "-rank(open-delay(high, 1)) * rank(open-delay(close, 1)) "
            "* rank(open-delay(low, 1))"
        ),
    },
    "x_wq_alpha021": {
        "family": "price_regime",
        "origin": "worldquant_alpha101_021",
        "formula": (
            "if(mean(close, 8)+stddev(close, 8)<mean(close, 2), -1, "
            "if(mean(close, 2)<mean(close, 8)-stddev(close, 8), 1, "
            "if(volume/adv20>=1, 1, -1)))"
        ),
    },
    "x_wq_alpha022": {
        "family": "price_volume_correlation_change",
        "origin": "worldquant_alpha101_022",
        "formula": "-delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))",
    },
    "x_wq_alpha023": {
        "family": "breakout_reversal",
        "origin": "worldquant_alpha101_023",
        "formula": "if(mean(high, 20)<high, -delta(high, 2), 0)",
    },
    "x_wq_alpha025": {
        "family": "amount_weighted_reversal",
        "origin": "worldquant_alpha101_025",
        "formula": "rank(-returns * adv20 * vwap * (high-close)); vwap=amount/volume",
    },
}

# ``SPECS`` is intentionally exported as the short interface name requested by
# the factor framework.  The descriptive aliases make discovery convenient
# without creating a second mutable registry.
SPECS = ALPHA101_EXTRA_FACTOR_SPECS
ALPHA101_FACTOR_SPECS = ALPHA101_EXTRA_FACTOR_SPECS

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


def alpha101_extra_factor_metadata(feature: str) -> dict[str, str] | None:
    """Return a copy of one registered Alpha101 factor's metadata."""

    metadata = SPECS.get(feature)
    return dict(metadata) if metadata else None


# Alias mirrors the short names used by some factor registries.
alpha101_factor_metadata = alpha101_extra_factor_metadata


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rolling_mean(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    return values.groupby(codes, sort=False, dropna=True).transform(
        lambda group: group.rolling(window, min_periods=window).mean()
    )


def _rolling_std(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    return values.groupby(codes, sort=False, dropna=True).transform(
        lambda group: group.rolling(window, min_periods=window).std(ddof=0)
    )


def _group_indices(codes: pd.Series) -> list[pd.Index]:
    return list(codes.groupby(codes, sort=False, dropna=True).groups.values())


def _rolling_corr(
    left: pd.Series,
    right: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for indices in groups:
        left_group = left.loc[indices]
        right_group = right.loc[indices]
        correlation = left_group.rolling(window, min_periods=window).corr(right_group)
        valid_window = (
            left_group.notna().rolling(window, min_periods=window).sum().eq(window)
            & right_group.notna().rolling(window, min_periods=window).sum().eq(window)
        )
        result.loc[indices] = correlation.where(valid_window)
    return result


def _rolling_cov(
    left: pd.Series,
    right: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for indices in groups:
        left_group = left.loc[indices]
        right_group = right.loc[indices]
        covariance = left_group.rolling(window, min_periods=window).cov(right_group)
        valid_window = (
            left_group.notna().rolling(window, min_periods=window).sum().eq(window)
            & right_group.notna().rolling(window, min_periods=window).sum().eq(window)
        )
        result.loc[indices] = covariance.where(valid_window)
    return result


def _last_rank(values: np.ndarray) -> float:
    """Percentile rank of the endpoint, matching pandas' average tie rule."""

    if values.size == 0 or not np.isfinite(values).all():
        return np.nan
    endpoint = values[-1]
    less = float(np.count_nonzero(values < endpoint))
    equal = float(np.count_nonzero(values == endpoint))
    return (less + (equal + 1.0) / 2.0) / float(values.size)


def _rolling_last_rank(
    values: pd.Series,
    codes: pd.Series,
    window: int,
) -> pd.Series:
    return values.groupby(codes, sort=False, dropna=True).transform(
        lambda group: group.rolling(window, min_periods=window).apply(
            _last_rank, raw=True
        )
    )


def _cross_section_percentile(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    return numeric.groupby(dates, sort=False).rank(pct=True)


def _cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    percentile = _cross_section_percentile(values, dates)
    return percentile.sub(0.5).mul(2.0).clip(-1.0, 1.0)


def _delta(values: pd.Series, codes: pd.Series, periods: int) -> pd.Series:
    return values.sub(values.groupby(codes, sort=False, dropna=True).shift(periods))


def add_alpha101_extra_factors(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append causal Alpha101-style factors and date-wise ranks.

    ``frame`` is copied deeply and sorted by ``code`` and parsed ``date``.
    Every rolling expression requires a complete window, so insufficient
    history or invalid input remains ``NaN`` rather than being imputed.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("factor input must be a pandas.DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "Alpha101 extra factors missing input columns: " + ", ".join(missing)
        )

    output = frame.copy(deep=True)
    date_key = pd.to_datetime(output["date"], errors="coerce")
    order = pd.DataFrame(
        {"_code": output["code"], "_date_key": date_key}, index=output.index
    ).sort_values(["_code", "_date_key"], kind="mergesort").index
    output = output.loc[order].reset_index(drop=True)
    date_key = date_key.loc[order].reset_index(drop=True)

    codes = output["code"]
    numeric = {
        column: _numeric(output[column])
        for column in ("open", "high", "low", "close", "volume", "amount")
    }
    open_price = numeric["open"].where(numeric["open"].gt(0))
    high_price = numeric["high"].where(numeric["high"].gt(0))
    low_price = numeric["low"].where(numeric["low"].gt(0))
    close_price = numeric["close"].where(numeric["close"].gt(0))
    volume = numeric["volume"].where(numeric["volume"].gt(0))
    amount = numeric["amount"].where(numeric["amount"].gt(0))
    groups = _group_indices(codes)

    open_rank = _cross_section_percentile(open_price, date_key)
    high_rank = _cross_section_percentile(high_price, date_key)
    low_rank = _cross_section_percentile(low_price, date_key)
    volume_rank = _cross_section_percentile(volume, date_key)

    previous_close = close_price.groupby(codes, sort=False, dropna=True).shift(1)
    returns = close_price.div(previous_close.where(previous_close.gt(0))).sub(1.0)
    delta_close_1 = _delta(close_price, codes, 1)
    delta_volume_1 = _delta(volume, codes, 1)
    adv20 = _rolling_mean(volume, codes, 20)

    raw: dict[str, pd.Series] = {}

    # Alpha 003: cross-sectional ranks are causal because each date is ranked
    # independently before the per-security time-series correlation.
    raw["wq_alpha003"] = _rolling_corr(open_rank, volume_rank, groups, 10).mul(-1.0)

    raw["wq_alpha004"] = _rolling_last_rank(low_rank, codes, 9).mul(-1.0)

    raw["wq_alpha006"] = _rolling_corr(open_price, volume, groups, 10).mul(-1.0)

    delta_close_7 = _delta(close_price, codes, 7)
    abs_delta_rank_60 = _rolling_last_rank(delta_close_7.abs(), codes, 60)
    alpha007_momentum = abs_delta_rank_60.mul(-delta_close_7.apply(np.sign))
    alpha007_valid = (
        volume.notna()
        & adv20.notna()
        & delta_close_7.notna()
        & abs_delta_rank_60.notna()
    )
    raw["wq_alpha007"] = alpha007_momentum.where(volume.gt(adv20), -1.0).where(
        alpha007_valid
    )

    raw["wq_alpha012"] = delta_volume_1.apply(np.sign).mul(delta_close_1).mul(-1.0)

    raw["wq_alpha016"] = _rolling_cov(high_rank, volume_rank, groups, 5).mul(-1.0)

    body = close_price.sub(open_price)
    body_std_5 = _rolling_std(body.abs(), codes, 5)
    close_open_corr_10 = _rolling_corr(close_price, open_price, groups, 10)
    raw["wq_alpha018"] = body_std_5.add(body).add(close_open_corr_10).mul(-1.0)

    previous_high = high_price.groupby(codes, sort=False, dropna=True).shift(1)
    previous_low = low_price.groupby(codes, sort=False, dropna=True).shift(1)
    high_gap_rank = _cross_section_percentile(open_price.sub(previous_high), date_key)
    close_gap_rank = _cross_section_percentile(
        open_price.sub(previous_close), date_key
    )
    low_gap_rank = _cross_section_percentile(open_price.sub(previous_low), date_key)
    raw["wq_alpha020"] = high_gap_rank.mul(close_gap_rank).mul(low_gap_rank).mul(-1.0)

    close_mean_8 = _rolling_mean(close_price, codes, 8)
    close_std_8 = _rolling_std(close_price, codes, 8)
    close_mean_2 = _rolling_mean(close_price, codes, 2)
    volume_ratio_20 = volume.div(adv20.where(adv20.gt(0)))
    regime_valid = (
        close_mean_8.notna()
        & close_std_8.notna()
        & close_mean_2.notna()
        & volume_ratio_20.notna()
    )
    alpha021_values = np.where(
        close_mean_8.add(close_std_8).lt(close_mean_2),
        -1.0,
        np.where(
            close_mean_2.lt(close_mean_8.sub(close_std_8)),
            1.0,
            np.where(volume_ratio_20.ge(1.0), 1.0, -1.0),
        ),
    )
    raw["wq_alpha021"] = pd.Series(
        alpha021_values, index=output.index, dtype=float
    ).where(regime_valid)

    high_volume_corr_5 = _rolling_corr(high_price, volume, groups, 5)
    corr_change_5 = _delta(high_volume_corr_5, codes, 5)
    close_std_20 = _rolling_std(close_price, codes, 20)
    raw["wq_alpha022"] = corr_change_5.mul(
        _cross_section_percentile(close_std_20, date_key)
    ).mul(-1.0)

    high_mean_20 = _rolling_mean(high_price, codes, 20)
    delta_high_2 = _delta(high_price, codes, 2)
    alpha023_valid = high_mean_20.notna() & delta_high_2.notna()
    alpha023_values = pd.Series(
        np.where(high_mean_20.lt(high_price), -delta_high_2, 0.0),
        index=output.index,
        dtype=float,
    )
    raw["wq_alpha023"] = alpha023_values.where(alpha023_valid)

    vwap = amount.div(volume.where(volume.gt(0)))
    raw["wq_alpha025"] = returns.mul(adv20).mul(vwap).mul(
        high_price.sub(close_price)
    ).mul(-1.0)

    expected_raw_names = [name.removeprefix("x_") for name in SPECS]
    if list(raw) != expected_raw_names:
        raise RuntimeError("Alpha101 factor calculation order does not match SPECS")

    feature_columns: list[str] = []
    for raw_name, values in raw.items():
        feature_name = f"x_{raw_name}"
        output[feature_name] = _cross_section_rank(values, date_key)
        feature_columns.append(feature_name)
    return output, feature_columns


# Short alias for callers that use the batch name without ``extra``.
add_alpha101_factors = add_alpha101_extra_factors
