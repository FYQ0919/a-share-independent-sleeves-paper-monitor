"""Small, auditable collection of Qlib Alpha158-style extra factors.

The project already has an Alpha158-lite layer and a separate framework layer.
This module deliberately stays independent from both of them: it only needs
daily OHLCV rows and computes every intermediate series inside the module.
Values are calculated per instrument in ``code/date`` order and then ranked
cross-sectionally on each signal date into the project's ``[-1, 1]`` scale.

The formulas follow the names used by Qlib's Alpha158 rolling feature family
where practical (IMAX/IMIN/IMXD, CNTN, SUMP/SUMN/SUMD, CORR/CORD).  The two
interval path factors are small extensions because Alpha158's point-in-window
statistics do not directly expose recovery from a drawdown trough.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


# Keep the metadata shape compatible with app.framework_factors: ``origin`` is
# the auditable source/reference, while ``formula`` is the human-readable
# causal expression used for review and factor registry entries.
QLIB_EXTRA_FACTOR_SPECS: dict[str, dict[str, str]] = {
    "x_imax_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "IdxMax(high, 20) / 20",
    },
    "x_imin_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "IdxMin(low, 20) / 20",
    },
    "x_imxd_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "(IdxMax(high, 20) - IdxMin(low, 20)) / 20",
    },
    "x_imax_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "IdxMax(high, 60) / 60",
    },
    "x_imin_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "IdxMin(low, 60) / 60",
    },
    "x_imxd_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "(IdxMax(high, 60) - IdxMin(low, 60)) / 60",
    },
    "x_cntn_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "mean(ret_1 < 0, 20)",
    },
    "x_cntn_60": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "mean(ret_1 < 0, 60)",
    },
    "x_sump_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "sum(max(ret_1, 0), 20) / sum(abs(ret_1), 20)",
    },
    "x_sumn_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "sum(min(ret_1, 0), 20) / sum(abs(ret_1), 20)",
    },
    "x_sumd_20": {
        "family": "price_trend",
        "origin": "qlib_alpha158_style",
        "formula": "sum(ret_1, 20) / sum(abs(ret_1), 20)",
    },
    "x_corr_20": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha158_style",
        "formula": "corr(close, volume, 20)",
    },
    "x_cord_20": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha158_style",
        "formula": "corr(close / Ref(close, 1), volume / Ref(volume, 1), 20)",
    },
    "x_interval_drawdown_60": {
        "family": "risk_tail",
        "origin": "qlib_alpha158_style_extension",
        "formula": "min_u(close_u / max_{v<=u}(close_v) - 1, u in last 60)",
    },
    "x_interval_recovery_60": {
        "family": "risk_tail",
        "origin": "qlib_alpha158_style_extension",
        "formula": "close_t / close_{argmin_u(drawdown_u)} - 1, u in last 60",
    },
}


_REQUIRED_COLUMNS = {"code", "date", "open", "high", "low", "close", "volume"}


def qlib_extra_factor_metadata(feature: str) -> dict[str, str] | None:
    """Return a copy of one factor's audit metadata, if it is registered."""

    metadata = QLIB_EXTRA_FACTOR_SPECS.get(feature)
    return dict(metadata) if metadata else None


def _group_indices(frame: pd.DataFrame) -> Iterable[pd.Index]:
    """Yield sorted row indexes for each instrument without changing input."""

    return frame.groupby("code", sort=False, dropna=True).groups.values()


def _rolling_apply_by_code(
    values: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
    function: Callable[[np.ndarray], float],
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for indices in groups:
        group_values = values.loc[indices]
        result.loc[indices] = group_values.rolling(
            window, min_periods=window
        ).apply(function, raw=True)
    return result


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


def _rolling_mean_by_code(
    values: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for indices in groups:
        result.loc[indices] = values.loc[indices].rolling(
            window, min_periods=window
        ).mean()
    return result


def _rolling_corr_by_code(
    left: pd.Series,
    right: pd.Series,
    groups: Iterable[pd.Index],
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for indices in groups:
        left_values = left.loc[indices]
        right_values = right.loc[indices]
        corr = left_values.rolling(window, min_periods=window).corr(right_values)
        # pandas' rolling correlation uses pairwise valid observations.  The
        # Alpha158-style expression requires a complete causal window, so a
        # missing value in either input invalidates that endpoint explicitly.
        valid_window = (
            left_values.notna().rolling(window, min_periods=window).sum().eq(window)
            & right_values.notna().rolling(window, min_periods=window).sum().eq(window)
        )
        result.loc[indices] = corr.where(valid_window)
    return result


def _index_of_max(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return np.nan
    return float(np.argmax(values))


def _index_of_min(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return np.nan
    return float(np.argmin(values))


def _interval_drawdown(values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown contained in one causal window."""

    if not np.isfinite(values).all() or (values <= 0).any():
        return np.nan
    running_peak = np.maximum.accumulate(values)
    path_drawdown = values / running_peak - 1.0
    return float(np.min(path_drawdown))


def _interval_recovery(values: np.ndarray) -> float:
    """Recovery from the worst in-window drawdown trough to the endpoint.

    A window that has not experienced a drawdown, or whose trough is the
    current observation, receives zero rather than being treated as a
    spurious recovery from the first observation.
    """

    if not np.isfinite(values).all() or (values <= 0).any():
        return np.nan
    running_peak = np.maximum.accumulate(values)
    path_drawdown = values / running_peak - 1.0
    trough_index = int(np.argmin(path_drawdown))
    trough_drawdown = float(path_drawdown[trough_index])
    if trough_drawdown >= 0.0 or trough_index >= len(values) - 1:
        return 0.0
    trough_price = float(values[trough_index])
    return float(values[-1] / trough_price - 1.0) if trough_price > 0 else np.nan


def _cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.groupby(dates, sort=False).rank(pct=True).sub(0.5).mul(2.0)


def add_qlib_extra_factors(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append causal Qlib-style factors and cross-sectional ranks.

    Parameters
    ----------
    frame:
        Daily rows containing ``code``, ``date``, ``open``, ``high``, ``low``,
        ``close`` and ``volume``.  Additional columns are copied through.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        A sorted copy of the input and the 15 new ``x_*`` feature names.  Each
        feature is ranked separately within its signal date to ``[-1, 1]``;
        rows without a complete causal window remain ``NaN``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("因子输入必须是 pandas.DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Qlib 扩展因子缺少输入列: {', '.join(missing)}")

    output = frame.copy(deep=True)
    date_key = pd.to_datetime(output["date"], errors="coerce")
    order = pd.DataFrame(
        {"code": output["code"], "_date_key": date_key}, index=output.index
    ).sort_values(["code", "_date_key"], kind="mergesort").index
    output = output.loc[order].reset_index(drop=True)
    date_key = date_key.loc[order].reset_index(drop=True)

    # Recompute causal returns from OHLCV rather than trusting possibly
    # precomputed columns supplied by another feature layer.
    code = output["code"]
    close = pd.to_numeric(output["close"], errors="coerce")
    high = pd.to_numeric(output["high"], errors="coerce")
    low = pd.to_numeric(output["low"], errors="coerce")
    volume = pd.to_numeric(output["volume"], errors="coerce")
    previous_close = close.groupby(code, sort=False).shift(1)
    previous_volume = volume.groupby(code, sort=False).shift(1)
    ret_1 = close.div(previous_close.where(previous_close.gt(0))).sub(1.0)
    volume_change_1 = volume.div(previous_volume.where(previous_volume.gt(0))).sub(1.0)

    groups = _group_indices(output)
    raw: dict[str, pd.Series] = {}

    for window in (20, 60):
        imax = _rolling_apply_by_code(high, groups, window, _index_of_max)
        imin = _rolling_apply_by_code(low, groups, window, _index_of_min)
        raw[f"imax_{window}"] = imax.div(window)
        raw[f"imin_{window}"] = imin.div(window)
        raw[f"imxd_{window}"] = imax.sub(imin).div(window)

    negative_day = ret_1.lt(0).astype(float).where(ret_1.notna())
    raw["cntn_20"] = _rolling_mean_by_code(negative_day, groups, 20)
    raw["cntn_60"] = _rolling_mean_by_code(negative_day, groups, 60)

    absolute_return_sum = _rolling_sum_by_code(ret_1.abs(), groups, 20)
    positive_return_sum = _rolling_sum_by_code(ret_1.clip(lower=0), groups, 20)
    negative_return_sum = _rolling_sum_by_code(ret_1.clip(upper=0), groups, 20)
    raw["sump_20"] = positive_return_sum.div(absolute_return_sum.replace(0, np.nan))
    raw["sumn_20"] = negative_return_sum.div(absolute_return_sum.replace(0, np.nan))
    raw["sumd_20"] = positive_return_sum.add(negative_return_sum).div(
        absolute_return_sum.replace(0, np.nan)
    )

    raw["corr_20"] = _rolling_corr_by_code(close, volume, groups, 20)
    raw["cord_20"] = _rolling_corr_by_code(ret_1, volume_change_1, groups, 20)

    raw["interval_drawdown_60"] = _rolling_apply_by_code(
        close, groups, 60, _interval_drawdown
    )
    raw["interval_recovery_60"] = _rolling_apply_by_code(
        close, groups, 60, _interval_recovery
    )

    expected_raw_names = [name.removeprefix("x_") for name in QLIB_EXTRA_FACTOR_SPECS]
    if list(raw) != expected_raw_names:
        raise RuntimeError("Qlib 扩展因子计算顺序与元数据不一致")

    feature_columns: list[str] = []
    for raw_name, values in raw.items():
        feature_name = f"x_{raw_name}"
        output[feature_name] = _cross_section_rank(values, date_key)
        feature_columns.append(feature_name)
    return output, feature_columns

