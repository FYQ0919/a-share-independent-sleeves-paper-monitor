from __future__ import annotations

import math

import numpy as np
import pandas as pd


FRAMEWORK_FACTOR_SPECS: dict[str, dict[str, str]] = {
    "x_upper_shadow_1": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha158_style",
        "formula": "(high - max(open, close)) / open",
    },
    "x_lower_shadow_1": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha158_style",
        "formula": "(min(open, close) - low) / open",
    },
    "x_body_range_1": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha158_style",
        "formula": "(close - open) / (high - low)",
    },
    "x_true_range_1": {
        "family": "risk_tail",
        "origin": "qlib_alpha158_style",
        "formula": "max(high-low, abs(high-prev_close), abs(low-prev_close)) / prev_close",
    },
    "x_overnight_intraday_spread_1": {
        "family": "microstructure_proxy",
        "origin": "qlib_alpha158_style",
        "formula": "overnight_gap - intraday_return",
    },
}

for _window in (10, 20, 40, 60):
    FRAMEWORK_FACTOR_SPECS.update(
        {
            f"x_parkinson_vol_{_window}": {
                "family": "risk_tail",
                "origin": "parkinson_volatility",
                "formula": f"sqrt(mean(log(high/low)^2, {_window}) / (4*log(2)))",
            },
            f"x_return_autocorr_{_window}": {
                "family": "price_trend",
                "origin": "tsfresh_style",
                "formula": f"corr(return_1, lag(return_1, 1), {_window})",
            },
            f"x_up_day_ratio_{_window}": {
                "family": "price_trend",
                "origin": "qlib_alpha158_style",
                "formula": f"mean(return_1 > 0, {_window})",
            },
            f"x_amihud_{_window}": {
                "family": "liquidity",
                "origin": "amihud_illiquidity",
                "formula": f"mean(abs(return_1) / amount, {_window})",
            },
            f"x_signed_volume_pressure_{_window}": {
                "family": "microstructure_proxy",
                "origin": "qlib_alpha158_style",
                "formula": f"sum(sign(return_1) * volume, {_window}) / sum(volume, {_window})",
            },
            f"x_rsv_{_window}": {
                "family": "price_trend",
                "origin": "qlib_alpha158_style",
                "formula": f"(close - min(low, {_window})) / (max(high, {_window}) - min(low, {_window}))",
            },
        }
    )

for _window in (20, 60):
    FRAMEWORK_FACTOR_SPECS[f"x_return_skew_{_window}"] = {
        "family": "risk_tail",
        "origin": "tsfresh_style",
        "formula": f"skew(return_1, {_window})",
    }


def framework_factor_metadata(feature: str) -> dict[str, str] | None:
    metadata = FRAMEWORK_FACTOR_SPECS.get(feature)
    return dict(metadata) if metadata else None


def add_framework_factors(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append auditable, backward-looking research factors to an Alpha158-lite frame."""
    required = {
        "code", "date", "open", "high", "low", "close", "volume", "amount",
        "ret_1", "gap_1", "intraday_return",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"框架复现因子缺少输入列: {', '.join(missing)}")

    output = frame.copy().sort_values(["code", "date"]).reset_index(drop=True)
    grouped = output.groupby("code", group_keys=False)
    previous_close = grouped["close"].shift(1)
    daily_range = output["high"].sub(output["low"])
    safe_open = output["open"].where(output["open"].gt(0))
    safe_previous_close = previous_close.where(previous_close.gt(0))
    safe_range = daily_range.replace(0, np.nan)

    raw: dict[str, pd.Series] = {
        "upper_shadow_1": output["high"].sub(output[["open", "close"]].max(axis=1)).div(safe_open),
        "lower_shadow_1": output[["open", "close"]].min(axis=1).sub(output["low"]).div(safe_open),
        "body_range_1": output["close"].sub(output["open"]).div(safe_range),
        "true_range_1": pd.concat(
            [
                daily_range,
                output["high"].sub(previous_close).abs(),
                output["low"].sub(previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1).div(safe_previous_close),
        "overnight_intraday_spread_1": output["gap_1"].sub(output["intraday_return"]),
    }

    log_high_low_sq = np.log(
        output["high"].where(output["high"].gt(0)).div(
            output["low"].where(output["low"].gt(0))
        )
    ).pow(2)
    lagged_return = grouped["ret_1"].shift(1)
    up_day = output["ret_1"].gt(0).astype(float).where(output["ret_1"].notna())
    scaled_illiquidity = output["ret_1"].abs().mul(100_000_000).div(
        output["amount"].where(output["amount"].gt(0))
    )
    signed_volume = np.sign(output["ret_1"]).mul(output["volume"])

    for window in (10, 20, 40, 60):
        parkinson_mean = log_high_low_sq.groupby(output["code"]).transform(
            lambda values, w=window: values.rolling(w).mean()
        )
        raw[f"parkinson_vol_{window}"] = parkinson_mean.div(4 * math.log(2)).pow(0.5)
        raw[f"return_autocorr_{window}"] = output["ret_1"].groupby(output["code"]).transform(
            lambda values, w=window: values.rolling(w).corr(lagged_return.loc[values.index])
        )
        raw[f"up_day_ratio_{window}"] = up_day.groupby(output["code"]).transform(
            lambda values, w=window: values.rolling(w).mean()
        )
        raw[f"amihud_{window}"] = scaled_illiquidity.groupby(output["code"]).transform(
            lambda values, w=window: values.rolling(w).mean()
        )
        signed_sum = signed_volume.groupby(output["code"]).transform(
            lambda values, w=window: values.rolling(w).sum()
        )
        volume_sum = grouped["volume"].transform(
            lambda values, w=window: values.rolling(w).sum()
        )
        raw[f"signed_volume_pressure_{window}"] = signed_sum.div(volume_sum.replace(0, np.nan))
        rolling_high = grouped["high"].transform(
            lambda values, w=window: values.rolling(w).max()
        )
        rolling_low = grouped["low"].transform(
            lambda values, w=window: values.rolling(w).min()
        )
        raw[f"rsv_{window}"] = output["close"].sub(rolling_low).div(
            rolling_high.sub(rolling_low).replace(0, np.nan)
        )

    for window in (20, 60):
        raw[f"return_skew_{window}"] = grouped["ret_1"].transform(
            lambda values, w=window: values.rolling(w).skew()
        )

    ranked: dict[str, pd.Series] = {}
    feature_columns: list[str] = []
    for raw_name, values in raw.items():
        feature_name = f"x_{raw_name}"
        if feature_name not in FRAMEWORK_FACTOR_SPECS:
            raise RuntimeError(f"框架复现因子缺少元数据: {feature_name}")
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        ranked[feature_name] = numeric.groupby(output["date"]).rank(pct=True).sub(0.5).mul(2)
        feature_columns.append(feature_name)

    output = pd.concat([output, pd.DataFrame(ranked, index=output.index)], axis=1)
    return output, feature_columns
