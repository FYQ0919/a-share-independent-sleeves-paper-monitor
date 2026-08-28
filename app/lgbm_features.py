from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestEngine


LABEL_HORIZON = 10


def build_alpha158_lite(history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engine = BacktestEngine()
    frame = engine._features(history).sort_values(["code", "date"]).reset_index(drop=True)
    grouped = frame.groupby("code", group_keys=False)
    previous_close = grouped["close"].shift(1)
    frame["gap_1"] = frame["open"].div(previous_close).sub(1)
    frame["amplitude_1"] = frame["high"].sub(frame["low"]).div(frame["open"])
    intraday_range = frame["high"].sub(frame["low"])
    frame["close_location_1"] = frame["close"].sub(frame["low"]).div(
        intraday_range.replace(0, np.nan)
    )
    frame.loc[intraday_range.eq(0), "close_location_1"] = 0.5
    frame["volume_change_1"] = grouped["volume"].pct_change(fill_method=None)
    frame["amount_change_1"] = grouped["amount"].pct_change(fill_method=None)

    raw_features = [
        "gap_1", "amplitude_1", "close_location_1", "intraday_return",
        "volume_change_1", "amount_change_1", "pe", "pb",
    ]
    for window in (2, 3, 5, 10, 20, 40, 60, 120):
        name = f"return_{window}"
        frame[name] = grouped["close"].pct_change(window, fill_method=None)
        raw_features.append(name)

    for window in (5, 10, 20, 40, 60, 120):
        close_mean = grouped["close"].transform(lambda values, w=window: values.rolling(w).mean())
        close_high = grouped["close"].transform(lambda values, w=window: values.rolling(w).max())
        close_low = grouped["close"].transform(lambda values, w=window: values.rolling(w).min())
        amount_mean = grouped["amount"].transform(lambda values, w=window: values.rolling(w).mean())
        volume_mean = grouped["volume"].transform(lambda values, w=window: values.rolling(w).mean())
        turnover_mean = grouped["turnover"].transform(lambda values, w=window: values.rolling(w).mean())
        absolute_return = grouped["ret_1"].transform(
            lambda values, w=window: values.abs().rolling(w).sum()
        )
        downside = grouped["ret_1"].transform(
            lambda values, w=window: values.clip(upper=0).pow(2).rolling(w).mean().pow(0.5)
        )
        feature_values = {
            f"ma_ratio_{window}": frame["close"].div(close_mean).sub(1),
            f"high_position_{window}": frame["close"].div(close_high).sub(1),
            f"low_position_{window}": frame["close"].div(close_low).sub(1),
            f"volatility_{window}": grouped["ret_1"].transform(
                lambda values, w=window: values.rolling(w).std()
            ),
            f"downside_vol_{window}": downside,
            f"amount_ratio_{window}": frame["amount"].div(amount_mean).sub(1),
            f"volume_ratio_{window}": frame["volume"].div(volume_mean).sub(1),
            f"turnover_ratio_{window}": frame["turnover"].div(turnover_mean).sub(1),
            f"efficiency_{window}": frame[f"return_{window}"].div(
                absolute_return.replace(0, np.nan)
            ),
            f"max_return_{window}": grouped["ret_1"].transform(
                lambda values, w=window: values.rolling(w).max()
            ),
            f"min_return_{window}": grouped["ret_1"].transform(
                lambda values, w=window: values.rolling(w).min()
            ),
        }
        for name, values in feature_values.items():
            frame[name] = values
            raw_features.append(name)

    log_amount_change = np.log1p(frame["amount"].clip(lower=0)).groupby(frame["code"]).diff()
    for window in (10, 20, 40):
        name = f"price_amount_corr_{window}"
        frame[name] = frame.groupby("code", group_keys=False).apply(
            lambda group, w=window: group["ret_1"].rolling(w).corr(
                log_amount_change.loc[group.index]
            ),
            include_groups=False,
        ).reset_index(level=0, drop=True)
        raw_features.append(name)

    feature_columns = []
    ranked_features = {}
    for raw_name in raw_features:
        feature_name = f"x_{raw_name}"
        numeric = pd.to_numeric(frame[raw_name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ranks = numeric.groupby(frame["date"]).rank(pct=True)
        ranked_features[feature_name] = ranks.sub(0.5).mul(2)
        feature_columns.append(feature_name)

    frame = pd.concat([frame, pd.DataFrame(ranked_features, index=frame.index)], axis=1)
    entry_open = grouped["open"].shift(-1)
    exit_open = grouped["open"].shift(-(LABEL_HORIZON + 1))
    frame["label_end_date"] = grouped["date"].shift(-(LABEL_HORIZON + 1))
    frame["label_return"] = exit_open.div(entry_open).sub(1)
    frame["label_rank"] = frame["label_return"].groupby(frame["date"]).rank(pct=True).sub(0.5)
    return frame, feature_columns


def mature_training_mask(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    cutoff = pd.Timestamp(as_of)
    end_dates = pd.to_datetime(frame["label_end_date"], errors="coerce")
    return frame["label_return"].notna() & end_dates.notna() & end_dates.le(cutoff)
