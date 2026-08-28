from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from app.research_factors import research_factor_metadata


@dataclass(frozen=True)
class MiningWindow:
    name: str
    start: date
    end: date
    role: str


@dataclass(frozen=True)
class FactorMiningConfig:
    discovery: MiningWindow
    selection: MiningWindow
    diagnostic: MiningWindow
    min_cross_section: int = 10
    min_ic_days: int = 126
    min_abs_discovery_ic: float = 0.01
    min_direction_hit_rate: float = 0.50
    min_selection_retention: float = 0.0
    max_pair_correlation: float = 0.80
    max_factors: int = 15
    ic_lookback_days: int = 504

    def to_dict(self) -> dict:
        payload = asdict(self)
        for name in ("discovery", "selection", "diagnostic"):
            payload[name]["start"] = payload[name]["start"].isoformat()
            payload[name]["end"] = payload[name]["end"].isoformat()
        return payload


def factor_family(feature: str) -> str:
    if feature.startswith("g_"):
        return "generated"
    research_metadata = research_factor_metadata(feature)
    if research_metadata:
        return research_metadata["family"]
    name = feature.removeprefix("x_")
    if name in {"pe", "pb"}:
        return "valuation"
    if name.startswith(("return_", "ma_ratio_", "high_position_", "low_position_", "efficiency_")):
        return "price_trend"
    if name.startswith(("volatility_", "downside_vol_", "max_return_", "min_return_", "amplitude_")):
        return "risk_tail"
    if name.startswith(("amount_", "volume_", "turnover_")):
        return "liquidity"
    if name.startswith(("price_amount_corr_", "close_location_", "gap_", "intraday_return")):
        return "microstructure_proxy"
    return "other"


def build_factor_catalog(feature_columns: Iterable[str]) -> list[dict]:
    catalog = []
    for feature in feature_columns:
        row = {
            "feature": feature,
            "family": factor_family(feature),
            "point_in_time_contract": "signal-date close and earlier only",
            "enabled": True,
        }
        research_metadata = research_factor_metadata(feature)
        if research_metadata:
            row.update(
                origin="framework_replication",
                reference=research_metadata["origin"],
                formula=research_metadata["formula"],
                batch=research_metadata["batch"],
            )
        else:
            row["origin"] = "alpha158_lite"
        catalog.append(row)
    return catalog


def daily_rank_ic(
    frame: pd.DataFrame,
    feature_columns: list[str],
    min_cross_section: int = 10,
) -> pd.DataFrame:
    required = {"date", "label_return", "label_end_date", *feature_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"因子数据缺少列: {', '.join(missing)}")
    eligible = frame[
        frame["label_return"].notna() & frame["label_end_date"].notna()
    ].copy()
    rows = []
    for signal_date, cross in eligible.groupby("date", sort=True):
        labels = pd.to_numeric(cross["label_return"], errors="coerce")
        valid_labels = labels.notna()
        if int(valid_labels.sum()) < min_cross_section:
            continue
        label_rank = labels.rank(pct=True)
        values = cross[feature_columns].apply(pd.to_numeric, errors="coerce")
        valid_features = values.columns[
            values.notna().sum().ge(min_cross_section) & values.std(ddof=0).gt(0)
        ]
        correlations = pd.Series(np.nan, index=feature_columns, dtype=float)
        if len(valid_features):
            correlations.loc[valid_features] = values[valid_features].corrwith(label_rank)
        row = correlations.to_dict()
        row.update(
            date=pd.Timestamp(signal_date),
            label_end_date=pd.to_datetime(cross["label_end_date"], errors="coerce").max(),
            cross_section_size=int(valid_labels.sum()),
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["date", "label_end_date", "cross_section_size", *feature_columns])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def window_ic_rows(ic_rows: pd.DataFrame, window: MiningWindow) -> pd.DataFrame:
    signal_dates = pd.to_datetime(ic_rows["date"], errors="coerce")
    label_end_dates = pd.to_datetime(ic_rows["label_end_date"], errors="coerce")
    return ic_rows[
        signal_dates.ge(pd.Timestamp(window.start))
        & signal_dates.le(pd.Timestamp(window.end))
        & label_end_dates.le(pd.Timestamp(window.end))
    ].copy()


def summarize_ic(values: pd.Series, reference_direction: int | None = None) -> dict:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "ic_days": 0,
            "mean_rank_ic": 0.0,
            "abs_mean_rank_ic": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "direction": 0,
            "direction_hit_rate": 0.0,
        }
    mean_ic = float(clean.mean())
    std = float(clean.std(ddof=0))
    direction = int(reference_direction or (1 if mean_ic >= 0 else -1))
    return {
        "ic_days": int(len(clean)),
        "mean_rank_ic": mean_ic,
        "abs_mean_rank_ic": abs(mean_ic),
        "ic_std": std,
        "ic_ir": mean_ic / std if std > 0 else 0.0,
        "direction": direction,
        "direction_hit_rate": float(clean.mul(direction).gt(0).mean()),
    }


def factor_diagnostics(
    ic_rows: pd.DataFrame,
    feature_columns: list[str],
    config: FactorMiningConfig,
) -> pd.DataFrame:
    staged = {
        "discovery": window_ic_rows(ic_rows, config.discovery),
        "selection": window_ic_rows(ic_rows, config.selection),
        "diagnostic": window_ic_rows(ic_rows, config.diagnostic),
    }
    records = []
    for feature in feature_columns:
        discovery = summarize_ic(staged["discovery"][feature])
        direction = int(discovery["direction"])
        selection = summarize_ic(staged["selection"][feature], direction or None)
        diagnostic = summarize_ic(staged["diagnostic"][feature], direction or None)
        discovery_strength = float(discovery["abs_mean_rank_ic"])
        aligned_selection = float(selection["mean_rank_ic"]) * direction
        retention = aligned_selection / discovery_strength if discovery_strength > 0 else 0.0
        records.append(
            {
                "feature": feature,
                "family": factor_family(feature),
                **{f"discovery_{key}": value for key, value in discovery.items()},
                **{f"selection_{key}": value for key, value in selection.items()},
                **{f"diagnostic_{key}": value for key, value in diagnostic.items()},
                "selection_aligned_ic": aligned_selection,
                "selection_retention": retention,
                "stability_score": discovery_strength
                * np.sqrt(
                    float(discovery["direction_hit_rate"])
                    * float(selection["direction_hit_rate"])
                )
                * max(0.0, min(1.0, retention)),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["stability_score", "discovery_abs_mean_rank_ic", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def discovery_feature_correlation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    discovery: MiningWindow,
) -> pd.DataFrame:
    signal_dates = pd.to_datetime(frame["date"], errors="coerce")
    label_end_dates = pd.to_datetime(frame["label_end_date"], errors="coerce")
    rows = frame[
        signal_dates.ge(pd.Timestamp(discovery.start))
        & signal_dates.le(pd.Timestamp(discovery.end))
        & label_end_dates.le(pd.Timestamp(discovery.end))
    ]
    return rows[feature_columns].corr(min_periods=100)


def select_stable_nonredundant_factors(
    diagnostics: pd.DataFrame,
    correlation: pd.DataFrame,
    config: FactorMiningConfig,
) -> tuple[pd.DataFrame, list[str]]:
    output = diagnostics.copy()
    selected: list[str] = []
    reasons: dict[str, str] = {}
    max_correlations: dict[str, float] = {}
    for row in output.itertuples(index=False):
        feature = str(row.feature)
        reason = ""
        if row.discovery_ic_days < config.min_ic_days:
            reason = "discovery_days"
        elif row.selection_ic_days < config.min_ic_days:
            reason = "selection_days"
        elif row.discovery_abs_mean_rank_ic < config.min_abs_discovery_ic:
            reason = "weak_discovery_ic"
        elif row.discovery_direction_hit_rate < config.min_direction_hit_rate:
            reason = "unstable_discovery_direction"
        elif row.selection_direction_hit_rate < config.min_direction_hit_rate:
            reason = "unstable_selection_direction"
        elif row.selection_retention <= config.min_selection_retention:
            reason = "selection_sign_reversal"
        max_corr = max(
            (
                abs(float(correlation.loc[feature, existing]))
                for existing in selected
                if feature in correlation.index
                and existing in correlation.columns
                and pd.notna(correlation.loc[feature, existing])
            ),
            default=0.0,
        )
        if not reason and max_corr >= config.max_pair_correlation:
            reason = "redundant"
        if not reason and len(selected) >= config.max_factors:
            reason = "factor_limit"
        if not reason:
            selected.append(feature)
            reason = "selected"
        reasons[feature] = reason
        max_correlations[feature] = max_corr
    output["selected"] = output["feature"].isin(selected)
    output["selection_reason"] = output["feature"].map(reasons)
    output["max_abs_corr_to_selected"] = output["feature"].map(max_correlations).fillna(0.0)
    return output, selected


def normalized_ic_weights(ic_history: pd.DataFrame, features: list[str]) -> pd.Series:
    means = ic_history[features].apply(pd.to_numeric, errors="coerce").mean()
    directions = np.sign(means).replace(0.0, np.nan)
    hit_rates = ic_history[features].mul(directions, axis=1).gt(0).mean()
    raw = means.mul(0.5 + 0.5 * hit_rates)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(raw.abs().sum()) == 0:
        return pd.Series(0.0, index=features)
    cap = float(raw.abs().median() * 3)
    if cap > 0:
        raw = raw.clip(-cap, cap)
    return raw.div(raw.abs().sum())


def expanding_factor_scores(
    frame: pd.DataFrame,
    ic_rows: pd.DataFrame,
    selected_features: list[str],
    prediction_start: date,
    prediction_end: date,
    config: FactorMiningConfig,
) -> tuple[pd.Series, list[dict]]:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    if not selected_features:
        return output, []
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).dropna().unique()))
    dates = dates[
        (dates >= pd.Timestamp(prediction_start)) & (dates <= pd.Timestamp(prediction_end))
    ]
    audits = []
    month_groups = pd.Series(dates, index=dates).groupby([dates.year, dates.month])
    label_end = pd.to_datetime(ic_rows["label_end_date"], errors="coerce")
    for (year, month), month_dates in month_groups:
        first_prediction = pd.Timestamp(month_dates.iloc[0])
        available = ic_rows[label_end.lt(first_prediction)].tail(config.ic_lookback_days)
        if len(available) < config.min_ic_days:
            continue
        weights = normalized_ic_weights(available, selected_features)
        month_index = pd.DatetimeIndex(month_dates.to_numpy())
        mask = frame["date"].isin(month_index)
        raw = frame.loc[mask, selected_features].fillna(0.0).mul(weights, axis=1).sum(axis=1)
        output.loc[mask] = raw.groupby(frame.loc[mask, "date"]).rank(pct=True)
        max_label_end = pd.to_datetime(available["label_end_date"], errors="coerce").max()
        audits.append(
            {
                "month": f"{year:04d}-{month:02d}",
                "prediction_start": first_prediction.date().isoformat(),
                "max_training_label_end_date": max_label_end.date().isoformat(),
                "strictly_mature": bool(max_label_end < first_prediction),
                "ic_days": int(len(available)),
                "weights": {key: float(value) for key, value in weights.items()},
            }
        )
    return output, audits


def correlation_edges(
    correlation: pd.DataFrame,
    threshold: float,
    limit: int = 100,
) -> list[dict]:
    edges = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                edges.append({"left": left, "right": right, "correlation": float(value)})
    return sorted(edges, key=lambda row: abs(row["correlation"]), reverse=True)[:limit]
