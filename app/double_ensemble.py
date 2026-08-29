from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from app.lgbm_strategy import eligible_training_mask


@dataclass(frozen=True)
class DoubleEnsembleConfig:
    """CPU-scaled structural reproduction of Qlib DoubleEnsemble."""

    num_models: int = 4
    feature_ratios: tuple[float, ...] = (1.0, 0.80, 0.60, 0.40)
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 300
    n_estimators: int = 100
    feature_fraction: float = 0.80
    residual_weight_floor: float = 0.50
    residual_weight_ceiling: float = 1.50
    min_features: int = 16
    seed: int = 20260829

    def validate(self) -> None:
        if self.num_models < 2:
            raise ValueError("DoubleEnsemble requires at least two submodels")
        if len(self.feature_ratios) != self.num_models:
            raise ValueError("feature_ratios must match num_models")
        if self.feature_ratios[0] != 1.0:
            raise ValueError("the first submodel must use all features")
        if any(not 0.0 < ratio <= 1.0 for ratio in self.feature_ratios):
            raise ValueError("feature ratios must be in (0, 1]")
        if not 0.0 < self.residual_weight_floor <= self.residual_weight_ceiling:
            raise ValueError("invalid residual weight range")

    def to_dict(self) -> dict:
        return asdict(self)


def cross_sectional_regression_target(train: pd.DataFrame) -> np.ndarray:
    """Continuous date-wise return rank used by the paper-style regressors."""
    target = (
        train["label_return"]
        .groupby(train["date"], sort=False)
        .rank(method="average", pct=True)
        .sub(0.5)
    )
    return target.to_numpy(dtype=np.float32)


def residual_sample_weights(
    dates: Iterable,
    residuals: np.ndarray | None,
    floor: float = 0.50,
    ceiling: float = 1.50,
) -> np.ndarray:
    """Balance dates, then emphasize difficult samples within each date."""
    date_series = pd.Series(pd.to_datetime(np.asarray(list(dates))))
    counts = date_series.groupby(date_series, sort=False).transform("size").to_numpy(
        dtype=float
    )
    date_balance = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    date_balance *= len(date_balance) / max(date_balance.sum(), 1e-12)
    if residuals is None:
        return date_balance.astype(np.float32)

    error = pd.Series(np.abs(np.asarray(residuals, dtype=float)))
    percentile = error.groupby(date_series, sort=False).rank(
        method="average", pct=True
    )
    difficulty = floor + (ceiling - floor) * percentile.to_numpy(dtype=float)
    weights = date_balance * difficulty
    weights *= len(weights) / max(weights.sum(), 1e-12)
    return weights.astype(np.float32)


def residual_feature_scores(
    matrix: np.ndarray,
    residuals: np.ndarray,
    previous_features: list[int],
    previous_gains: np.ndarray,
) -> np.ndarray:
    """Combine prior tree gain with residual correlation for dynamic selection."""
    values = np.asarray(matrix, dtype=np.float64)
    residual = np.asarray(residuals, dtype=np.float64)
    residual = residual - np.nanmean(residual)
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    numerator = np.abs(np.nansum(centered * residual[:, None], axis=0))
    denominator = np.sqrt(
        np.nansum(centered * centered, axis=0) * np.nansum(residual * residual)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros(values.shape[1], dtype=float),
        where=denominator > 1e-12,
    )
    gain_scores = np.zeros(values.shape[1], dtype=float)
    if len(previous_features):
        gain_scores[np.asarray(previous_features, dtype=int)] = np.asarray(
            previous_gains, dtype=float
        )
    for scores in (correlations, gain_scores):
        maximum = float(np.nanmax(scores)) if len(scores) else 0.0
        if maximum > 0:
            scores /= maximum
    return 0.5 * correlations + 0.5 * gain_scores


def selected_feature_indices(scores: np.ndarray, ratio: float, minimum: int) -> list[int]:
    count = min(len(scores), max(minimum, int(np.ceil(len(scores) * ratio))))
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
    return sorted(order[:count].astype(int).tolist())


def lightgbm_params(config: DoubleEnsembleConfig, submodel_index: int) -> dict:
    return {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_child_samples,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "seed": config.seed + submodel_index,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def fit_double_ensemble(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_columns: list[str],
    config: DoubleEnsembleConfig,
) -> tuple[np.ndarray, dict]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("DoubleEnsemble reproduction requires lightgbm") from exc

    config.validate()
    all_train = train[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    all_predict = predict[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    target = cross_sectional_regression_target(train)
    ensemble_train = np.zeros(len(train), dtype=float)
    ensemble_predict = np.zeros(len(predict), dtype=float)
    residuals = None
    feature_indices = list(range(len(feature_columns)))
    feature_scores = np.ones(len(feature_columns), dtype=float)
    model_audit = []

    for model_index in range(config.num_models):
        if model_index:
            feature_indices = selected_feature_indices(
                feature_scores,
                config.feature_ratios[model_index],
                config.min_features,
            )
        sample_weights = residual_sample_weights(
            train["date"],
            residuals,
            config.residual_weight_floor,
            config.residual_weight_ceiling,
        )
        names = [feature_columns[index] for index in feature_indices]
        dataset = lgb.Dataset(
            all_train[:, feature_indices],
            label=target,
            weight=sample_weights,
            feature_name=names,
            free_raw_data=True,
        )
        model = lgb.train(
            lightgbm_params(config, model_index),
            dataset,
            num_boost_round=config.n_estimators,
        )
        train_prediction = model.predict(all_train[:, feature_indices])
        prediction = model.predict(all_predict[:, feature_indices])
        ensemble_train = (
            ensemble_train * model_index + train_prediction
        ) / (model_index + 1)
        ensemble_predict = (
            ensemble_predict * model_index + prediction
        ) / (model_index + 1)
        residuals = target - ensemble_train
        gains = model.feature_importance(importance_type="gain").astype(float)
        feature_scores = residual_feature_scores(
            all_train, residuals, feature_indices, gains
        )
        model_audit.append({
            "submodel": model_index + 1,
            "feature_count": len(feature_indices),
            "weight_min": float(sample_weights.min()),
            "weight_max": float(sample_weights.max()),
            "residual_rmse": float(np.sqrt(np.mean(residuals**2))),
            "top_features": [
                feature_columns[index]
                for index in np.argsort(-feature_scores, kind="mergesort")[:10]
            ],
        })

    return ensemble_predict, {"submodels": model_audit}


def fit_expanding_double_ensemble_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: DoubleEnsembleConfig,
    prediction_start: date,
    prediction_end: date,
) -> tuple[pd.Series, dict]:
    config.validate()
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).unique()))
    prediction_dates = dates[
        (dates >= pd.Timestamp(prediction_start))
        & (dates <= pd.Timestamp(prediction_end))
    ]
    maturity_audit = []
    feature_counts: dict[str, int] = {feature: 0 for feature in feature_columns}
    grouped_months = pd.Series(prediction_dates, index=prediction_dates).groupby(
        [prediction_dates.year, prediction_dates.month]
    )

    for month, month_dates in grouped_months:
        first_prediction_date = pd.Timestamp(month_dates.iloc[0])
        train = frame.loc[eligible_training_mask(frame, first_prediction_date)].copy()
        counts = train.groupby("date")["code"].transform("size")
        train = train.loc[counts.ge(5)].sort_values(
            ["date", "code"], kind="mergesort"
        )
        if train["date"].nunique() < 252 or len(train) < 5_000:
            continue
        month_mask = frame["date"].isin(pd.DatetimeIndex(month_dates.values))
        predict = frame.loc[month_mask]
        month_prediction, audit = fit_double_ensemble(
            train, predict, feature_columns, config
        )
        predictions.loc[month_mask] = month_prediction
        for submodel in audit["submodels"]:
            for feature in submodel["top_features"]:
                feature_counts[feature] += 1
        max_label_end = pd.to_datetime(
            train["label_end_date"], errors="coerce"
        ).max()
        maturity_audit.append({
            "month": f"{month[0]:04d}-{month[1]:02d}",
            "prediction_start": first_prediction_date.date().isoformat(),
            "max_training_label_end_date": max_label_end.date().isoformat(),
            "strictly_mature": bool(max_label_end < first_prediction_date),
            "training_rows": int(len(train)),
            "training_dates": int(train["date"].nunique()),
            "submodels": audit["submodels"],
        })

    top_features = sorted(
        (
            {"feature": feature, "top10_frequency": count}
            for feature, count in feature_counts.items()
        ),
        key=lambda row: (-row["top10_frequency"], row["feature"]),
    )
    return predictions, {
        "config": config.to_dict(),
        "refit_months": len(maturity_audit),
        "maturity_audit": maturity_audit,
        "all_refits_strictly_mature": bool(maturity_audit)
        and all(row["strictly_mature"] for row in maturity_audit),
        "top_feature_frequency": top_features[:25],
    }
