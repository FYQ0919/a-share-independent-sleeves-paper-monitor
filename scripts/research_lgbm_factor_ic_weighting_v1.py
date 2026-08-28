from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    normalized_curve,
    research_summary,
    topk_dropout_backtest,
)


DISCOVERY_END = date(2017, 12, 31)
SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
DIAGNOSTIC_START = date(2020, 1, 1)
DIAGNOSTIC_END = date(2026, 8, 25)
MAX_FACTORS = 15
MAX_PAIR_CORRELATION = 0.80
IC_LOOKBACK_DAYS = 504
MIN_IC_DAYS = 126

OUTPUT_PATH = BASE_DIR / "data" / "lgbm_factor_ic_weighting_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_factor_ic_weighting_v1.md"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_factor_ic_weighting_curves.csv"
FACTOR_PATH = BASE_DIR / "reports" / "lgbm_factor_ic_diagnostics.csv"

# The blend grid is frozen before the 2020-2026 diagnostic. Zero is retained
# so the correlation overlay can be rejected instead of forcing a promotion.
CANDIDATES = {
    "current_lgbm": {"ic_weight": 0.00},
    "factor_ic_10": {"ic_weight": 0.10},
    "factor_ic_20": {"ic_weight": 0.20},
    "factor_ic_30": {"ic_weight": 0.30},
}


def validate_candidates() -> str:
    weights = [float(config["ic_weight"]) for config in CANDIDATES.values()]
    if weights != [0.0, 0.1, 0.2, 0.3]:
        raise RuntimeError("相关性加权候选必须保持冻结的小网格")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def daily_rank_ic(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    eligible = frame[
        frame["label_return"].notna() & frame["label_end_date"].notna()
    ].copy()
    rows = []
    for signal_date, cross in eligible.groupby("date", sort=True):
        label_rank = cross["label_return"].rank(pct=True)
        values = cross[feature_columns].apply(pd.to_numeric, errors="coerce")
        correlations = values.corrwith(label_rank)
        row = correlations.to_dict()
        row["date"] = pd.Timestamp(signal_date)
        row["label_end_date"] = pd.to_datetime(cross["label_end_date"]).max()
        row["cross_section_size"] = int(label_rank.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def factor_statistics(ic_rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    records = []
    for feature in feature_columns:
        values = pd.to_numeric(ic_rows[feature], errors="coerce").dropna()
        mean_ic = float(values.mean()) if len(values) else 0.0
        direction = 1.0 if mean_ic >= 0 else -1.0
        hit_rate = float((values.mul(direction) > 0).mean()) if len(values) else 0.0
        records.append({
            "feature": feature,
            "ic_days": int(len(values)),
            "mean_rank_ic": mean_ic,
            "abs_mean_rank_ic": abs(mean_ic),
            "ic_std": float(values.std(ddof=0)) if len(values) else 0.0,
            "direction": int(direction),
            "direction_hit_rate": hit_rate,
            "selection_strength": abs(mean_ic) * hit_rate,
        })
    return pd.DataFrame(records).sort_values(
        ["selection_strength", "abs_mean_rank_ic", "feature"],
        ascending=[False, False, True],
    )


def select_nonredundant_factors(
    statistics: pd.DataFrame,
    factor_correlation: pd.DataFrame,
    max_factors: int = MAX_FACTORS,
    max_pair_correlation: float = MAX_PAIR_CORRELATION,
) -> tuple[list[str], dict[str, float]]:
    selected: list[str] = []
    redundancy: dict[str, float] = {}
    eligible = statistics[
        statistics["ic_days"].ge(MIN_IC_DAYS)
        & statistics["direction_hit_rate"].ge(0.50)
    ]
    for feature in eligible["feature"]:
        max_correlation = max(
            (
                abs(float(factor_correlation.loc[feature, existing]))
                for existing in selected
                if pd.notna(factor_correlation.loc[feature, existing])
            ),
            default=0.0,
        )
        if max_correlation >= max_pair_correlation:
            continue
        selected.append(str(feature))
        redundancy[str(feature)] = max_correlation
        if len(selected) >= max_factors:
            break
    if len(selected) < 5:
        raise RuntimeError("去冗余后稳定相关因子不足5个")
    return selected, redundancy


def normalized_ic_weights(ic_history: pd.DataFrame, features: list[str]) -> pd.Series:
    means = ic_history[features].apply(pd.to_numeric, errors="coerce").mean()
    directions = np.sign(means).replace(0.0, np.nan)
    directional_hits = ic_history[features].mul(directions, axis=1).gt(0).mean()
    raw = means.mul(0.5 + 0.5 * directional_hits)
    if not np.isfinite(raw.to_numpy(dtype=float)).all() or raw.abs().sum() == 0:
        return pd.Series(0.0, index=features)
    cap = float(raw.abs().median() * 3)
    if cap > 0:
        raw = raw.clip(lower=-cap, upper=cap)
    return raw.div(raw.abs().sum())


def expanding_factor_scores(
    frame: pd.DataFrame,
    ic_rows: pd.DataFrame,
    selected_features: list[str],
    prediction_start: date,
    prediction_end: date,
) -> tuple[pd.Series, dict]:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"]).unique()))
    prediction_dates = dates[
        (dates >= pd.Timestamp(prediction_start))
        & (dates <= pd.Timestamp(prediction_end))
    ]
    audits = []
    monthly_weights = []
    grouped_months = pd.Series(prediction_dates, index=prediction_dates).groupby(
        [prediction_dates.year, prediction_dates.month]
    )
    for month, month_dates in grouped_months:
        prediction_date = pd.Timestamp(month_dates.iloc[0])
        available = ic_rows[
            pd.to_datetime(ic_rows["label_end_date"]).lt(prediction_date)
        ].tail(IC_LOOKBACK_DAYS)
        if len(available) < MIN_IC_DAYS:
            continue
        weights = normalized_ic_weights(available, selected_features)
        month_mask = frame["date"].isin(pd.DatetimeIndex(month_dates.values))
        raw_score = frame.loc[month_mask, selected_features].fillna(0.0).mul(
            weights, axis=1
        ).sum(axis=1)
        output.loc[month_mask] = raw_score.groupby(frame.loc[month_mask, "date"]).rank(
            pct=True
        )
        monthly_weights.append(weights.rename(f"{month[0]:04d}-{month[1]:02d}"))
        audits.append({
            "month": f"{month[0]:04d}-{month[1]:02d}",
            "prediction_start": prediction_date.date().isoformat(),
            "max_training_label_end_date": pd.to_datetime(
                available["label_end_date"]
            ).max().date().isoformat(),
            "ic_days": int(len(available)),
            "top_weights": [
                {"feature": str(feature), "weight": float(value)}
                for feature, value in weights.reindex(
                    weights.abs().sort_values(ascending=False).index
                ).head(8).items()
            ],
        })
    weight_frame = pd.DataFrame(monthly_weights)
    mean_abs_weights = (
        weight_frame.abs().mean().sort_values(ascending=False)
        if not weight_frame.empty
        else pd.Series(dtype=float)
    )
    return output, {
        "refit_months": len(audits),
        "lookback_ic_days": IC_LOOKBACK_DAYS,
        "maturity_audit": audits,
        "mean_absolute_weights": [
            {"feature": str(feature), "mean_abs_weight": float(value)}
            for feature, value in mean_abs_weights.items()
        ],
    }


def blend_factor_overlay(
    current_scores: pd.DataFrame,
    factor_scores: pd.Series,
    ic_weight: float,
) -> pd.DataFrame:
    output = current_scores.copy()
    aligned = factor_scores.reindex(output.index)
    output["score"] = output["score"].mul(1 - ic_weight).add(
        aligned.mul(ic_weight), fill_value=0.0
    )
    output.loc[current_scores["score"].isna() | aligned.isna(), "score"] = np.nan
    return output


def selection_folds(result: dict) -> dict:
    return {
        "2018": research_summary(result, date(2018, 1, 1), date(2018, 12, 31)),
        "2019": research_summary(result, date(2019, 1, 1), date(2019, 12, 31)),
    }


def candidate_rank(row: dict) -> tuple:
    fold_sharpes = [fold["performance"]["sharpe"] for fold in row["selection_folds"].values()]
    performance = row["selection"]["performance"]
    activity = row["selection"]["activity"]
    return (
        min(fold_sharpes),
        float(np.median(fold_sharpes)),
        performance["sharpe"],
        performance["annual_return"],
        performance["max_drawdown"],
        -activity["annual_turnover"],
    )


def export_curves(weighted_result: dict, current_result: dict) -> int:
    curves = pd.concat(
        {
            "factor_ic_weighted": normalized_curve(weighted_result, "strategy_return"),
            "current_lgbm": normalized_curve(current_result, "strategy_return"),
            "pool_benchmark": normalized_curve(current_result, "benchmark_return"),
        },
        axis=1,
        join="inner",
    )
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVE_PATH, encoding="utf-8-sig", float_format="%.10f")
    return len(curves)


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history = history[history["date"].le(pd.Timestamp(DIAGNOSTIC_END))].copy()
    frame, feature_columns = build_alpha158_lite(history)

    ic_rows = daily_rank_ic(frame, feature_columns)
    discovery_ic = ic_rows[
        pd.to_datetime(ic_rows["label_end_date"]).le(pd.Timestamp(DISCOVERY_END))
    ]
    statistics = factor_statistics(discovery_ic, feature_columns)
    discovery_rows = frame[
        frame["label_end_date"].notna()
        & pd.to_datetime(frame["label_end_date"]).le(pd.Timestamp(DISCOVERY_END))
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        factor_correlation = discovery_rows[feature_columns].corr(min_periods=500)
    selected_features, redundancy = select_nonredundant_factors(
        statistics, factor_correlation
    )
    diagnostics = statistics.copy()
    diagnostics["selected"] = diagnostics["feature"].isin(selected_features)
    diagnostics["max_abs_corr_to_earlier_selected"] = diagnostics["feature"].map(
        redundancy
    )
    diagnostics.to_csv(FACTOR_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")

    baseline_prediction = baseline_scores(frame, SELECTION_START, DIAGNOSTIC_END)
    lgbm_prediction, lgbm_audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, SELECTION_START, DIAGNOSTIC_END
    )
    current_scores = blend_scores(
        frame, lgbm_prediction, baseline_prediction, WINNER_CONFIG["baseline_weight"]
    )
    factor_scores, factor_audit = expanding_factor_scores(
        frame, ic_rows, selected_features, SELECTION_START, DIAGNOSTIC_END
    )

    candidates = []
    score_cache = {}
    for name, config in CANDIDATES.items():
        scores = blend_factor_overlay(current_scores, factor_scores, config["ic_weight"])
        score_cache[name] = scores
        result = topk_dropout_backtest(
            history, scores, SELECTION_START, SELECTION_END
        )
        candidates.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
            "selection_folds": selection_folds(result),
        })
    winner = max(candidates, key=candidate_rank)
    candidates.sort(key=candidate_rank, reverse=True)

    weighted_result = topk_dropout_backtest(
        history, score_cache[winner["name"]], DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    current_result = topk_dropout_backtest(
        history, score_cache["current_lgbm"], DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    weighted_diagnostic = research_summary(
        weighted_result, DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    current_diagnostic = research_summary(
        current_result, DIAGNOSTIC_START, DIAGNOSTIC_END
    )
    checks = {
        "correlation_weight_selected": winner["config"]["ic_weight"] > 0,
        "annual_return_not_lower": (
            weighted_diagnostic["performance"]["annual_return"]
            >= current_diagnostic["performance"]["annual_return"]
        ),
        "sharpe_not_lower": (
            weighted_diagnostic["performance"]["sharpe"]
            >= current_diagnostic["performance"]["sharpe"]
        ),
        "drawdown_not_worse": (
            weighted_diagnostic["performance"]["max_drawdown"]
            >= current_diagnostic["performance"]["max_drawdown"]
        ),
        "turnover_not_higher_by_10pct": (
            weighted_diagnostic["activity"]["annual_turnover"]
            <= current_diagnostic["activity"]["annual_turnover"] * 1.10
        ),
    }
    passed = all(checks.values())
    curve_rows = export_curves(weighted_result, current_result)

    selected_details = []
    stats_by_feature = statistics.set_index("feature")
    average_weights = {
        item["feature"]: item["mean_abs_weight"]
        for item in factor_audit["mean_absolute_weights"]
    }
    for feature in selected_features:
        row = stats_by_feature.loc[feature]
        selected_details.append({
            "feature": feature,
            "discovery_mean_rank_ic": float(row["mean_rank_ic"]),
            "direction_hit_rate": float(row["direction_hit_rate"]),
            "max_abs_corr_to_earlier_selected": float(redundancy[feature]),
            "mean_abs_live_weight": float(average_weights.get(feature, 0.0)),
        })

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "candidate_passed_not_deployed" if passed else "research_only_not_promoted",
        "strategy": "Expanding LGBM LambdaRank plus causal factor-IC overlay",
        "windows": {
            "factor_discovery": ["2016-08-25", DISCOVERY_END.isoformat()],
            "blend_selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "revealed_diagnostic": [DIAGNOSTIC_START.isoformat(), DIAGNOSTIC_END.isoformat()],
        },
        "label_contract": "signal close; next-open entry; t+11 open exit; 10-session return",
        "execution_contract": "Top5; next-open; rebalance every 10 sessions; 12bp one-way cost",
        "correlation_method": (
            "daily cross-sectional Rank IC; select by absolute mean IC and sign hit rate; "
            "greedy removal when factor absolute correlation is at least 0.80"
        ),
        "weight_method": (
            "monthly causal weights from the most recent 504 fully mature IC days; "
            "signed IC magnitude with concentration clipping"
        ),
        "candidate_manifest_sha256": manifest,
        "universe_snapshot_run_id": snapshot_id,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_details,
        "winner": winner,
        "current_lgbm_diagnostic": current_diagnostic,
        "weighted_diagnostic": weighted_diagnostic,
        "promotion_gate": {"passed": passed, "checks": checks},
        "lgbm_audit": {
            "refit_months": lgbm_audit["refit_months"],
            "first": lgbm_audit["maturity_audit"][0],
            "last": lgbm_audit["maturity_audit"][-1],
        },
        "factor_audit": {
            "refit_months": factor_audit["refit_months"],
            "lookback_ic_days": factor_audit["lookback_ic_days"],
            "first": factor_audit["maturity_audit"][0],
            "last": factor_audit["maturity_audit"][-1],
            "mean_absolute_weights": factor_audit["mean_absolute_weights"],
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "curve_rows": curve_rows,
        "factor_diagnostics": str(FACTOR_PATH.relative_to(BASE_DIR)),
        "production_change": False,
        "warnings": warnings + [
            "2020-2026 has already been revealed and is a historical diagnostic, not a fresh blind test.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "Current-vintage adjusted prices are not strict point-in-time prices.",
            "Multiplying raw feature values does not meaningfully reweight a tree model; the IC score is blended at ranking level.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    base_perf = current_diagnostic["performance"]
    weighted_perf = weighted_diagnostic["performance"]
    REPORT_PATH.write_text(
        "\n".join([
            "# LGBM 因子相关性加权研究 v1",
            "",
            f"- 状态：{'历史门禁通过，未部署' if passed else '未通过晋级门禁'}",
            f"- 选择冠军：`{winner['name']}`，IC 排名权重 {winner['config']['ic_weight']:.0%}",
            f"- 入选非冗余因子：{len(selected_features)} 个",
            f"- 现版年化：{base_perf['annual_return']:.2%}",
            f"- 加权版年化：{weighted_perf['annual_return']:.2%}",
            f"- 现版最大回撤：{base_perf['max_drawdown']:.2%}",
            f"- 加权版最大回撤：{weighted_perf['max_drawdown']:.2%}",
            f"- 现版夏普：{base_perf['sharpe']:.3f}",
            f"- 加权版夏普：{weighted_perf['sharpe']:.3f}",
            f"- 晋级门禁：{'通过' if passed else '未通过'}",
            "- 生产修改：否",
            "",
            "## 选择期候选",
            "",
            "| 候选 | IC权重 | 年化 | 最大回撤 | 夏普 | 2018夏普 | 2019夏普 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                "| {name} | {weight:.0%} | {annual:.2%} | {drawdown:.2%} | {sharpe:.3f} | {sharpe_2018:.3f} | {sharpe_2019:.3f} |".format(
                    name=row["name"],
                    weight=row["config"]["ic_weight"],
                    annual=row["selection"]["performance"]["annual_return"],
                    drawdown=row["selection"]["performance"]["max_drawdown"],
                    sharpe=row["selection"]["performance"]["sharpe"],
                    sharpe_2018=row["selection_folds"]["2018"]["performance"]["sharpe"],
                    sharpe_2019=row["selection_folds"]["2019"]["performance"]["sharpe"],
                )
                for row in sorted(candidates, key=lambda item: item["config"]["ic_weight"])
            ],
            "",
            "## 训练期高相关且去冗余因子",
            "",
            *[
                f"- `{item['feature']}`：Rank IC {item['discovery_mean_rank_ic']:+.4f}，方向命中率 {item['direction_hit_rate']:.1%}"
                for item in selected_details[:10]
            ],
            "",
            "因子发现仅使用2017年底前成熟标签，混合权重仅使用2018-2019选择；2020-2026只作为已揭示历史诊断。",
        ]),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "factor_diagnostics": str(FACTOR_PATH),
        "curves": str(CURVE_PATH),
        "selected_features": selected_details,
        "winner": winner,
        "current_lgbm_diagnostic": current_diagnostic,
        "weighted_diagnostic": weighted_diagnostic,
        "promotion_gate": payload["promotion_gate"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
