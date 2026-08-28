from __future__ import annotations

from datetime import date, datetime
import hashlib
import json

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR
from app.lgbm_features import LABEL_HORIZON, build_alpha158_lite, mature_training_mask
from analyze_strategy import period_summary
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from optimize_turnover_policy_v1 import period_activity


OUTPUT_PATH = BASE_DIR / "data" / "qlib_ridge_topk_research_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "qlib_ridge_topk_research_v1.md"
CURVES_PATH = BASE_DIR / "reports" / "qlib_ridge_topk_2020_curves.csv"
SOURCE_OPTIMIZATION = BASE_DIR / "data" / "factor_optimization.json"

DATA_START = date(2016, 8, 25)
SELECTION_START = date(2018, 1, 1)
SELECTION_END = date(2019, 12, 31)
TEST_START = date(2020, 1, 1)
TEST_END = date(2020, 12, 31)

BASELINE_WEIGHTS = {
    "medium_reversal": 0.33,
    "inefficiency": 0.23,
    "liquidity_cooling": 0.19,
    "low_volatility": 0.05,
    "risk": 0.10,
    "value": 0.10,
}

# Frozen before evaluation. blend_baseline is the share of the current
# contrarian score in the final cross-sectional prediction rank.
CANDIDATES = {
    "ridge_a10": {"alpha": 10.0, "blend_baseline": 0.0},
    "ridge_a100": {"alpha": 100.0, "blend_baseline": 0.0},
    "ridge_a1000": {"alpha": 1000.0, "blend_baseline": 0.0},
    "ridge_a100_blend20": {"alpha": 100.0, "blend_baseline": 0.20},
    "ridge_a100_blend40": {"alpha": 100.0, "blend_baseline": 0.40},
}


def validate_candidates() -> str:
    for name, config in CANDIDATES.items():
        if set(config) != {"alpha", "blend_baseline"}:
            raise RuntimeError(f"{name} 参数不完整")
        if config["alpha"] <= 0 or not 0 <= config["blend_baseline"] <= 1:
            raise RuntimeError(f"{name} 参数越界")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def baseline_scores(frame: pd.DataFrame, start: date, end: date) -> pd.Series:
    engine = BacktestEngine()
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    selected = frame[
        (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    ]
    for _, cross in selected.groupby("date"):
        eligible = cross[
            cross["tradestatus"].eq(1)
            & cross["is_st"].eq(0)
            & cross["close"].gt(1)
            & cross["amount_20"].ge(10_000_000)
        ].dropna(subset=[
            "ret_5", "ret_20", "ret_60", "volatility_20", "amount_20",
            "amount_60", "absolute_return_60", "ma_20",
        ])
        if "universe_member" in eligible.columns:
            eligible = eligible[eligible["universe_member"].fillna(False).astype(bool)]
        if eligible.empty:
            continue
        scored = engine._score(eligible.copy(), BASELINE_WEIGHTS)
        output.loc[scored.index] = scored["score"].rank(pct=True)
    return output


def fit_expanding_ridge_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    alpha: float,
    prediction_start: date,
    prediction_end: date,
) -> tuple[pd.Series, dict]:
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    date_position = {pd.Timestamp(value): index for index, value in enumerate(dates)}
    prediction_dates = dates[
        (dates >= pd.Timestamp(prediction_start)) & (dates <= pd.Timestamp(prediction_end))
    ]
    monthly_betas = []
    maturity_audit = []

    for month, month_dates in pd.Series(prediction_dates, index=prediction_dates).groupby(
        [prediction_dates.year, prediction_dates.month]
    ):
        first_prediction_date = pd.Timestamp(month_dates.iloc[0])
        first_index = date_position[first_prediction_date]
        mature_index = first_index - (LABEL_HORIZON + 1)
        if mature_index < 0:
            continue
        mature_cutoff = dates[mature_index]
        # ``mature_cutoff`` is retained in the audit as a conservative global
        # reference, but maturity is checked per symbol and per endpoint.
        train = frame[mature_training_mask(frame, first_prediction_date)].copy()
        if train["date"].nunique() < 252 or len(train) < 5_000:
            continue
        x_train = train[feature_columns].fillna(0.0).to_numpy(dtype=float)
        # Rank only labels that are observable at this refit date. Ranking
        # before the maturity filter would leak immature peers' future returns.
        y_train = (
            train["label_return"]
            .groupby(train["date"])
            .rank(pct=True)
            .sub(0.5)
            .to_numpy(dtype=float)
        )
        x_design = np.column_stack([np.ones(len(x_train)), x_train])
        penalty = np.eye(x_design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y_train)

        month_mask = frame["date"].isin(pd.DatetimeIndex(month_dates.values))
        x_predict = frame.loc[month_mask, feature_columns].fillna(0.0).to_numpy(dtype=float)
        predictions.loc[month_mask] = beta[0] + x_predict @ beta[1:]
        monthly_betas.append(beta[1:])
        maturity_audit.append({
            "month": f"{month[0]:04d}-{month[1]:02d}",
            "prediction_start": first_prediction_date.date().isoformat(),
            "last_mature_signal_date": mature_cutoff.date().isoformat(),
            "max_training_label_end_date": pd.to_datetime(
                train["label_end_date"]
            ).max().date().isoformat(),
            "gap_sessions": LABEL_HORIZON + 1,
            "training_rows": int(len(train)),
            "training_dates": int(train["date"].nunique()),
        })

    mean_abs_beta = np.mean(np.abs(np.vstack(monthly_betas)), axis=0) if monthly_betas else np.zeros(len(feature_columns))
    importance = sorted(
        (
            {"feature": feature_columns[index], "mean_abs_coefficient": float(value)}
            for index, value in enumerate(mean_abs_beta)
        ),
        key=lambda row: row["mean_abs_coefficient"],
        reverse=True,
    )
    return predictions, {
        "alpha": alpha,
        "refit_months": len(monthly_betas),
        "maturity_audit": maturity_audit,
        "top_feature_importance": importance[:20],
    }


def blend_scores(
    frame: pd.DataFrame,
    model_prediction: pd.Series,
    baseline_prediction: pd.Series,
    baseline_weight: float,
) -> pd.DataFrame:
    score_columns = ["date", "code", "tradestatus", "is_st", "close", "amount_20"]
    if "universe_member" in frame.columns:
        score_columns.append("universe_member")
    scores = frame[score_columns].copy()
    model_rank = model_prediction.groupby(frame["date"]).rank(pct=True)
    scores["score"] = model_rank.mul(1 - baseline_weight).add(
        baseline_prediction.mul(baseline_weight), fill_value=0
    )
    scores.loc[
        ~scores["tradestatus"].eq(1)
        | ~scores["is_st"].eq(0)
        | ~scores["close"].gt(1)
        | ~scores["amount_20"].ge(10_000_000),
        "score",
    ] = np.nan
    if "universe_member" in scores.columns:
        scores.loc[~scores["universe_member"].fillna(False).astype(bool), "score"] = np.nan
    return scores


def topk_dropout_backtest(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    start: date,
    end: date,
    top_k: int = 5,
    n_drop: int = 1,
    rebalance_days: int = 10,
    cost_bps: float = 12.0,
) -> dict:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    open_prices = data.pivot(index="date", columns="code", values="open").reindex(dates)
    period_returns = open_prices.shift(-1).div(open_prices).sub(1).replace([np.inf, -np.inf], np.nan)
    if "universe_member" in data.columns:
        membership = (
            data.pivot(index="date", columns="code", values="universe_member")
            .reindex(index=dates, columns=open_prices.columns)
            .fillna(False)
            .astype(bool)
        )
        period_returns = period_returns.where(membership)
    benchmark_returns = period_returns.mean(axis=1, skipna=True).fillna(0.0)
    score_lookup = scores.set_index(["date", "code"])["score"]

    pending_targets = {}
    planned_holdings = []
    rebalance_records = []
    targets = pd.DataFrame(index=dates, columns=open_prices.columns, dtype=float)
    for signal_index, signal_date in enumerate(dates):
        if signal_index in pending_targets:
            planned_holdings = pending_targets.pop(signal_index)
        if signal_index % rebalance_days != 0:
            continue
        execution_index = signal_index + 1
        if execution_index >= len(dates) - 1:
            continue
        try:
            cross = score_lookup.loc[signal_date].dropna().sort_values(ascending=False)
        except KeyError:
            continue
        cross = cross[cross.index.isin(open_prices.columns)]
        if len(cross) < top_k:
            continue
        if not planned_holdings:
            target_codes = cross.head(top_k).index.astype(str).tolist()
            dropped = []
        else:
            available_holdings = [code for code in planned_holdings if code in cross.index]
            missing = [code for code in planned_holdings if code not in cross.index]
            ranked_holdings = sorted(available_holdings, key=lambda code: float(cross.loc[code]))
            challengers = [code for code in cross.index.astype(str) if code not in planned_holdings]
            drop_count = min(n_drop, len(ranked_holdings) + len(missing), len(challengers))
            dropped = missing[:drop_count]
            remaining_drop = drop_count - len(dropped)
            dropped.extend(ranked_holdings[:remaining_drop])
            target_codes = [code for code in planned_holdings if code not in dropped]
            target_codes.extend(challengers[:drop_count])
            if len(target_codes) < top_k:
                target_codes.extend(
                    code for code in cross.index.astype(str)
                    if code not in target_codes
                )
                target_codes = target_codes[:top_k]

        execution_date = dates[execution_index]
        targets.loc[execution_date, :] = 0.0
        targets.loc[execution_date, target_codes] = 1 / len(target_codes)
        pending_targets[execution_index] = target_codes
        rebalance_records.append({
            "signal_date": signal_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "holdings": target_codes,
            "dropped": dropped,
        })

    current_weights = pd.Series(0.0, index=open_prices.columns)
    daily_returns = pd.Series(0.0, index=dates)
    turnovers = pd.Series(0.0, index=dates)
    active = pd.Series(False, index=dates)
    for trade_date in dates:
        target = targets.loc[trade_date]
        if target.notna().any():
            target = target.fillna(0.0)
            turnovers.loc[trade_date] = target.sub(current_weights).abs().sum()
            current_weights = target
        active.loc[trade_date] = current_weights.sum() > 0
        returns_today = period_returns.loc[trade_date].fillna(0.0)
        gross_return = float(current_weights.mul(returns_today).sum())
        daily_returns.loc[trade_date] = gross_return
        growth = 1 + gross_return
        if current_weights.sum() > 0 and growth > 0:
            current_weights = current_weights.mul(1 + returns_today).div(growth)

    costs = turnovers * cost_bps / 10_000
    strategy_returns = (daily_returns - costs).where(active, 0.0)
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    metrics = BacktestEngine._metrics(
        strategy_returns, benchmark_returns, strategy_equity, benchmark_equity, turnovers, costs
    )
    return_periods = [
        {
            "period_start": dates[index].date().isoformat(),
            "period_end": dates[index + 1].date().isoformat(),
            "strategy_return": float(strategy_returns.iloc[index]),
            "benchmark_return": float(benchmark_returns.iloc[index]),
            "turnover": float(turnovers.iloc[index]),
            "cost": float(costs.iloc[index]),
        }
        for index in range(len(dates) - 1)
    ]
    return {
        "metrics": metrics,
        "return_periods": return_periods,
        "rebalances": rebalance_records,
        "config": {
            "top_k": top_k,
            "n_drop": n_drop,
            "rebalance_days": rebalance_days,
            "cost_bps": cost_bps,
        },
    }


def research_summary(result: dict, start: date, end: date) -> dict:
    return {
        "performance": period_summary(result, start, end),
        "activity": period_activity(result, start, end),
    }


def candidate_rank(row: dict) -> tuple:
    perf = row["selection"]["performance"]
    activity = row["selection"]["activity"]
    return (
        perf["sharpe"],
        perf["annual_return"],
        perf["information_ratio"],
        -activity["annual_turnover"],
        perf["max_drawdown"],
    )


def production_baseline(
    history: pd.DataFrame,
    codes: list[str],
    include_test_result: bool = False,
) -> dict:
    def run_window(start: date, end: date) -> dict:
        engine = BacktestEngine()
        engine._factor_weights_override = BASELINE_WEIGHTS
        config = BacktestConfig(
            mode="live",
            start_date=start,
            end_date=end,
            codes=codes,
            top_n=5,
            rebalance_days=10,
            initial_capital=1_000_000,
            cost_bps=12,
            strategy="contrarian",
            rebalance_policy="adaptive",
            min_hold_days=5,
            rank_buffer=20,
            score_gap=15,
            max_replacements=1,
            entry_rank=3,
            max_adaptive_per_cycle=1,
        )
        return engine.run(history, config, [])

    selection_result = run_window(SELECTION_START, SELECTION_END)
    test_result = run_window(TEST_START, TEST_END)
    output = {
        "selection": research_summary(selection_result, SELECTION_START, SELECTION_END),
        "test": research_summary(test_result, TEST_START, TEST_END),
    }
    if include_test_result:
        output["_test_result"] = test_result
    return output


def normalized_curve(result: dict, return_column: str) -> pd.Series:
    periods = pd.DataFrame(result["return_periods"])
    periods["period_start"] = pd.to_datetime(periods["period_start"])
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    returns = periods.set_index("period_end")[return_column].astype(float)
    curve = (1 + returns).cumprod()
    initial = pd.Series([1.0], index=[periods["period_start"].iloc[0]])
    return pd.concat([initial, curve]).sort_index()


def export_test_curves(
    winner_result: dict,
    production_result: dict,
) -> pd.DataFrame:
    curves = pd.concat(
        {
            "qlib_ridge_top5": normalized_curve(winner_result, "strategy_return"),
            "production_baseline": normalized_curve(production_result, "strategy_return"),
            "pool_benchmark": normalized_curve(winner_result, "benchmark_return"),
        },
        axis=1,
        join="inner",
    )
    if curves.empty or not np.allclose(curves.iloc[0].to_numpy(), 1.0):
        raise RuntimeError("2020曲线无法对齐到共同起点")
    for column in list(curves.columns):
        curves[f"{column}_drawdown"] = curves[column].div(curves[column].cummax()).sub(1)
    curves.index.name = "date"
    curves.to_csv(CURVES_PATH, encoding="utf-8-sig", float_format="%.10f")
    return curves


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    history = history[history["date"] <= pd.Timestamp(TEST_END)].copy()
    codes = sorted(history["code"].unique().tolist())
    frame, feature_columns = build_alpha158_lite(history)
    baseline_prediction = baseline_scores(frame, SELECTION_START, TEST_END)

    selection_model_cache = {}
    candidate_rows = []
    for name, config in CANDIDATES.items():
        alpha = config["alpha"]
        if alpha not in selection_model_cache:
            selection_model_cache[alpha] = fit_expanding_ridge_predictions(
                frame, feature_columns, alpha, SELECTION_START, SELECTION_END
            )
        model_prediction, model_audit = selection_model_cache[alpha]
        scores = blend_scores(
            frame, model_prediction, baseline_prediction, config["blend_baseline"]
        )
        result = topk_dropout_backtest(
            history, scores, SELECTION_START, SELECTION_END
        )
        candidate_rows.append({
            "name": name,
            "config": config,
            "selection": research_summary(result, SELECTION_START, SELECTION_END),
            "model_refits": model_audit["refit_months"],
        })

    selection_winner = max(candidate_rows, key=candidate_rank)
    winner_config = selection_winner["config"]
    winner_prediction, winner_model_audit = fit_expanding_ridge_predictions(
        frame, feature_columns, winner_config["alpha"], SELECTION_START, TEST_END
    )
    winner_scores = blend_scores(
        frame, winner_prediction, baseline_prediction, winner_config["blend_baseline"]
    )
    winner_result = topk_dropout_backtest(history, winner_scores, TEST_START, TEST_END)
    winner_test = research_summary(winner_result, TEST_START, TEST_END)

    static_scores = blend_scores(frame, baseline_prediction, baseline_prediction, 1.0)
    static_result = topk_dropout_backtest(history, static_scores, TEST_START, TEST_END)
    static_test = research_summary(static_result, TEST_START, TEST_END)
    production = production_baseline(history, codes, include_test_result=True)
    production_result = production.pop("_test_result")
    production_test = production["test"]
    test_curves = export_test_curves(winner_result, production_result)
    test_checks = {
        "annual_return_improved_vs_production": (
            winner_test["performance"]["annual_return"]
            > production_test["performance"]["annual_return"]
        ),
        "sharpe_improved_vs_production": (
            winner_test["performance"]["sharpe"]
            > production_test["performance"]["sharpe"]
        ),
        "drawdown_not_worse_by_3pp": (
            winner_test["performance"]["max_drawdown"]
            >= production_test["performance"]["max_drawdown"] - 0.03
        ),
        "annual_turnover_not_higher": (
            winner_test["activity"]["annual_turnover"]
            <= production_test["activity"]["annual_turnover"]
        ),
    }
    test_passed = all(test_checks.values())
    candidate_rows.sort(key=candidate_rank, reverse=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "strategy": "Alpha158-lite expanding ridge ranker + Top5 Dropout",
        "lightgbm_substitution": (
            "LightGBM/sklearn are unavailable offline; NumPy ridge is used to validate the pipeline, "
            "not as a claim of LightGBM-equivalent capacity."
        ),
        "windows": {
            "data_and_model_warmup": [DATA_START.isoformat(), "2017-12-31"],
            "candidate_selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "online_expanding_test": [TEST_START.isoformat(), TEST_END.isoformat()],
            "2021_plus": "not_evaluated",
        },
        "label": (
            "signal close features; buy next open; target is cross-sectional rank of "
            "10-session next-open-to-open return"
        ),
        "portfolio": "Top5, evaluate every 10 sessions, drop at most 1, 12 bps one-way cost",
        "universe_snapshot_run_id": snapshot_id,
        "candidate_manifest_sha256": manifest,
        "candidate_count": len(CANDIDATES),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "curve_data": str(CURVES_PATH.relative_to(BASE_DIR)),
        "selection_rule": (
            "Rank candidates only on 2018-2019 by Sharpe, annual return, information ratio, "
            "lower annual turnover, and drawdown. Freeze hyperparameters after selecting the unique "
            "winner, then evaluate it on 2020 with monthly expanding refits using only mature labels."
        ),
        "production_baseline": production,
        "static_contrarian_topk_test": static_test,
        "selection_winner": selection_winner,
        "winner_model_audit": winner_model_audit,
        "winner_test": winner_test,
        "test_gate": {"passed": test_passed, "checks": test_checks},
        "production_change": False,
        "promotion_blockers": [
            "只有约16个月预热数据，远少于Qlib官方多年训练窗口",
            "使用冻结的当前Top50历史回填，存在成分、上市和存续偏差",
            "使用当前版本前复权价格，不满足严格as-of价格门禁",
            "NumPy岭回归仅验证流程，尚未复现LightGBM非线性模型",
            "即使2020测试通过，也需至少126个交易日前向模拟盘",
        ],
        "candidates": candidate_rows,
        "warnings": warnings,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    winner_selection = selection_winner["selection"]
    REPORT_PATH.write_text(
        f"""# Qlib-like Alpha158-lite + Ridge + TopK-Dropout v1

- 状态：研究结果，**未修改生产策略**
- 冻结股票池：`{snapshot_id}`
- 候选哈希：`{manifest}`
- 特征数：{len(feature_columns)}
- 预热：2016-08-25至2017-12-31
- 候选选择：2018-2019
- 在线扩展窗口测试：2020（超参数冻结，模型每月仅用已成熟标签重训）
- 训练选择冠军：`{selection_winner['name']}`
- 训练选择年化：{winner_selection['performance']['annual_return']:.2%}
- 训练选择夏普：{winner_selection['performance']['sharpe']:.3f}
- 2020策略年化：{winner_test['performance']['annual_return']:.2%}
- 2020策略夏普：{winner_test['performance']['sharpe']:.3f}
- 2020生产基线年化：{production_test['performance']['annual_return']:.2%}
- 2020生产基线夏普：{production_test['performance']['sharpe']:.3f}
- 2020策略年化换手：{winner_test['activity']['annual_turnover']:.2f}x
- 2020生产基线年化换手：{production_test['activity']['annual_turnover']:.2f}x
- 测试门禁：{'通过但仅限研究' if test_passed else '未通过'}

## 说明

- 当前离线环境没有LightGBM或scikit-learn，因此用NumPy岭回归验证完整研究链路。
- 每月扩展窗口重训，每行训练标签的实际结束日不得晚于预测月首日。
- 2021以后未运行新策略，避免继续使用已揭示窗口调参。
- 当前Top50回填与当前版本前复权价格仍阻止严格无前视认证。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "manifest": manifest,
        "feature_count": len(feature_columns),
        "selection_winner": selection_winner,
        "winner_test": winner_test,
        "production_test": production_test,
        "static_contrarian_topk_test": static_test,
        "test_gate": payload["test_gate"],
        "production_change": False,
        "curve_rows": len(test_curves),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
