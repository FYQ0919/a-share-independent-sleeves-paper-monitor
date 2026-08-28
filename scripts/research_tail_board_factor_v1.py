from __future__ import annotations

from datetime import date, datetime
import hashlib
import json

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine, _rank
from app.config import BASE_DIR
from analyze_strategy import period_summary
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import (
    START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    load_current_history,
)
from optimize_turnover_policy_v1 import period_activity


OUTPUT_PATH = BASE_DIR / "data" / "tail_board_factor_research_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "tail_board_factor_research_v1.md"
SOURCE_OPTIMIZATION = BASE_DIR / "data" / "factor_optimization.json"

BASELINE_NAME = "contrarian_quality_10"
BASELINE_WEIGHTS = {
    "medium_reversal": 0.33,
    "inefficiency": 0.23,
    "liquidity_cooling": 0.19,
    "low_volatility": 0.05,
    "risk": 0.10,
    "value": 0.10,
}

# These weights are fixed before any diagnostic or backtest is evaluated.
CANDIDATES = {
    BASELINE_NAME: BASELINE_WEIGHTS,
    "tail_board_02": {
        "medium_reversal": 0.32, "inefficiency": 0.23, "liquidity_cooling": 0.18,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "tail_board": 0.02,
    },
    "tail_board_04": {
        "medium_reversal": 0.31, "inefficiency": 0.22, "liquidity_cooling": 0.18,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "tail_board": 0.04,
    },
    "tail_board_06": {
        "medium_reversal": 0.30, "inefficiency": 0.22, "liquidity_cooling": 0.17,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "tail_board": 0.06,
    },
    "tail_board_08": {
        "medium_reversal": 0.29, "inefficiency": 0.21, "liquidity_cooling": 0.17,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "tail_board": 0.08,
    },
    "limit_proximity_03": {
        "medium_reversal": 0.32, "inefficiency": 0.22, "liquidity_cooling": 0.18,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "limit_proximity": 0.03,
    },
    "close_strength_03": {
        "medium_reversal": 0.32, "inefficiency": 0.22, "liquidity_cooling": 0.18,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "close_strength": 0.03,
    },
    "turnover_breakout_03": {
        "medium_reversal": 0.32, "inefficiency": 0.22, "liquidity_cooling": 0.18,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10, "turnover_breakout": 0.03,
    },
    "tail_close_mix": {
        "medium_reversal": 0.30, "inefficiency": 0.21, "liquidity_cooling": 0.17,
        "low_volatility": 0.05, "risk": 0.10, "value": 0.10,
        "tail_board": 0.05, "close_strength": 0.02,
    },
}

TRAINING_FOLDS = {
    "2016_2017": (START, date(2017, 12, 31)),
    "2018_stress": (date(2018, 1, 1), date(2018, 12, 31)),
    "2019_recovery": (date(2019, 1, 1), date(2019, 12, 31)),
    "2020_pandemic": (date(2020, 1, 1), TRAIN_END),
}


class TailBoardResearchEngine(BacktestEngine):
    def _features(self, history: pd.DataFrame):
        frame = super()._features(history)
        grouped = frame.groupby("code", group_keys=False)
        frame["listing_age_sessions"] = grouped.cumcount() + 1
        frame["turnover_20"] = grouped["turnover"].transform(
            lambda values: values.rolling(20).mean()
        )

        intraday_range = frame["high"].sub(frame["low"])
        frame["close_location"] = frame["close"].sub(frame["low"]).div(
            intraday_range.replace(0, np.nan)
        )
        frame.loc[intraday_range.eq(0), "close_location"] = 1.0
        frame["close_location"] = frame["close_location"].clip(0, 1)

        codes = frame["code"].astype(str)
        after_chinext_reform = frame["date"] >= pd.Timestamp("2020-08-24")
        twenty_percent_board = codes.str.startswith("688") | (
            codes.str.startswith("300") & after_chinext_reform
        )
        frame["limit_pct"] = np.where(twenty_percent_board, 20.0, 10.0)
        frame.loc[frame["is_st"].eq(1), "limit_pct"] = 5.0
        frame["limit_proximity"] = frame["pct_change"].div(frame["limit_pct"])
        frame["turnover_breakout"] = frame["turnover"].div(
            frame["turnover_20"].replace(0, np.nan)
        )

        seasoned = frame["listing_age_sessions"] >= 60
        tradable = seasoned & frame["is_st"].eq(0) & frame["tradestatus"].eq(1)
        return_component = frame["limit_proximity"].clip(lower=0, upper=1)
        turnover_confirmation = frame["turnover_breakout"].clip(lower=0, upper=2).div(2)
        frame["tail_board_raw"] = (
            return_component
            * frame["close_location"]
            * (0.75 + 0.25 * turnover_confirmation)
        ).where(tradable)
        frame.loc[~tradable, [
            "limit_proximity", "close_location", "turnover_breakout", "tail_board_raw"
        ]] = np.nan
        return frame

    def _score(self, cross: pd.DataFrame, factor_weights=None):
        scored = super()._score(cross, {"risk": 1.0})
        scored["limit_proximity_score"] = _rank(scored["limit_proximity"])
        scored["close_strength_score"] = _rank(scored["close_location"])
        scored["turnover_breakout_score"] = _rank(scored["turnover_breakout"])
        scored["tail_board_score"] = _rank(scored["tail_board_raw"])
        weights = factor_weights or self.FACTOR_WEIGHTS
        scored["score"] = sum(
            scored[f"{name}_score"] * weight for name, weight in weights.items()
        ) * 100
        return scored


def validate_candidates() -> str:
    for name, weights in CANDIDATES.items():
        if not np.isclose(sum(weights.values()), 1.0):
            raise RuntimeError(f"{name} 权重和不是1")
        if any(value < 0 for value in weights.values()):
            raise RuntimeError(f"{name} 包含负权重")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def install_score_cache(engine: TailBoardResearchEngine) -> dict[str, pd.DataFrame]:
    original_score = engine._score
    cache: dict[str, pd.DataFrame] = {}

    def cached_score(cross: pd.DataFrame, factor_weights=None):
        key = pd.Timestamp(cross["date"].iloc[0]).date().isoformat()
        if key not in cache:
            cache[key] = original_score(cross.copy(), {"risk": 1.0})
        scored = cache[key].copy()
        weights = factor_weights or engine.FACTOR_WEIGHTS
        scored["score"] = sum(
            scored[f"{factor}_score"] * weight for factor, weight in weights.items()
        ) * 100
        return scored

    engine._score = cached_score
    return cache


def rank_ic(frame: pd.DataFrame, factor: str, horizon: int) -> dict:
    working = frame.copy()
    grouped = working.groupby("code", group_keys=False)
    entry_open = grouped["open"].shift(-1)
    exit_open = grouped["open"].shift(-(horizon + 1))
    working["forward_return"] = exit_open.div(entry_open).sub(1)
    working = working[
        (working["date"] >= pd.Timestamp(START))
        & (working["date"] <= pd.Timestamp(TRAIN_END))
    ]
    values = []
    step = horizon if horizon > 1 else 1
    dates = sorted(working["date"].dropna().unique())[::step]
    for signal_date in dates:
        cross = working[working["date"] == signal_date][[factor, "forward_return"]].dropna()
        if len(cross) < 10:
            continue
        factor_rank = cross[factor].rank(pct=True)
        return_rank = cross["forward_return"].rank(pct=True)
        if factor_rank.std() == 0 or return_rank.std() == 0:
            continue
        values.append(float(factor_rank.corr(return_rank)))
    return {
        "horizon_sessions": horizon,
        "mean_ic": float(np.mean(values)) if values else 0.0,
        "positive_rate": float(np.mean(np.array(values) > 0)) if values else 0.0,
        "observations": len(values),
    }


def event_diagnostics(frame: pd.DataFrame) -> dict:
    training = frame[
        (frame["date"] >= pd.Timestamp(START))
        & (frame["date"] <= pd.Timestamp(TRAIN_END))
        & (frame["listing_age_sessions"] >= 60)
        & frame["is_st"].eq(0)
    ].copy()
    grouped = training.groupby("code", group_keys=False)
    entry_open = grouped["open"].shift(-1)
    diagnostics = {
        "eligible_rows": int(len(training)),
        "eligible_codes": int(training["code"].nunique()),
    }
    for horizon in (1, 3, 10):
        exit_open = grouped["open"].shift(-(horizon + 1))
        training[f"forward_{horizon}"] = exit_open.div(entry_open).sub(1)
    events = training[
        (training["limit_proximity"] >= 0.95)
        & (training["close_location"] >= 0.90)
    ]
    strong_closes = training[
        (training["limit_proximity"] >= 0.50)
        & (training["close_location"] >= 0.90)
    ]
    diagnostics["near_limit_events"] = int(len(events))
    diagnostics["near_limit_dates"] = int(events["date"].nunique())
    diagnostics["near_limit_codes"] = int(events["code"].nunique())
    diagnostics["strong_close_events"] = int(len(strong_closes))
    for label, sample in (("near_limit", events), ("strong_close", strong_closes)):
        diagnostics[label] = {
            f"open_to_open_{horizon}d_mean": float(sample[f"forward_{horizon}"].mean())
            for horizon in (1, 3, 10)
        }
        diagnostics[label].update({
            f"open_to_open_{horizon}d_win_rate": float(
                sample[f"forward_{horizon}"].gt(0).mean()
            )
            for horizon in (1, 3, 10)
        })
    return diagnostics


def period_result(result: dict, start: date, end: date) -> dict:
    return {
        "performance": period_summary(result, start, end),
        "activity": period_activity(result, start, end),
    }


def training_summary(result: dict) -> dict:
    aggregate = period_result(result, START, TRAIN_END)
    folds = {
        label: period_result(result, start, end)
        for label, (start, end) in TRAINING_FOLDS.items()
    }
    fold_sharpes = [item["performance"]["sharpe"] for item in folds.values()]
    return {
        "aggregate": aggregate,
        "folds": folds,
        "minimum_fold_sharpe": min(fold_sharpes),
        "median_fold_sharpe": float(np.median(fold_sharpes)),
    }


def training_rank(row: dict) -> tuple:
    aggregate = row["training"]["aggregate"]
    return (
        row["training"]["minimum_fold_sharpe"],
        row["training"]["median_fold_sharpe"],
        aggregate["performance"]["sharpe"],
        aggregate["performance"]["annual_return"],
        -aggregate["activity"]["annual_turnover"],
    )


def is_training_improver(candidate: dict, baseline: dict) -> bool:
    candidate_perf = candidate["training"]["aggregate"]["performance"]
    baseline_perf = baseline["training"]["aggregate"]["performance"]
    candidate_turnover = candidate["training"]["aggregate"]["activity"]["annual_turnover"]
    baseline_turnover = baseline["training"]["aggregate"]["activity"]["annual_turnover"]
    return (
        candidate_perf["annual_return"] > baseline_perf["annual_return"]
        and candidate_perf["sharpe"] > baseline_perf["sharpe"]
        and candidate_perf["max_drawdown"] >= baseline_perf["max_drawdown"] - 0.03
        and candidate_turnover <= baseline_turnover * 1.05
    )


def make_config(codes: list[str], end_date: date) -> BacktestConfig:
    return BacktestConfig(
        mode="live",
        start_date=START,
        end_date=end_date,
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


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    candidates, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(candidates)
    loaded_codes = sorted(history["code"].unique().tolist())

    engine = TailBoardResearchEngine()
    feature_frame = engine._features(history)
    diagnostics = {
        "events": event_diagnostics(feature_frame),
        "rank_ic": {
            factor: [rank_ic(feature_frame, factor, horizon) for horizon in (1, 3, 10)]
            for factor in (
                "tail_board_raw", "limit_proximity", "close_location", "turnover_breakout"
            )
        },
    }
    engine._features = lambda _: feature_frame
    score_cache = install_score_cache(engine)

    rows = []
    for name, weights in CANDIDATES.items():
        engine._factor_weights_override = weights
        result = engine.run(history, make_config(loaded_codes, TRAIN_END), warnings)
        rows.append({"name": name, "weights": weights, "training": training_summary(result)})

    baseline = next(row for row in rows if row["name"] == BASELINE_NAME)
    improvers = [row for row in rows if is_training_improver(row, baseline)]
    training_winner = max(improvers, key=training_rank) if improvers else baseline

    validation_targets = {BASELINE_NAME, training_winner["name"]}
    for row in rows:
        if row["name"] not in validation_targets:
            continue
        engine._factor_weights_override = row["weights"]
        result = engine.run(history, make_config(loaded_codes, VALIDATION_END), warnings)
        row["reused_validation_diagnostic"] = period_result(
            result, VALIDATION_START, VALIDATION_END
        )

    baseline_validation = baseline["reused_validation_diagnostic"]
    winner_validation = training_winner["reused_validation_diagnostic"]
    validation_checks = {
        "annual_return_improved": (
            winner_validation["performance"]["annual_return"]
            > baseline_validation["performance"]["annual_return"]
        ),
        "sharpe_not_worse_by_0_05": (
            winner_validation["performance"]["sharpe"]
            >= baseline_validation["performance"]["sharpe"] - 0.05
        ),
        "drawdown_not_worse_by_3pp": (
            winner_validation["performance"]["max_drawdown"]
            >= baseline_validation["performance"]["max_drawdown"] - 0.03
        ),
    }
    reused_validation_passed = all(validation_checks.values())
    rows.sort(key=training_rank, reverse=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_reused_validation_not_promotable",
        "research_question": (
            "Can daily close/limit-up proxies improve the existing next-open Top5 strategy?"
        ),
        "not_a_true_tail_auction_backtest": True,
        "missing_required_intraday_fields": [
            "14:30后分钟线", "首次封板时间", "封单额", "炸板次数", "尾盘可成交量", "集合竞价成交"
        ],
        "windows": {
            "training_selection": [START.isoformat(), TRAIN_END.isoformat()],
            "reused_validation_diagnostic": [
                VALIDATION_START.isoformat(), VALIDATION_END.isoformat()
            ],
            "2024_plus_candidate_evaluation": "not_run",
        },
        "execution": "收盘代理信号，下一交易日开盘生效，单边成本12bp，自适应Top5",
        "universe_snapshot_run_id": snapshot_id,
        "candidate_manifest_sha256": manifest,
        "candidate_count": len(CANDIDATES),
        "diagnostics": diagnostics,
        "baseline": baseline,
        "training_winner": training_winner,
        "reused_validation_checks": {
            "passed": reused_validation_passed,
            "checks": validation_checks,
            "promotion_effect": "none; this validation window was already revealed",
        },
        "production_change": False,
        "score_cache_dates": len(score_cache),
        "candidates": rows,
        "warnings": warnings + [
            "日线代理不能证明尾盘打板可成交，也不能识别封单和炸板",
            "当前Top50股票池不代表典型打板小盘股机会集",
            "2021-2023已被既往研究多次查看，本次仅作复用诊断而非新样本外证明",
            "2024-2026未对新候选运行，避免继续用已揭示数据调参",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    base_train = baseline["training"]["aggregate"]
    winner_train = training_winner["training"]["aggregate"]
    events = diagnostics["events"]
    REPORT_PATH.write_text(
        f"""# 尾盘打板日线代理因子研究 v1

- 状态：研究结果，**不可据此上线真正尾盘打板策略**
- 冻结股票池快照：`{snapshot_id}`
- 候选哈希：`{manifest}`
- 上市满60日、收盘接近涨停事件：{events['near_limit_events']} 条
- 强势收盘事件：{events['strong_close_events']} 条
- 训练冠军：`{training_winner['name']}`
- 训练年化：{base_train['performance']['annual_return']:.2%} -> {winner_train['performance']['annual_return']:.2%}
- 训练夏普：{base_train['performance']['sharpe']:.3f} -> {winner_train['performance']['sharpe']:.3f}
- 训练年化换手：{base_train['activity']['annual_turnover']:.2f}x -> {winner_train['activity']['annual_turnover']:.2f}x
- 复用验证诊断：{'通过历史检查但不可晋级' if reused_validation_passed else '未通过'}
- 生产修改：否

## 口径

- 仅使用日线涨跌幅、收盘位置和换手率确认强势收盘。
- 信号在收盘后生成，下一交易日开盘执行，不是假设尾盘一定成交。
- 剔除缓存内上市前60个交易日，避免IPO首日涨幅污染。
- 2021-2023已经被查看过，只能作为复用诊断；2024以后不测试新候选。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "manifest": manifest,
        "diagnostics": diagnostics,
        "baseline_training": base_train,
        "training_winner": training_winner,
        "reused_validation_checks": payload["reused_validation_checks"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
