from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR
from analyze_strategy import period_summary, summarize_result
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import (
    END,
    START,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    load_current_history,
)


OUTPUT_PATH = BASE_DIR / "data" / "training_factor_optimization_v2.json"
REPORT_PATH = BASE_DIR / "reports" / "training_factor_optimization_v2.md"
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

# This neighborhood is declared before evaluation and never uses 2024+ results.
CANDIDATES = {
    BASELINE_NAME: BASELINE_WEIGHTS,
    "q10_low_vol_03": {
        "medium_reversal": 0.34,
        "inefficiency": 0.24,
        "liquidity_cooling": 0.19,
        "low_volatility": 0.03,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_low_vol_08": {
        "medium_reversal": 0.32,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.18,
        "low_volatility": 0.08,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_downside_03": {
        "medium_reversal": 0.32,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.18,
        "low_volatility": 0.05,
        "downside_quality": 0.03,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_price_position_03": {
        "medium_reversal": 0.32,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.18,
        "low_volatility": 0.05,
        "price_position": 0.03,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_value_13": {
        "medium_reversal": 0.32,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.18,
        "low_volatility": 0.05,
        "risk": 0.10,
        "value": 0.13,
    },
    "q10_defensive_mix": {
        "medium_reversal": 0.31,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.17,
        "low_volatility": 0.07,
        "downside_quality": 0.03,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_stable_mix": {
        "medium_reversal": 0.31,
        "inefficiency": 0.21,
        "liquidity_cooling": 0.17,
        "low_volatility": 0.06,
        "downside_quality": 0.03,
        "price_position": 0.02,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_reversal_soft": {
        "medium_reversal": 0.30,
        "inefficiency": 0.25,
        "liquidity_cooling": 0.18,
        "low_volatility": 0.07,
        "risk": 0.10,
        "value": 0.10,
    },
    "q10_liquidity_soft": {
        "medium_reversal": 0.34,
        "inefficiency": 0.24,
        "liquidity_cooling": 0.12,
        "low_volatility": 0.08,
        "risk": 0.10,
        "value": 0.12,
    },
    "q10_value_defensive": {
        "medium_reversal": 0.30,
        "inefficiency": 0.21,
        "liquidity_cooling": 0.16,
        "low_volatility": 0.08,
        "downside_quality": 0.03,
        "risk": 0.08,
        "value": 0.14,
    },
    "q10_price_value": {
        "medium_reversal": 0.31,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.17,
        "low_volatility": 0.05,
        "price_position": 0.03,
        "risk": 0.09,
        "value": 0.13,
    },
    "q10_downside_price": {
        "medium_reversal": 0.31,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.17,
        "low_volatility": 0.06,
        "downside_quality": 0.02,
        "price_position": 0.02,
        "risk": 0.10,
        "value": 0.10,
    },
}

TRAINING_FOLDS = {
    "2016_2017": (START, date(2017, 12, 31)),
    "2018_stress": (date(2018, 1, 1), date(2018, 12, 31)),
    "2019_recovery": (date(2019, 1, 1), date(2019, 12, 31)),
    "2020_pandemic": (date(2020, 1, 1), TRAIN_END),
}


def validate_candidates() -> str:
    for name, weights in CANDIDATES.items():
        if not np.isclose(sum(weights.values()), 1.0):
            raise RuntimeError(f"{name} 权重和不是1")
        if any(value < 0 for value in weights.values()):
            raise RuntimeError(f"{name} 包含负权重")
    canonical = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def install_score_cache(engine: BacktestEngine) -> dict[str, pd.DataFrame]:
    original_score = engine._score
    cache: dict[str, pd.DataFrame] = {}

    def cached_score(cross: pd.DataFrame, factor_weights=None):
        key = pd.Timestamp(cross["date"].iloc[0]).date().isoformat()
        if key not in cache:
            cache[key] = original_score(cross.copy(), {"risk": 1.0})
        scored = cache[key].copy()
        weights = factor_weights or engine.FACTOR_WEIGHTS
        scored["score"] = sum(
            scored[f"{factor}_score"] * weight
            for factor, weight in weights.items()
        ) * 100
        return scored

    engine._score = cached_score
    return cache


def training_summary(result: dict) -> dict:
    aggregate = period_summary(result, START, TRAIN_END)
    folds = {
        label: period_summary(result, start, end)
        for label, (start, end) in TRAINING_FOLDS.items()
    }
    fold_sharpes = [item["sharpe"] for item in folds.values()]
    fold_returns = [item["annual_return"] for item in folds.values()]
    return {
        "aggregate": aggregate,
        "folds": folds,
        "minimum_fold_sharpe": min(fold_sharpes),
        "median_fold_sharpe": float(np.median(fold_sharpes)),
        "minimum_fold_annual_return": min(fold_returns),
        "positive_fold_count": sum(value > 0 for value in fold_returns),
    }


def training_rank(row: dict) -> tuple:
    summary = row["training"]
    aggregate = summary["aggregate"]
    return (
        summary["minimum_fold_sharpe"],
        summary["median_fold_sharpe"],
        aggregate["sharpe"],
        aggregate["annual_return"],
        aggregate["max_drawdown"],
    )


def validation_gate(candidate: dict, baseline: dict) -> dict:
    candidate_metrics = candidate["validation"]
    baseline_metrics = baseline["validation"]
    checks = {
        "annual_return_not_worse_by_1pp": (
            candidate_metrics["annual_return"] >= baseline_metrics["annual_return"] - 0.01
        ),
        "sharpe_not_worse_by_0_05": (
            candidate_metrics["sharpe"] >= baseline_metrics["sharpe"] - 0.05
        ),
        "drawdown_not_worse_by_3pp": (
            candidate_metrics["max_drawdown"] >= baseline_metrics["max_drawdown"] - 0.03
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    manifest = validate_candidates()
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    candidates, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(candidates)
    loaded_codes = sorted(history["code"].unique().tolist())

    engine = BacktestEngine()
    feature_frame = engine._features(history)
    engine._features = lambda _: feature_frame
    score_cache = install_score_cache(engine)
    base_config = BacktestConfig(
        mode="live",
        start_date=START,
        end_date=VALIDATION_END,
        codes=loaded_codes,
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

    rows = []
    for name, weights in CANDIDATES.items():
        engine._factor_weights_override = weights
        result = engine.run(history, base_config, warnings)
        rows.append({
            "name": name,
            "weights": weights,
            "training": training_summary(result),
            "validation": period_summary(result, VALIDATION_START, VALIDATION_END),
        })

    baseline = next(row for row in rows if row["name"] == BASELINE_NAME)
    baseline_train = baseline["training"]["aggregate"]
    training_improvers = [
        row
        for row in rows
        if row["training"]["aggregate"]["annual_return"] > baseline_train["annual_return"]
        and row["training"]["aggregate"]["sharpe"] > baseline_train["sharpe"]
        and row["training"]["aggregate"]["max_drawdown"] >= baseline_train["max_drawdown"] - 0.03
    ]
    training_winner = max(training_improvers, key=training_rank) if training_improvers else baseline
    gate = validation_gate(training_winner, baseline)
    accepted_for_forward_research = gate["passed"] and training_winner["name"] != BASELINE_NAME

    selected_name = training_winner["name"] if accepted_for_forward_research else BASELINE_NAME
    selected_weights = CANDIDATES[selected_name]
    engine._factor_weights_override = selected_weights
    full_result = engine.run(history, replace(base_config, end_date=END), warnings)
    diagnostics = {
        "training": period_summary(full_result, START, TRAIN_END),
        "validation": period_summary(full_result, VALIDATION_START, VALIDATION_END),
        "revealed_test_not_used_for_selection": period_summary(full_result, TEST_START, END),
        "full": summarize_result(full_result, selected_name),
    }

    rows.sort(key=training_rank, reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "objective": "Improve both annual return and Sharpe on 2016-2020 without changing execution",
        "windows": {
            "training": [START.isoformat(), TRAIN_END.isoformat()],
            "validation_once": [VALIDATION_START.isoformat(), VALIDATION_END.isoformat()],
            "revealed_test_diagnostic_only": [TEST_START.isoformat(), END.isoformat()],
        },
        "execution": "收盘信号，下一交易日开盘生效，单边成本12bp，自适应Top5参数不变",
        "universe_snapshot_run_id": snapshot_id,
        "candidate_manifest_sha256": manifest,
        "candidate_count": len(CANDIDATES),
        "selection_rule": (
            "Candidates must improve aggregate training annual return and Sharpe; rank only by "
            "2016-2020 fold robustness. Validation is a one-time rejection gate, never a reranking key."
        ),
        "baseline": baseline,
        "training_winner": training_winner,
        "validation_gate": gate,
        "accepted_for_forward_research": accepted_for_forward_research,
        "selected_for_diagnostics": selected_name,
        "diagnostics": diagnostics,
        "production_change": False,
        "promotion_requirements": [
            "不得使用已揭示的2024-2026结果重新选择候选或权重",
            "必须从下一交易日起冻结模型并累计至少126个前向模拟盘交易日",
            "必须补齐逐日历史成分股与as-of价格后才能通过严格无前视门禁",
        ],
        "score_cache_dates": len(score_cache),
        "candidates": rows,
        "warnings": warnings + [
            "使用冻结的当前Top50历史回填，仍存在成分、上市和存续偏差",
            "2024-2026已揭示，仅在训练期候选确定并通过验证门禁后做一次诊断",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    winner_train = training_winner["training"]["aggregate"]
    winner_validation = training_winner["validation"]
    REPORT_PATH.write_text(
        f"""# 2016-2020 因子优化 v2

- 状态：研究结果，**未修改生产因子**
- 冻结股票池快照：`{snapshot_id}`
- 候选数：{len(CANDIDATES)}
- 训练期优胜候选：`{training_winner['name']}`
- 训练期年化：{baseline_train['annual_return']:.2%} -> {winner_train['annual_return']:.2%}
- 训练期夏普：{baseline_train['sharpe']:.3f} -> {winner_train['sharpe']:.3f}
- 训练期最大回撤：{baseline_train['max_drawdown']:.2%} -> {winner_train['max_drawdown']:.2%}
- 一次性验证年化：{baseline['validation']['annual_return']:.2%} -> {winner_validation['annual_return']:.2%}
- 一次性验证夏普：{baseline['validation']['sharpe']:.3f} -> {winner_validation['sharpe']:.3f}
- 验证门禁：{'通过' if gate['passed'] else '未通过'}
- 前向研究候选：{'是' if accepted_for_forward_research else '否'}

## 约束

""" + "\n".join(f"- {item}" for item in payload["promotion_requirements"]),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "baseline_training": baseline_train,
        "training_winner": {
            "name": training_winner["name"],
            "weights": training_winner["weights"],
            "training": winner_train,
            "validation": winner_validation,
        },
        "validation_gate": gate,
        "accepted_for_forward_research": accepted_for_forward_research,
        "diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
