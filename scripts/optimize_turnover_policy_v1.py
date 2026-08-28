from __future__ import annotations

from datetime import date, datetime
import hashlib
import json

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


OUTPUT_PATH = BASE_DIR / "data" / "turnover_policy_optimization_v1.json"
REPORT_PATH = BASE_DIR / "reports" / "turnover_policy_optimization_v1.md"
SOURCE_OPTIMIZATION = BASE_DIR / "data" / "factor_optimization.json"

BASELINE_NAME = "d10_h5_b20_g15"
FACTOR_NAME = "contrarian_quality_10"
FACTOR_WEIGHTS = {
    "medium_reversal": 0.33,
    "inefficiency": 0.23,
    "liquidity_cooling": 0.19,
    "low_volatility": 0.05,
    "risk": 0.10,
    "value": 0.10,
}

# Frozen before evaluation. Every candidate keeps Top5, next-open execution,
# 12 bps one-way costs, and at most one ordinary adaptive replacement per cycle.
CANDIDATES = {
    BASELINE_NAME: {"rebalance_days": 10, "min_hold_days": 5, "rank_buffer": 20, "score_gap": 15.0},
    "d10_h10_b20_g20": {"rebalance_days": 10, "min_hold_days": 10, "rank_buffer": 20, "score_gap": 20.0},
    "d10_h10_b25_g20": {"rebalance_days": 10, "min_hold_days": 10, "rank_buffer": 25, "score_gap": 20.0},
    "d10_h10_b30_g25": {"rebalance_days": 10, "min_hold_days": 10, "rank_buffer": 30, "score_gap": 25.0},
    "d15_h5_b20_g15": {"rebalance_days": 15, "min_hold_days": 5, "rank_buffer": 20, "score_gap": 15.0},
    "d15_h10_b20_g20": {"rebalance_days": 15, "min_hold_days": 10, "rank_buffer": 20, "score_gap": 20.0},
    "d15_h10_b25_g20": {"rebalance_days": 15, "min_hold_days": 10, "rank_buffer": 25, "score_gap": 20.0},
    "d15_h15_b25_g25": {"rebalance_days": 15, "min_hold_days": 15, "rank_buffer": 25, "score_gap": 25.0},
    "d20_h5_b20_g15": {"rebalance_days": 20, "min_hold_days": 5, "rank_buffer": 20, "score_gap": 15.0},
    "d20_h10_b20_g20": {"rebalance_days": 20, "min_hold_days": 10, "rank_buffer": 20, "score_gap": 20.0},
    "d20_h15_b25_g25": {"rebalance_days": 20, "min_hold_days": 15, "rank_buffer": 25, "score_gap": 25.0},
    "d25_h10_b20_g20": {"rebalance_days": 25, "min_hold_days": 10, "rank_buffer": 20, "score_gap": 20.0},
    "d25_h15_b25_g25": {"rebalance_days": 25, "min_hold_days": 15, "rank_buffer": 25, "score_gap": 25.0},
    "d30_h10_b20_g20": {"rebalance_days": 30, "min_hold_days": 10, "rank_buffer": 20, "score_gap": 20.0},
    "d30_h15_b25_g25": {"rebalance_days": 30, "min_hold_days": 15, "rank_buffer": 25, "score_gap": 25.0},
}

TRAINING_FOLDS = {
    "2016_2017": (START, date(2017, 12, 31)),
    "2018_stress": (date(2018, 1, 1), date(2018, 12, 31)),
    "2019_recovery": (date(2019, 1, 1), date(2019, 12, 31)),
    "2020_pandemic": (date(2020, 1, 1), TRAIN_END),
}


def validate_candidates() -> str:
    required = {"rebalance_days", "min_hold_days", "rank_buffer", "score_gap"}
    for name, policy in CANDIDATES.items():
        if set(policy) != required:
            raise RuntimeError(f"{name} 参数不完整")
        if policy["rebalance_days"] < 10 or policy["min_hold_days"] < 5:
            raise RuntimeError(f"{name} 不是降换手候选")
        if policy["rank_buffer"] < 20 or policy["score_gap"] < 15:
            raise RuntimeError(f"{name} 放松了换仓门槛")
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


def period_activity(result: dict, start: date, end: date) -> dict:
    rows = [
        row for row in result["return_periods"]
        if start <= date.fromisoformat(row["period_end"]) <= end
    ]
    if not rows:
        raise ValueError("区间内没有换手记录")
    first = date.fromisoformat(rows[0]["period_start"])
    last = date.fromisoformat(rows[-1]["period_end"])
    years = max((last - first).days / 365.2425, 1 / 365.2425)
    turnovers = [float(row["turnover"]) for row in rows]
    costs = [float(row["cost"]) for row in rows]
    event_turnovers = [value for value in turnovers if value > 1e-12]
    return {
        "total_turnover": float(sum(turnovers)),
        "annual_turnover": float(sum(turnovers) / years),
        "rebalance_count": len(event_turnovers),
        "average_event_turnover": float(np.mean(event_turnovers)) if event_turnovers else 0.0,
        "total_cost_return_units": float(sum(costs)),
    }


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
    fold_returns = [item["performance"]["annual_return"] for item in folds.values()]
    return {
        "aggregate": aggregate,
        "folds": folds,
        "minimum_fold_sharpe": min(fold_sharpes),
        "median_fold_sharpe": float(np.median(fold_sharpes)),
        "positive_fold_count": sum(value > 0 for value in fold_returns),
    }


def training_rank(row: dict) -> tuple:
    training = row["training"]
    aggregate = training["aggregate"]
    performance = aggregate["performance"]
    activity = aggregate["activity"]
    return (
        training["minimum_fold_sharpe"],
        training["median_fold_sharpe"],
        performance["annual_return"],
        performance["sharpe"],
        -activity["annual_turnover"],
        performance["max_drawdown"],
    )


def is_training_improver(candidate: dict, baseline: dict) -> bool:
    candidate_perf = candidate["training"]["aggregate"]["performance"]
    candidate_activity = candidate["training"]["aggregate"]["activity"]
    baseline_perf = baseline["training"]["aggregate"]["performance"]
    baseline_activity = baseline["training"]["aggregate"]["activity"]
    return (
        candidate_perf["annual_return"] > baseline_perf["annual_return"]
        and candidate_perf["sharpe"] > baseline_perf["sharpe"]
        and candidate_perf["max_drawdown"] >= baseline_perf["max_drawdown"] - 0.03
        and candidate_activity["annual_turnover"] <= baseline_activity["annual_turnover"] * 0.95
    )


def validation_gate(candidate: dict, baseline: dict) -> dict:
    candidate_perf = candidate["validation"]["performance"]
    candidate_activity = candidate["validation"]["activity"]
    baseline_perf = baseline["validation"]["performance"]
    baseline_activity = baseline["validation"]["activity"]
    checks = {
        "annual_return_improved": candidate_perf["annual_return"] > baseline_perf["annual_return"],
        "sharpe_not_worse_by_0_05": candidate_perf["sharpe"] >= baseline_perf["sharpe"] - 0.05,
        "drawdown_not_worse_by_3pp": candidate_perf["max_drawdown"] >= baseline_perf["max_drawdown"] - 0.03,
        "annual_turnover_lower": candidate_activity["annual_turnover"] < baseline_activity["annual_turnover"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def make_config(policy: dict, codes: list[str], end_date: date) -> BacktestConfig:
    return BacktestConfig(
        mode="live",
        start_date=START,
        end_date=end_date,
        codes=codes,
        top_n=5,
        rebalance_days=policy["rebalance_days"],
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
        rebalance_policy="adaptive",
        min_hold_days=policy["min_hold_days"],
        rank_buffer=policy["rank_buffer"],
        score_gap=policy["score_gap"],
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

    engine = BacktestEngine()
    feature_frame = engine._features(history)
    engine._features = lambda _: feature_frame
    engine._factor_weights_override = FACTOR_WEIGHTS
    score_cache = install_score_cache(engine)

    rows = []
    for name, policy in CANDIDATES.items():
        result = engine.run(history, make_config(policy, loaded_codes, TRAIN_END), warnings)
        rows.append({
            "name": name,
            "policy": policy,
            "training": training_summary(result),
        })

    baseline = next(row for row in rows if row["name"] == BASELINE_NAME)
    training_improvers = [row for row in rows if is_training_improver(row, baseline)]
    training_winner = max(training_improvers, key=training_rank) if training_improvers else baseline

    validation_targets = {BASELINE_NAME, training_winner["name"]}
    for row in rows:
        if row["name"] not in validation_targets:
            continue
        result = engine.run(
            history,
            make_config(row["policy"], loaded_codes, VALIDATION_END),
            warnings,
        )
        row["validation"] = period_result(result, VALIDATION_START, VALIDATION_END)

    gate = validation_gate(training_winner, baseline)
    accepted_for_forward_research = gate["passed"] and training_winner["name"] != BASELINE_NAME

    selected_name = training_winner["name"] if accepted_for_forward_research else BASELINE_NAME
    selected_policy = CANDIDATES[selected_name]
    full_result = engine.run(history, make_config(selected_policy, loaded_codes, END), warnings)
    diagnostics = {
        "training": period_result(full_result, START, TRAIN_END),
        "validation": period_result(full_result, VALIDATION_START, VALIDATION_END),
        "revealed_test_not_used_for_selection": period_result(full_result, TEST_START, END),
        "full": {
            "performance": summarize_result(full_result, selected_name),
            "activity": period_activity(full_result, START, END),
        },
    }

    rows.sort(key=training_rank, reverse=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "objective": "Lower annual turnover and improve net annual return without changing factor weights",
        "windows": {
            "training": [START.isoformat(), TRAIN_END.isoformat()],
            "validation_once": [VALIDATION_START.isoformat(), VALIDATION_END.isoformat()],
            "revealed_test_diagnostic_only": [TEST_START.isoformat(), END.isoformat()],
        },
        "factor": {"name": FACTOR_NAME, "weights": FACTOR_WEIGHTS},
        "execution": "收盘信号，下一交易日开盘生效，单边成本12bp，自适应Top5",
        "universe_snapshot_run_id": snapshot_id,
        "candidate_manifest_sha256": manifest,
        "candidate_count": len(CANDIDATES),
        "selection_rule": (
            "Training candidates must improve net annual return and Sharpe, keep drawdown within 3pp, "
            "and lower annual turnover by at least 5%. Rank only by 2016-2020 fold robustness. "
            "Validation is a one-time rejection gate and is never used to rerank candidates."
        ),
        "baseline": baseline,
        "training_winner": training_winner,
        "validation_gate": gate,
        "accepted_for_forward_research": accepted_for_forward_research,
        "selected_for_diagnostics": selected_name,
        "diagnostics": diagnostics,
        "production_change": False,
        "promotion_requirements": [
            "不得使用已揭示的2024-2026结果重新选择调仓参数",
            "从下一交易日起冻结候选并累计至少126个前向模拟盘交易日",
            "补齐逐日历史成分股与as-of价格后再通过严格无前视门禁",
        ],
        "score_cache_dates": len(score_cache),
        "candidates": rows,
        "warnings": warnings + [
            "使用冻结的当前Top50历史回填，仍存在成分、上市和存续偏差",
            "2024-2026已揭示，仅在训练冠军通过验证门禁后做诊断",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_train = baseline["training"]["aggregate"]
    winner_train = training_winner["training"]["aggregate"]
    baseline_validation = baseline["validation"]
    winner_validation = training_winner["validation"]
    REPORT_PATH.write_text(
        f"""# 降换手执行策略优化 v1

- 状态：研究结果，**未修改生产调仓参数**
- 因子：`{FACTOR_NAME}`，权重不变
- 冻结股票池快照：`{snapshot_id}`
- 候选数：{len(CANDIDATES)}
- 训练期优胜候选：`{training_winner['name']}`
- 参数：`{json.dumps(training_winner['policy'], ensure_ascii=False)}`
- 训练期年化：{baseline_train['performance']['annual_return']:.2%} -> {winner_train['performance']['annual_return']:.2%}
- 训练期夏普：{baseline_train['performance']['sharpe']:.3f} -> {winner_train['performance']['sharpe']:.3f}
- 训练期年化换手：{baseline_train['activity']['annual_turnover']:.2f}x -> {winner_train['activity']['annual_turnover']:.2f}x
- 验证期年化：{baseline_validation['performance']['annual_return']:.2%} -> {winner_validation['performance']['annual_return']:.2%}
- 验证期夏普：{baseline_validation['performance']['sharpe']:.3f} -> {winner_validation['performance']['sharpe']:.3f}
- 验证期年化换手：{baseline_validation['activity']['annual_turnover']:.2f}x -> {winner_validation['activity']['annual_turnover']:.2f}x
- 验证门禁：{'通过' if gate['passed'] else '未通过'}
- 前向研究候选：{'是' if accepted_for_forward_research else '否'}

## 约束

""" + "\n".join(f"- {item}" for item in payload["promotion_requirements"]),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "report": str(REPORT_PATH),
        "manifest": manifest,
        "baseline_training": baseline_train,
        "training_winner": {
            "name": training_winner["name"],
            "policy": training_winner["policy"],
            "training": winner_train,
            "validation": winner_validation,
        },
        "validation_gate": gate,
        "accepted_for_forward_research": accepted_for_forward_research,
        "selected_for_diagnostics": selected_name,
        "diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
