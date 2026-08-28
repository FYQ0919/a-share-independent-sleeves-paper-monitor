from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from app.backtest_audit import (
    execution_timing_audit,
    feature_prefix_invariance,
    future_perturbation_invariance,
    split_isolation_audit,
    strict_gate_audit,
)
from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR
from analyze_strategy import period_summary
from optimize_factor_strategy import (
    END,
    MODELS,
    START,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    load_current_history,
)


PROTOCOL_PATH = BASE_DIR / "data" / "research_protocol.json"
OUTPUT_JSON = BASE_DIR / "reports" / "no_lookahead_overfit_audit.json"
OUTPUT_MD = BASE_DIR / "reports" / "no_lookahead_overfit_audit.md"


def research_models() -> dict[str, dict[str, float]]:
    return {
        name: weights
        for name, weights in MODELS.items()
        if name == "current_contrarian" or name.startswith("contrarian_")
    }


def manifest_hash(models: dict[str, dict[str, float]]) -> str:
    canonical = json.dumps(models, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    models = research_models()
    actual_manifest = manifest_hash(models)
    manifest_passed = actual_manifest == protocol["candidate_manifest_sha256"]
    if not manifest_passed:
        raise RuntimeError(
            "Candidate manifest changed after holdout reveal. Review and explicitly version the research protocol."
        )

    optimization = json.loads(
        (BASE_DIR / "data" / "factor_optimization.json").read_text(encoding="utf-8")
    )
    selected_name = optimization["selected"]["name"]
    weights = models[selected_name]
    history, requested_codes, warnings = load_current_history()
    loaded_codes = sorted(history["code"].unique().tolist())
    available_dates = pd.DatetimeIndex(sorted(pd.to_datetime(history["date"]).unique()))
    cutoff = available_dates[available_dates <= pd.Timestamp(VALIDATION_END)][-1].date()

    config = BacktestConfig(
        mode="live",
        start_date=START,
        end_date=END,
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
        universe_policy="current_snapshot",
        price_adjustment_policy="current_vintage_qfq",
    )
    engine = BacktestEngine()
    engine._factor_weights_override = weights
    result = engine.run(history, config, warnings)
    checks = {
        "feature_prefix_invariance": feature_prefix_invariance(BacktestEngine(), history, cutoff),
        "future_perturbation_invariance": future_perturbation_invariance(
            BacktestEngine(), history, cutoff, weights
        ),
        "execution_timing": execution_timing_audit(result),
        "split_isolation": split_isolation_audit(
            result,
            [
                ("training", START, TRAIN_END),
                ("validation", VALIDATION_START, VALIDATION_END),
                ("test", TEST_START, END),
            ],
        ),
        "strict_point_in_time_gate": strict_gate_audit(history, config),
        "candidate_manifest_lock": {
            "passed": manifest_passed,
            "candidate_count": len(models),
            "expected_sha256": protocol["candidate_manifest_sha256"],
            "actual_sha256": actual_manifest,
        },
    }
    causal_checks = [
        checks["feature_prefix_invariance"]["passed"],
        checks["future_perturbation_invariance"]["passed"],
        checks["execution_timing"]["passed"],
        checks["split_isolation"]["passed"],
        checks["candidate_manifest_lock"]["passed"],
    ]
    strict_certified = all(causal_checks) and checks["strict_point_in_time_gate"]["passed"]
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "status": "certified" if strict_certified else "blocked",
        "strict_no_lookahead_certified": strict_certified,
        "selected_model": selected_name,
        "weights": weights,
        "universe": {"requested": len(requested_codes), "loaded": len(loaded_codes)},
        "checks": checks,
        "corrected_period_metrics": {
            "training": period_summary(result, START, TRAIN_END),
            "validation": period_summary(result, VALIDATION_START, VALIDATION_END),
            "test": period_summary(result, TEST_START, END),
        },
        "overfit_controls": {
            "candidate_manifest_frozen": True,
            "selection_uses_test": False,
            "holdout_status": protocol["holdout"]["status"],
            "holdout_may_be_used_for_reselection": protocol["holdout"]["may_be_used_for_reselection"],
            "forward_paper_start": protocol["forward_paper"]["start"],
            "minimum_forward_sessions": protocol["forward_paper"]["minimum_sessions"],
            "promotion_allowed_now": strict_certified,
        },
        "blocking_reasons": [
            "Current Top50 constituents were backfilled through history instead of reconstructed point in time.",
            "BaoStock cache uses current-vintage forward-adjusted prices without an as-of adjustment vintage.",
            "The 2024-2026 holdout has been revealed and is now consumed; it cannot be reused for tuning.",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    check_lines = "\n".join(
        f"- {'PASS' if value['passed'] else 'BLOCK'}: `{name}`"
        for name, value in checks.items()
    )
    test_metrics = payload["corrected_period_metrics"]["test"]
    markdown = f"""# 未来函数与过拟合审计

- 总结论：**{'CERTIFIED' if strict_certified else 'BLOCKED'}**
- 因子组合：`{selected_name}`
- 因子前缀不变性截止日：`{cutoff.isoformat()}`

## 自动检查

{check_lines}

## 修正后留出期指标

- 累计收益：{test_metrics['total_return']:.2%}
- 年化收益：{test_metrics['annual_return']:.2%}
- 最大回撤：{test_metrics['max_drawdown']:.2%}
- 夏普：{test_metrics['sharpe']:.3f}

## 严格认证阻断项

- 当前 Top50 被回填到历史，不是历史时点成分股池。
- 当前缓存是今日版本的前复权价，没有 as-of 复权版本。
- 2024-2026 留出集已经揭盲，不得再用于调因子或参数。
- 下一个可用的真正前向验收窗口从 `2026-08-26` 开始，至少积累 126 个交易日。
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "json": str(OUTPUT_JSON),
        "markdown": str(OUTPUT_MD),
        "checks": {name: value["passed"] for name, value in checks.items()},
        "test_metrics": test_metrics,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
