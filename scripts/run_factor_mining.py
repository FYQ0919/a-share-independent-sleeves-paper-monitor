from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR
from app.factor_generation import (
    apply_factor_expressions,
    generate_factor_expressions,
    update_factor_registry,
)
from app.research_factors import add_research_factors
from app.factor_mining import (
    FactorMiningConfig,
    MiningWindow,
    build_factor_catalog,
    correlation_edges,
    daily_rank_ic,
    discovery_feature_correlation,
    expanding_factor_scores,
    factor_diagnostics,
    select_stable_nonredundant_factors,
)
from app.lgbm_features import build_alpha158_lite
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_qlib_ridge_topk_v1 import (
    baseline_scores,
    research_summary,
    topk_dropout_backtest,
)


SOURCE_OPTIMIZATION = BASE_DIR / "data" / "factor_optimization.json"
DATA_DIR = BASE_DIR / "data" / "factor_mining"
REPORT_DIR = BASE_DIR / "reports" / "factor_mining"
OUTPUT_PATH = DATA_DIR / "latest.json"
DIAGNOSTICS_PATH = REPORT_DIR / "latest_factor_diagnostics.csv"
IC_PATH = REPORT_DIR / "latest_daily_rank_ic.csv"
CORRELATION_PATH = REPORT_DIR / "latest_discovery_correlation.csv"
REPORT_PATH = REPORT_DIR / "latest.md"
REGISTRY_PATH = DATA_DIR / "registry.json"


CONFIG = FactorMiningConfig(
    discovery=MiningWindow("discovery", date(2016, 8, 25), date(2017, 12, 31), "factor discovery"),
    selection=MiningWindow("selection", date(2018, 1, 1), date(2019, 12, 31), "candidate selection"),
    diagnostic=MiningWindow("diagnostic", date(2020, 1, 1), date(2026, 8, 25), "revealed diagnostic only"),
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _score_frame(frame: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    columns = ["date", "code", "tradestatus", "is_st", "close", "amount_20"]
    if "universe_member" in frame.columns:
        columns.append("universe_member")
    result = frame[columns].copy()
    result["score"] = score.reindex(result.index)
    invalid = (
        ~result["tradestatus"].eq(1)
        | ~result["is_st"].eq(0)
        | ~result["close"].gt(1)
        | ~result["amount_20"].ge(10_000_000)
    )
    if "universe_member" in result.columns:
        invalid |= ~result["universe_member"].fillna(False).astype(bool)
    result.loc[invalid, "score"] = np.nan
    return result


def _performance_pair(
    history: pd.DataFrame,
    factor_scores: pd.DataFrame,
    baseline_score_frame: pd.DataFrame,
    start: date,
    end: date,
) -> dict:
    candidate = topk_dropout_backtest(history, factor_scores, start, end)
    baseline = topk_dropout_backtest(history, baseline_score_frame, start, end)
    return {
        "factor_ensemble": research_summary(candidate, start, end),
        "current_rule_baseline": research_summary(baseline, start, end),
    }


def _candidate_gate(selection: dict, selected_count: int, maturity_audit: list[dict]) -> dict:
    candidate = selection["factor_ensemble"]
    baseline = selection["current_rule_baseline"]
    candidate_perf = candidate["performance"]
    baseline_perf = baseline["performance"]
    checks = {
        "at_least_five_stable_factors": selected_count >= 5,
        "all_monthly_labels_strictly_mature": bool(maturity_audit)
        and all(row["strictly_mature"] for row in maturity_audit),
        "selection_annual_return_retains_90pct": candidate_perf["annual_return"]
        >= baseline_perf["annual_return"] * 0.90,
        "selection_sharpe_not_lower": candidate_perf["sharpe"] >= baseline_perf["sharpe"],
        "selection_drawdown_not_worse_by_3pp": candidate_perf["max_drawdown"]
        >= baseline_perf["max_drawdown"] - 0.03,
        "selection_turnover_not_higher_by_10pct": candidate["activity"]["annual_turnover"]
        <= baseline["activity"]["annual_turnover"] * 1.10,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _revealed_diagnostic_review(diagnostic: dict) -> dict:
    candidate = diagnostic["factor_ensemble"]["performance"]
    baseline = diagnostic["current_rule_baseline"]["performance"]
    checks = {
        "annual_return_retains_90pct": candidate["annual_return"]
        >= baseline["annual_return"] * 0.90,
        "sharpe_not_lower": candidate["sharpe"] >= baseline["sharpe"],
        "drawdown_not_worse_by_3pp": candidate["max_drawdown"]
        >= baseline["max_drawdown"] - 0.03,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_effect": "deployment_blocker_only; never used to rank, weight, or replace factors",
    }


def _manifest(config: FactorMiningConfig, feature_columns: list[str]) -> str:
    canonical = json.dumps(
        {"config": config.to_dict(), "features": feature_columns},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}%"


def _number(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def _write_report(payload: dict) -> None:
    selection = payload["portfolio_evaluation"]["selection"]
    diagnostic = payload["portfolio_evaluation"]["revealed_diagnostic"]
    selection_window = payload["config"]["selection"]
    diagnostic_window = payload["config"]["diagnostic"]
    selection_label = f"{selection_window['start']} 至 {selection_window['end']} 选择窗口"
    diagnostic_label = f"{diagnostic_window['start']} 至 {diagnostic_window['end']} 隔离窗口"
    selected_rows = [row for row in payload["factor_diagnostics"] if row["selected"]]
    catalog_lookup = {row["feature"]: row for row in payload["factor_catalog"]}
    selected_framework_rows = [
        catalog_lookup[name]
        for name in payload.get("selected_framework_features", [])
        if name in catalog_lookup
    ]
    lines = [
        "# 多因子挖掘报告",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 从 {payload['feature_count']} 个候选因子中选出 {payload['selected_factor_count']} 个稳定、低冗余因子。",
        f"- 复现 Qlib、经典微观结构和时序特征框架思路的因子 {payload['framework_factor_count']} 个，其中 {payload['selected_framework_factor_count']} 个入选。",
        f"- 本轮自动生成 {payload['generated_factor_count']} 个结构化新因子，其中 {payload['selected_generated_factor_count']} 个入选。",
        f"- 因子注册表版本：`{payload['registry_version']}`；本轮首次发现 {len(payload['new_generated_factors'])} 个公式。",
        f"- {selection_label}门禁：{'通过' if payload['candidate_gate']['passed'] else '未通过'}。",
        f"- {diagnostic_label}：{'通过' if payload['revealed_diagnostic_review']['passed'] else '失败，否决部署'}。",
        "- 门禁只使用发现窗口与选择窗口；隔离窗口只做部署否决，不参与筛选。",
        "- 生产模型、模拟盘和推送配置均未修改。",
        "",
        "## 执行口径",
        "",
        "收盘生成排名，下一交易日开盘执行；Top5；每10个交易日调仓；单边成本12bp。",
        "",
        "## 组合表现",
        "",
        "| 区间 | 策略 | 年化收益 | 最大回撤 | 夏普 | 年换手 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, block in ((selection_label, selection), (diagnostic_label, diagnostic)):
        for name, display in (("factor_ensemble", "因子组合"), ("current_rule_baseline", "当前规则基线")):
            row = block[name]
            perf = row["performance"]
            lines.append(
                f"| {label} | {display} | {_pct(perf['annual_return'])} | "
                f"{_pct(perf['max_drawdown'])} | {_number(perf['sharpe'])} | "
                f"{row['activity']['annual_turnover']:.2f}x |"
            )
    lines.extend(["", "## 入选因子", "", "| 因子 | 类别 | 发现IC | 选择IC | 诊断IC | 稳定分 |", "|---|---|---:|---:|---:|---:|"])
    for row in selected_rows:
        lines.append(
            f"| `{row['feature']}` | {row['family']} | {row['discovery_mean_rank_ic']:.4f} | "
            f"{row['selection_mean_rank_ic']:.4f} | {row['diagnostic_mean_rank_ic']:.4f} | "
            f"{row['stability_score']:.5f} |"
        )
    lines.extend(["", "## 入选的框架复现因子", "", "| 因子 | 参考来源 | 公式 |", "|---|---|---|"])
    for row in selected_framework_rows:
        lines.append(f"| `{row['feature']}` | {row['reference']} | `{row['formula']}` |")
    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- 股票池是当前 Top50 的历史回填，存在成分股与存续偏差。",
            "- 当前前复权价格不是严格 point-in-time 复权数据。",
            "- 估值字段没有逐字段可用日期证明，不能获得严格无前视认证。",
            "- 已揭盲诊断区间不得用于继续调阈值或重选因子。",
            "",
            f"详细数据：`{DIAGNOSTICS_PATH.relative_to(BASE_DIR)}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config: FactorMiningConfig = CONFIG,
    history: pd.DataFrame | None = None,
    requested_codes: list[str] | None = None,
    snapshot_run_id: str | None = None,
    input_warnings: list[str] | None = None,
    update_mode: str = "frozen_research",
) -> dict:
    warnings = list(input_warnings or [])
    if history is None:
        source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
        candidates, frozen_snapshot_id = frozen_research_candidates(source["generated_at"])
        history, loaded_requested_codes, load_warnings = load_current_history(candidates)
        requested_codes = loaded_requested_codes
        snapshot_run_id = frozen_snapshot_id
        warnings.extend(load_warnings)
    if requested_codes is None:
        requested_codes = sorted(history["code"].astype(str).unique().tolist())
    snapshot_run_id = snapshot_run_id or "unspecified"
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(config.diagnostic.end))].copy()
    frame, alpha158_feature_columns = build_alpha158_lite(history)
    frame, framework_feature_columns, framework_batches = add_research_factors(frame)
    base_feature_columns = alpha158_feature_columns + framework_feature_columns

    base_ic_rows = daily_rank_ic(frame, base_feature_columns, config.min_cross_section)
    base_diagnostics = factor_diagnostics(base_ic_rows, base_feature_columns, config)
    expressions = generate_factor_expressions(base_diagnostics)
    frame, generated_feature_columns = apply_factor_expressions(frame, expressions)
    feature_columns = base_feature_columns + generated_feature_columns

    ic_rows = daily_rank_ic(frame, feature_columns, config.min_cross_section)
    diagnostics = factor_diagnostics(ic_rows, feature_columns, config)
    correlation = discovery_feature_correlation(
        frame, feature_columns, config.discovery
    )
    diagnostics, selected = select_stable_nonredundant_factors(
        diagnostics, correlation, config
    )
    prediction_scores, maturity_audit = expanding_factor_scores(
        frame,
        ic_rows,
        selected,
        config.selection.start,
        config.diagnostic.end,
        config,
    )
    factor_score_frame = _score_frame(frame, prediction_scores)
    baseline_prediction = baseline_scores(frame, config.selection.start, config.diagnostic.end)
    baseline_score_frame = _score_frame(frame, baseline_prediction)
    selection_evaluation = _performance_pair(
        history,
        factor_score_frame,
        baseline_score_frame,
        config.selection.start,
        config.selection.end,
    )
    diagnostic_evaluation = _performance_pair(
        history,
        factor_score_frame,
        baseline_score_frame,
        config.diagnostic.start,
        config.diagnostic.end,
    )
    gate = _candidate_gate(selection_evaluation, len(selected), maturity_audit)
    diagnostic_review = _revealed_diagnostic_review(diagnostic_evaluation)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")
    ic_rows.to_csv(IC_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")
    correlation.to_csv(CORRELATION_PATH, encoding="utf-8-sig", float_format="%.8f")

    manifest = _manifest(config, feature_columns)
    try:
        previous_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        previous_registry = None
    generated_at = datetime.now().astimezone().isoformat()
    registry = _json_ready(update_factor_registry(
        previous_registry,
        expressions,
        diagnostics,
        selected,
        generated_at,
        manifest,
    ))
    temporary_registry = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_registry.replace(REGISTRY_PATH)

    generated_lookup = {row.name: row for row in expressions}
    factor_catalog = build_factor_catalog(base_feature_columns)
    factor_catalog.extend({
        "feature": row.name,
        "family": row.family,
        "point_in_time_contract": "signal-date causal factor inputs only",
        "enabled": True,
        "generated": True,
        "operator": row.operator,
        "formula": row.formula,
        "inputs": [row.left, row.right],
    } for row in expressions)
    selected_generated = [name for name in selected if name in generated_lookup]
    selected_framework = [name for name in selected if name in framework_feature_columns]

    payload = _json_ready(
        {
            "generated_at": generated_at,
            "status": (
                "selection_gate_passed_revealed_diagnostic_failed"
                if gate["passed"] and not diagnostic_review["passed"]
                else "candidate_passed_not_deployed"
                if gate["passed"]
                else "research_only_not_promoted"
            ),
            "system": "causal_multi_factor_mining_v3_framework_replication",
            "update_mode": update_mode,
            "manifest_sha256": manifest,
            "config": config.to_dict(),
            "label_contract": "signal close; next-open entry; t+11 open exit; 10-session return",
            "execution_contract": "Top5; next-open; 10-session rebalance; 12bp one-way cost",
            "selection_policy": "discovery IC -> independent selection stability gate -> discovery correlation deduplication",
            "diagnostic_policy": "latest quarantine is revealed and excluded from feature selection, weights, and promotion gate",
            "feature_count": len(feature_columns),
            "base_factor_count": len(base_feature_columns),
            "alpha158_factor_count": len(alpha158_feature_columns),
            "framework_factor_count": len(framework_feature_columns),
            "framework_factor_batches": {
                name: len(features) for name, features in framework_batches.items()
            },
            "selected_framework_factor_count": len(selected_framework),
            "selected_framework_features": selected_framework,
            "generated_factor_count": len(generated_feature_columns),
            "selected_generated_factor_count": len(selected_generated),
            "new_generated_factors": registry["new_factors"],
            "registry_version": registry["version"],
            "factor_catalog": factor_catalog,
            "selected_factor_count": len(selected),
            "selected_features": selected,
            "factor_diagnostics": diagnostics.to_dict(orient="records"),
            "correlation_edges": correlation_edges(correlation, config.max_pair_correlation),
            "monthly_maturity_audit": maturity_audit,
            "candidate_gate": gate,
            "revealed_diagnostic_review": diagnostic_review,
            "portfolio_evaluation": {
                "selection": selection_evaluation,
                "revealed_diagnostic": diagnostic_evaluation,
            },
            "universe": {
                "snapshot_run_id": snapshot_run_id,
                "requested_count": len(requested_codes),
                "loaded_count": int(history["code"].nunique()),
            },
            "artifacts": {
                "factor_diagnostics": str(DIAGNOSTICS_PATH.relative_to(BASE_DIR)),
                "daily_rank_ic": str(IC_PATH.relative_to(BASE_DIR)),
                "discovery_correlation": str(CORRELATION_PATH.relative_to(BASE_DIR)),
                "report": str(REPORT_PATH.relative_to(BASE_DIR)),
                "factor_registry": str(REGISTRY_PATH.relative_to(BASE_DIR)),
            },
            "production_change": False,
            "promotion_eligible": False,
            "warnings": warnings
            + [
                "当前Top50历史回填存在成分股与存续偏差",
                "当前前复权数据不是严格point-in-time复权数据",
                "诊断隔离窗口不能用于重新选择因子或阈值",
                "自动生成因子只更新研究注册表，不会修改生产模型或模拟盘",
                "框架复现因子仅供研究评测，不会修改冻结LGBM特征架构",
            ],
        }
    )
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timestamped = DATA_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    timestamped.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload)
    return payload


def main() -> None:
    payload = run()
    selection = payload["portfolio_evaluation"]["selection"]
    candidate = selection["factor_ensemble"]["performance"]
    baseline = selection["current_rule_baseline"]["performance"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_factor_count": payload["selected_factor_count"],
                "selected_features": payload["selected_features"],
                "selection_candidate_annual_return": candidate["annual_return"],
                "selection_baseline_annual_return": baseline["annual_return"],
                "selection_candidate_sharpe": candidate["sharpe"],
                "selection_baseline_sharpe": baseline["sharpe"],
                "candidate_gate": payload["candidate_gate"],
                "output": str(OUTPUT_PATH),
                "production_change": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
