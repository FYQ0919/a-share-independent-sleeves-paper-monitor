from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import BASE_DIR, settings
from app.alpha101_extra_factors import add_alpha101_extra_factors
from app.barra_residual_factors import add_barra_residual_factors
from app.factor_mining import (
    daily_rank_ic,
    discovery_feature_correlation,
    factor_diagnostics,
    select_stable_nonredundant_factors,
)
from app.lgbm_features import build_alpha158_lite
from app.qlib_extra_factors import add_qlib_extra_factors
from app.qlib_multiscale_factors import add_qlib_multiscale_factors
from app.risk_liquidity_factors import add_risk_liquidity_factors
from app.storage import Storage
from optimize_factor_strategy import load_current_history
from update_factor_registry import _latest_candidates, rolling_factor_config


BATCH_BUILDERS = {
    "qlib_extra_v1": add_qlib_extra_factors,
    "risk_liquidity_v1": add_risk_liquidity_factors,
    "alpha101_extra_v1": add_alpha101_extra_factors,
    "qlib_multiscale_v1": add_qlib_multiscale_factors,
    "barra_residual_v1": add_barra_residual_factors,
}


def _add_combined_v1(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output, qlib_features = add_qlib_extra_factors(frame)
    output, risk_features = add_risk_liquidity_factors(output)
    return output, qlib_features + risk_features


def _alpha158_only(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return frame.copy(), []


def _add_framework_replication_v2(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    output, alpha101_features = add_alpha101_extra_factors(frame)
    output, qlib_multiscale_features = add_qlib_multiscale_factors(output)
    output, barra_features = add_barra_residual_factors(output)
    return output, alpha101_features + qlib_multiscale_features + barra_features


BATCH_BUILDERS.update({
    "alpha158_only": _alpha158_only,
    "combined_v1": _add_combined_v1,
    "framework_replication_v2": _add_framework_replication_v2,
})
OUTPUT_DIR = BASE_DIR / "reports" / "factor_mining" / "batches"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def evaluate_batch(batch: str) -> dict:
    if batch not in BATCH_BUILDERS:
        raise ValueError(f"未知因子批次: {batch}")
    storage = Storage(settings.database_path)
    candidates, snapshot_run_id = _latest_candidates(storage)
    history, requested_codes, warnings = load_current_history(candidates)
    history["date"] = pd.to_datetime(history["date"])
    config = rolling_factor_config(history)

    # Batch acceptance never reads the already revealed latest quarantine prices.
    research_history = history[
        history["date"].le(pd.Timestamp(config.selection.end))
    ].copy()
    frame, alpha_features = build_alpha158_lite(research_history)
    frame, batch_features = BATCH_BUILDERS[batch](frame)
    feature_columns = alpha_features + batch_features
    ic_rows = daily_rank_ic(frame, feature_columns, config.min_cross_section)
    diagnostics = factor_diagnostics(ic_rows, feature_columns, config)
    correlation = discovery_feature_correlation(frame, feature_columns, config.discovery)
    diagnostics, selected = select_stable_nonredundant_factors(
        diagnostics, correlation, config
    )
    batch_rows = diagnostics[diagnostics["feature"].isin(batch_features)].copy()
    quality_failure_reasons = {
        "discovery_days", "selection_days", "weak_discovery_ic",
        "unstable_discovery_direction", "unstable_selection_direction",
        "selection_sign_reversal",
    }
    batch_rows["quality_gate_passed"] = ~batch_rows["selection_reason"].isin(
        quality_failure_reasons
    )
    selected_batch = [name for name in selected if name in batch_features]
    payload = _json_ready({
        "generated_at": datetime.now().astimezone().isoformat(),
        "batch": batch,
        "snapshot_run_id": snapshot_run_id,
        "requested_count": len(requested_codes),
        "loaded_count": int(research_history["code"].nunique()),
        "quarantine_prices_loaded": False,
        "discovery": config.to_dict()["discovery"],
        "selection": config.to_dict()["selection"],
        "alpha158_count": len(alpha_features),
        "batch_factor_count": len(batch_features),
        "quality_gate_passed_count": int(batch_rows["quality_gate_passed"].sum()),
        "selected_in_combined_pool_count": len(selected_batch),
        "selected_in_combined_pool": selected_batch,
        "selected_full_pool": selected,
        "diagnostics": batch_rows.to_dict(orient="records"),
        "warnings": warnings,
        "production_change": False,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{batch}.json"
    csv_path = OUTPUT_DIR / f"{batch}.csv"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    batch_rows.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    payload["output"] = str(output_path.relative_to(BASE_DIR))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", choices=sorted(BATCH_BUILDERS))
    args = parser.parse_args()
    payload = evaluate_batch(args.batch)
    print(json.dumps({
        "batch": payload["batch"],
        "batch_factor_count": payload["batch_factor_count"],
        "quality_gate_passed_count": payload["quality_gate_passed_count"],
        "selected_in_combined_pool": payload["selected_in_combined_pool"],
        "quarantine_prices_loaded": payload["quarantine_prices_loaded"],
        "output": payload["output"],
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
