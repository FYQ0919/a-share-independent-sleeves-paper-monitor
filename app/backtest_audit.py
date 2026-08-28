from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine


CAUSAL_FEATURE_COLUMNS = (
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120", "ma_20",
    "amount_5", "amount_20", "amount_60", "volatility_20", "volatility_60",
    "downside_volatility_60", "absolute_return_60", "absolute_return_120",
    "high_60", "drawdown_60", "high_120", "drawdown_120",
    "intraday_return", "intraday_strength_20",
)


def feature_prefix_invariance(
    engine: BacktestEngine,
    history: pd.DataFrame,
    cutoff: date,
) -> Dict[str, Any]:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    prefix_input = data[data["date"] <= pd.Timestamp(cutoff)].copy()
    full = engine._features(data)
    prefix = engine._features(prefix_input)
    keys = ["date", "code"]
    columns = [name for name in CAUSAL_FEATURE_COLUMNS if name in full and name in prefix]
    joined = prefix[keys + columns].merge(
        full[keys + columns], on=keys, how="inner", suffixes=("_prefix", "_full"), validate="one_to_one"
    )
    mismatches = 0
    max_abs_diff = 0.0
    for column in columns:
        left = pd.to_numeric(joined[f"{column}_prefix"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(joined[f"{column}_full"], errors="coerce").to_numpy(dtype=float)
        equal = np.isclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        mismatches += int((~equal).sum())
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(left[finite] - right[finite]))))
    return {
        "passed": mismatches == 0,
        "cutoff": cutoff.isoformat(),
        "rows_compared": int(len(joined)),
        "features_compared": columns,
        "mismatches": mismatches,
        "max_abs_diff": max_abs_diff,
    }


def future_perturbation_invariance(
    engine: BacktestEngine,
    history: pd.DataFrame,
    cutoff: date,
    factor_weights: Dict[str, float],
) -> Dict[str, Any]:
    data = history.copy()
    data["date"] = pd.to_datetime(data["date"])
    future_mask = data["date"] > pd.Timestamp(cutoff)
    perturbed = data.copy()
    for column, multiplier in {
        "open": 7.0, "high": 11.0, "low": 5.0, "close": 9.0,
        "volume": 101.0, "amount": 103.0, "pe": -13.0, "pb": 17.0,
    }.items():
        if column in perturbed:
            perturbed.loc[future_mask, column] = (
                pd.to_numeric(perturbed.loc[future_mask, column], errors="coerce") * multiplier
            )
    original_features = engine._features(data)
    perturbed_features = engine._features(perturbed)
    original = engine._scored_cross(original_features, pd.Timestamp(cutoff), factor_weights)
    changed = engine._scored_cross(perturbed_features, pd.Timestamp(cutoff), factor_weights)
    columns = ["code", "score"] + [f"{name}_score" for name in factor_weights]
    original = original[columns].sort_values("code").reset_index(drop=True)
    changed = changed[columns].sort_values("code").reset_index(drop=True)
    same_codes = original["code"].astype(str).tolist() == changed["code"].astype(str).tolist()
    score_columns = [name for name in columns if name != "code"]
    same_scores = same_codes and np.allclose(
        original[score_columns].to_numpy(dtype=float),
        changed[score_columns].to_numpy(dtype=float),
        rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    return {
        "passed": bool(same_codes and same_scores),
        "cutoff": cutoff.isoformat(),
        "future_rows_perturbed": int(future_mask.sum()),
        "cross_section_size": int(len(original)),
        "same_codes": same_codes,
        "same_scores": bool(same_scores),
    }


def execution_timing_audit(result: Dict[str, Any]) -> Dict[str, Any]:
    rebalances = result.get("rebalances", [])
    invalid_rebalances = [
        row for row in rebalances
        if pd.Timestamp(row["execution_date"]) <= pd.Timestamp(row["signal_date"])
    ]
    periods = result.get("return_periods", [])
    invalid_periods = [
        row for row in periods
        if pd.Timestamp(row["period_end"]) <= pd.Timestamp(row["period_start"])
    ]
    return {
        "passed": not invalid_rebalances and not invalid_periods,
        "rebalances_checked": len(rebalances),
        "return_periods_checked": len(periods),
        "invalid_rebalances": len(invalid_rebalances),
        "invalid_return_periods": len(invalid_periods),
    }


def split_isolation_audit(
    result: Dict[str, Any],
    splits: Iterable[tuple[str, date, date]],
) -> Dict[str, Any]:
    periods = pd.DataFrame(result.get("return_periods", []))
    if periods.empty:
        return {"passed": False, "reason": "return_periods missing"}
    periods["period_end"] = pd.to_datetime(periods["period_end"])
    assigned = pd.Series(0, index=periods.index, dtype=int)
    details = []
    for label, start, end in splits:
        mask = (
            (periods["period_end"] >= pd.Timestamp(start))
            & (periods["period_end"] <= pd.Timestamp(end))
        )
        assigned.loc[mask] += 1
        selected = periods.loc[mask]
        details.append({
            "label": label,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "return_periods": int(mask.sum()),
            "first_period_end": selected.iloc[0]["period_end"].date().isoformat() if not selected.empty else None,
            "last_period_end": selected.iloc[-1]["period_end"].date().isoformat() if not selected.empty else None,
        })
    overlaps = int((assigned > 1).sum())
    return {
        "passed": overlaps == 0,
        "assignment_rule": "A return belongs only to the split containing its period_end",
        "overlapping_return_periods": overlaps,
        "splits": details,
    }


def strict_gate_audit(history: pd.DataFrame, config: BacktestConfig) -> Dict[str, Any]:
    strict_config = replace(config, strict_no_lookahead=True)
    try:
        BacktestEngine._validate_research_contract(history, strict_config)
    except ValueError as exc:
        return {"passed": False, "blocked": True, "reason": str(exc)}
    return {"passed": True, "blocked": False, "reason": "strict point-in-time contract satisfied"}
