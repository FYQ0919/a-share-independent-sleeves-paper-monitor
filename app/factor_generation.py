from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd


ALLOWED_OPERATORS = {"product", "spread", "gated"}


@dataclass(frozen=True)
class FactorExpression:
    name: str
    operator: str
    left: str
    right: str
    family: str
    formula: str
    depth: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def _expression_name(operator: str, left: str, right: str) -> str:
    identity = f"{operator}|{left}|{right}"
    digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:12]
    return f"g_{operator}_{digest}"


def _valid_anchor_rows(diagnostics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "feature", "family", "stability_score", "selection_retention",
        "discovery_abs_mean_rank_ic",
    }
    missing = required.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"基础因子诊断缺少列: {', '.join(sorted(missing))}")
    rows = diagnostics.copy()
    rows = rows[
        rows["feature"].astype(str).str.startswith("x_")
        & rows["selection_retention"].gt(0)
        & rows["discovery_abs_mean_rank_ic"].gt(0)
    ]
    return rows.sort_values(
        ["stability_score", "discovery_abs_mean_rank_ic", "feature"],
        ascending=[False, False, True],
    )


def generate_factor_expressions(
    diagnostics: pd.DataFrame,
    max_anchors: int = 12,
    max_per_family: int = 3,
    max_expressions: int = 48,
) -> list[FactorExpression]:
    if min(max_anchors, max_per_family, max_expressions) < 1:
        raise ValueError("自动因子生成上限必须为正数")
    rows = _valid_anchor_rows(diagnostics)
    family_counts: dict[str, int] = {}
    anchors = []
    for row in rows.itertuples(index=False):
        family = str(row.family)
        if family_counts.get(family, 0) >= max_per_family:
            continue
        anchors.append((str(row.feature), family))
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(anchors) >= max_anchors:
            break

    expressions: list[FactorExpression] = []
    seen: set[tuple[str, str, str]] = set()

    def add(operator: str, left: tuple[str, str], right: tuple[str, str]) -> None:
        if len(expressions) >= max_expressions:
            return
        left_name, left_family = left
        right_name, right_family = right
        if operator in {"product", "spread"} and right_name < left_name:
            left_name, right_name = right_name, left_name
            left_family, right_family = right_family, left_family
        identity = (operator, left_name, right_name)
        if identity in seen:
            return
        seen.add(identity)
        formulas = {
            "product": f"{left_name} * {right_name}",
            "spread": f"0.5 * ({left_name} - {right_name})",
            "gated": f"{left_name} * ({right_name} + 1) / 2",
        }
        expressions.append(FactorExpression(
            name=_expression_name(operator, left_name, right_name),
            operator=operator,
            left=left_name,
            right=right_name,
            family=f"generated:{left_family}+{right_family}",
            formula=formulas[operator],
        ))

    for left_index, left in enumerate(anchors):
        for right in anchors[left_index + 1 :]:
            if left[1] == right[1]:
                add("spread", left, right)
            else:
                add("product", left, right)
                add("gated", left, right)
            if len(expressions) >= max_expressions:
                return expressions
    return expressions


def apply_factor_expressions(
    frame: pd.DataFrame,
    expressions: Iterable[FactorExpression],
) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    generated: dict[str, pd.Series] = {}
    names = []
    for expression in expressions:
        if expression.operator not in ALLOWED_OPERATORS:
            raise ValueError(f"不支持的自动因子算子: {expression.operator}")
        missing = [
            name for name in (expression.left, expression.right)
            if name not in output.columns
        ]
        if missing:
            raise ValueError(f"自动因子输入不存在: {', '.join(missing)}")
        left = pd.to_numeric(output[expression.left], errors="coerce")
        right = pd.to_numeric(output[expression.right], errors="coerce")
        if expression.operator == "product":
            raw = left.mul(right)
        elif expression.operator == "spread":
            raw = left.sub(right).mul(0.5)
        else:
            raw = left.mul(right.add(1).mul(0.5))
        ranked = raw.groupby(output["date"]).rank(pct=True).sub(0.5).mul(2)
        generated[expression.name] = ranked
        names.append(expression.name)
    if generated:
        output = pd.concat([output, pd.DataFrame(generated, index=output.index)], axis=1)
    return output, names


def update_factor_registry(
    previous: dict | None,
    expressions: Iterable[FactorExpression],
    diagnostics: pd.DataFrame,
    selected_features: list[str],
    generated_at: str,
    source_manifest: str,
) -> dict:
    previous = previous or {}
    prior_factors = {
        row["name"]: row for row in previous.get("factors", []) if "name" in row
    }
    diagnostic_lookup = diagnostics.set_index("feature").to_dict(orient="index")
    expression_rows = list(expressions)
    current_names = {row.name for row in expression_rows}
    factors = []
    new_names = []
    for expression in expression_rows:
        prior = prior_factors.get(expression.name, {})
        if not prior:
            new_names.append(expression.name)
        selected = expression.name in selected_features
        metrics = diagnostic_lookup.get(expression.name, {})
        factors.append({
            **expression.to_dict(),
            "first_seen": prior.get("first_seen", generated_at),
            "last_evaluated": generated_at,
            "evaluation_count": int(prior.get("evaluation_count", 0)) + 1,
            "selection_count": int(prior.get("selection_count", 0)) + int(selected),
            "selected": selected,
            "status": "active_research" if selected else "rejected",
            "metrics": {
                key: metrics.get(key)
                for key in (
                    "discovery_mean_rank_ic", "selection_mean_rank_ic",
                    "diagnostic_mean_rank_ic", "stability_score",
                    "max_abs_corr_to_selected", "selection_reason",
                )
            },
        })
    retired = []
    for name, row in prior_factors.items():
        if name in current_names:
            continue
        retired_row = dict(row)
        retired_row.update({"selected": False, "status": "retired", "last_evaluated": generated_at})
        retired.append(retired_row)
    factors.extend(retired)
    canonical = json.dumps(
        {
            "source_manifest": source_manifest,
            "active": sorted(row["name"] for row in factors if row.get("selected")),
            "formulas": sorted(
                (row["name"], row.get("formula", "")) for row in factors
                if row.get("status") != "retired"
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    version = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    history = list(previous.get("history", []))[-23:]
    history.append({
        "generated_at": generated_at,
        "version": version,
        "generated_count": len(expression_rows),
        "new_count": len(new_names),
        "selected_generated_count": sum(row.name in selected_features for row in expression_rows),
    })
    return {
        "updated_at": generated_at,
        "version": version,
        "source_manifest": source_manifest,
        "generated_count": len(expression_rows),
        "new_factor_count": len(new_names),
        "new_factors": new_names,
        "selected_generated_count": sum(row.name in selected_features for row in expression_rows),
        "factors": factors,
        "history": history,
        "production_change": False,
    }
