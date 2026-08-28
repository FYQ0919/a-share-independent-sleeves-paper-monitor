from __future__ import annotations

import pandas as pd

from app.alpha101_extra_factors import (
    ALPHA101_EXTRA_FACTOR_SPECS,
    add_alpha101_extra_factors,
    alpha101_extra_factor_metadata,
)
from app.barra_residual_factors import (
    BARRA_RESIDUAL_FACTOR_SPECS,
    add_barra_residual_factors,
    barra_residual_factor_metadata,
)
from app.framework_factors import (
    FRAMEWORK_FACTOR_SPECS,
    add_framework_factors,
    framework_factor_metadata,
)
from app.qlib_extra_factors import (
    QLIB_EXTRA_FACTOR_SPECS,
    add_qlib_extra_factors,
    qlib_extra_factor_metadata,
)
from app.qlib_multiscale_factors import (
    QLIB_MULTISCALE_FACTOR_SPECS,
    add_qlib_multiscale_factors,
    qlib_multiscale_factor_metadata,
)
from app.risk_liquidity_factors import (
    RISK_LIQUIDITY_FACTOR_SPECS,
    add_risk_liquidity_factors,
    risk_liquidity_factor_metadata,
)


RESEARCH_FACTOR_BATCHES = {
    "framework_v1": FRAMEWORK_FACTOR_SPECS,
    "qlib_extra_v1": QLIB_EXTRA_FACTOR_SPECS,
    "risk_liquidity_v1": RISK_LIQUIDITY_FACTOR_SPECS,
    "alpha101_extra_v1": ALPHA101_EXTRA_FACTOR_SPECS,
    "qlib_multiscale_v1": QLIB_MULTISCALE_FACTOR_SPECS,
    "barra_residual_v1": BARRA_RESIDUAL_FACTOR_SPECS,
}


def _validate_unique_names() -> None:
    seen: dict[str, str] = {}
    for batch, specs in RESEARCH_FACTOR_BATCHES.items():
        for name in specs:
            if name in seen:
                raise RuntimeError(f"研究因子重名: {name} ({seen[name]}, {batch})")
            seen[name] = batch


_validate_unique_names()


def research_factor_metadata(feature: str) -> dict[str, str] | None:
    for batch, lookup in (
        ("framework_v1", framework_factor_metadata),
        ("qlib_extra_v1", qlib_extra_factor_metadata),
        ("risk_liquidity_v1", risk_liquidity_factor_metadata),
        ("alpha101_extra_v1", alpha101_extra_factor_metadata),
        ("qlib_multiscale_v1", qlib_multiscale_factor_metadata),
        ("barra_residual_v1", barra_residual_factor_metadata),
    ):
        metadata = lookup(feature)
        if metadata:
            return {**metadata, "batch": batch}
    return None


def add_research_factors(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    output, framework = add_framework_factors(frame)
    output, qlib_extra = add_qlib_extra_factors(output)
    output, risk_liquidity = add_risk_liquidity_factors(output)
    output, alpha101_extra = add_alpha101_extra_factors(output)
    output, qlib_multiscale = add_qlib_multiscale_factors(output)
    output, barra_residual = add_barra_residual_factors(output)
    batches = {
        "framework_v1": framework,
        "qlib_extra_v1": qlib_extra,
        "risk_liquidity_v1": risk_liquidity,
        "alpha101_extra_v1": alpha101_extra,
        "qlib_multiscale_v1": qlib_multiscale,
        "barra_residual_v1": barra_residual,
    }
    features = (
        framework
        + qlib_extra
        + risk_liquidity
        + alpha101_extra
        + qlib_multiscale
        + barra_residual
    )
    if len(features) != len(set(features)):
        raise RuntimeError("研究因子聚合结果存在重名")
    return output, features, batches
