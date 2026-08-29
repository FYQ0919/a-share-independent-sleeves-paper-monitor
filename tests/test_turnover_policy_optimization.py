from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from optimize_turnover_policy_v1 import (  # noqa: E402
    is_training_improver,
    validate_candidates,
    validation_gate,
)


def _row(annual_return, sharpe, drawdown, annual_turnover):
    period = {
        "performance": {
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
        },
        "activity": {"annual_turnover": annual_turnover},
    }
    return {"training": {"aggregate": period}, "validation": period}


def test_candidate_manifest_only_tightens_turnover_controls():
    assert len(validate_candidates()) == 64


def test_training_improver_requires_return_sharpe_and_turnover_improvements():
    baseline = _row(0.05, 0.30, -0.20, 20.0)
    assert is_training_improver(_row(0.06, 0.31, -0.21, 18.0), baseline)
    assert not is_training_improver(_row(0.06, 0.31, -0.21, 19.5), baseline)
    assert not is_training_improver(_row(0.04, 0.31, -0.21, 18.0), baseline)


def test_validation_gate_rejects_lower_annual_return():
    baseline = _row(0.05, 0.30, -0.20, 20.0)
    candidate = _row(0.04, 0.29, -0.21, 18.0)
    gate = validation_gate(candidate, baseline)
    assert not gate["passed"]
    assert not gate["checks"]["annual_return_improved"]
    assert gate["checks"]["annual_turnover_lower"]
