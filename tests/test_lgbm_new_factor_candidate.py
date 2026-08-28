import pandas as pd
import pytest

from scripts.train_lgbm_new_factors_candidate_v1 import (
    CANDIDATE_WEIGHT,
    MODEL_VERSION,
    RULE_WEIGHT,
    blend_candidate_model_ranks,
)


def test_forward_candidate_contract_is_frozen():
    assert MODEL_VERSION == "lgbm_new_factors_blend_v1_candidate"
    assert CANDIDATE_WEIGHT == pytest.approx(0.25)
    assert RULE_WEIGHT == pytest.approx(0.40)


def test_candidate_blend_uses_cross_sectional_ranks():
    baseline = pd.Series([30.0, 20.0, 10.0], index=[3, 4, 5])
    candidate = pd.Series([10.0, 20.0, 30.0], index=[3, 4, 5])

    output = blend_candidate_model_ranks(baseline, candidate, 0.25)

    assert output.index.tolist() == [3, 4, 5]
    assert output.iloc[0] > output.iloc[-1]
    assert output.between(0.0, 1.0).all()


def test_candidate_blend_rejects_invalid_weight():
    with pytest.raises(ValueError, match="模型权重"):
        blend_candidate_model_ranks(pd.Series([1.0]), pd.Series([1.0]), -0.1)
