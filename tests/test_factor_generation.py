import pandas as pd

from app.factor_generation import (
    FactorExpression,
    apply_factor_expressions,
    generate_factor_expressions,
    update_factor_registry,
)


def _diagnostics():
    return pd.DataFrame([
        {
            "feature": f"x_factor_{index}",
            "family": "risk_tail" if index < 3 else "liquidity",
            "stability_score": 1.0 - index * 0.05,
            "selection_retention": 0.8,
            "discovery_abs_mean_rank_ic": 0.08 - index * 0.005,
        }
        for index in range(6)
    ])


def test_expression_generation_is_deterministic_bounded_and_structured():
    first = generate_factor_expressions(_diagnostics(), max_expressions=12)
    second = generate_factor_expressions(_diagnostics(), max_expressions=12)

    assert first == second
    assert len(first) == 12
    assert len({row.name for row in first}) == 12
    assert {row.operator for row in first}.issubset({"product", "spread", "gated"})


def test_generated_values_before_cutoff_ignore_future_input_changes():
    dates = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame({
        "date": dates.repeat(3),
        "x_left": [0.1, 0.5, 0.9] * len(dates),
        "x_right": [0.8, 0.4, 0.2] * len(dates),
    })
    expression = FactorExpression(
        "g_test", "gated", "x_left", "x_right", "generated:test", "test"
    )
    cutoff = dates[4]
    original, names = apply_factor_expressions(frame, [expression])
    changed = frame.copy()
    changed.loc[changed["date"].gt(cutoff), ["x_left", "x_right"]] *= 100
    perturbed, changed_names = apply_factor_expressions(changed, [expression])

    assert names == changed_names == ["g_test"]
    pd.testing.assert_series_equal(
        original.loc[original["date"].le(cutoff), "g_test"].reset_index(drop=True),
        perturbed.loc[perturbed["date"].le(cutoff), "g_test"].reset_index(drop=True),
    )


def test_registry_tracks_new_selected_and_retired_factors():
    expressions = [FactorExpression(
        "g_one", "product", "x_a", "x_b", "generated:test", "x_a * x_b"
    )]
    diagnostics = pd.DataFrame([{
        "feature": "g_one",
        "discovery_mean_rank_ic": 0.03,
        "selection_mean_rank_ic": 0.02,
        "diagnostic_mean_rank_ic": 0.01,
        "stability_score": 0.02,
        "max_abs_corr_to_selected": 0.4,
        "selection_reason": "selected",
    }])
    first = update_factor_registry(
        None, expressions, diagnostics, ["g_one"], "2026-08-28T12:00:00", "abc"
    )
    second = update_factor_registry(
        first, [], diagnostics.iloc[0:0], [], "2026-09-28T12:00:00", "def"
    )

    assert first["new_factors"] == ["g_one"]
    assert first["selected_generated_count"] == 1
    assert second["factors"][0]["status"] == "retired"
    assert second["factors"][0]["selection_count"] == 1
