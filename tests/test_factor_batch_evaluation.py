from scripts.evaluate_factor_batch import BATCH_BUILDERS


def test_factor_batches_are_explicit_and_separate():
    assert set(BATCH_BUILDERS) == {
        "alpha158_only",
        "qlib_extra_v1",
        "risk_liquidity_v1",
        "combined_v1",
        "alpha101_extra_v1",
        "qlib_multiscale_v1",
        "barra_residual_v1",
        "framework_replication_v2",
    }
    assert BATCH_BUILDERS["qlib_extra_v1"] is not BATCH_BUILDERS["risk_liquidity_v1"]
