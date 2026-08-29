from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest_data import DemoHistoryProvider
from app.config import settings
from app.lgbm_blend_strategy import MODEL_VERSION, LgbmBlendStrategyModel
from app.lgbm_strategy import load_frozen_strategy_model
from app.paper_account import LgbmPaperAccountService
from app.pipeline import ResearchPipeline
from app.storage import Storage


MODEL_DIR = Path("data/models") / MODEL_VERSION


def demo_history():
    codes = [f"{600000 + index:06d}" for index in range(12)]
    history, _ = DemoHistoryProvider().load(
        codes, {}, date(2023, 1, 1), date(2026, 8, 25)
    )
    return history


def test_blend_loader_checks_frozen_models_and_weights():
    model = load_frozen_strategy_model(MODEL_DIR)

    assert isinstance(model, LgbmBlendStrategyModel)
    assert model.metadata["artifact_status"] == "frozen_for_forward_shadow_only"
    assert model.baseline_model_weight == 0.75
    assert model.candidate_model_weight == 0.25
    assert model.rule_weight == 0.40
    assert len(model.baseline_features) == 85
    assert len(model.candidate_features) == 100
    assert model.metadata["baseline_model_audit"]["strictly_mature"] is True
    assert model.metadata["candidate_model_audit"]["strictly_mature"] is True


def test_blend_daily_rank_is_causal_after_signal_date():
    history = demo_history()
    cutoff = date(2026, 6, 30)
    model = LgbmBlendStrategyModel(MODEL_DIR)
    original, metadata = model.rank(history, cutoff)

    changed = history.copy()
    future = changed["date"].gt(pd.Timestamp(cutoff))
    changed.loc[future, ["open", "high", "low", "close", "volume", "amount"]] *= 7.0
    rerun, changed_metadata = model.rank(changed, cutoff)

    assert metadata["signal_date"] == cutoff.isoformat()
    assert changed_metadata["signal_date"] == cutoff.isoformat()
    assert original["code"].tolist() == rerun["code"].tolist()
    np.testing.assert_allclose(
        original["score"].to_numpy(), rerun["score"].to_numpy(), rtol=0, atol=0
    )


def test_blend_pipeline_uses_independent_portfolio_and_paper_account(tmp_path):
    test_settings = replace(
        settings,
        strategy_model_backend="lgbm_active",
        lgbm_model_version=MODEL_VERSION,
        lgbm_model_dir=MODEL_DIR,
        lgbm_universe_mode="frozen_snapshot",
        independent_sleeves_enabled=False,
        index_hedge_enabled=False,
        database_path=tmp_path / "research.db",
        report_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
    )
    pipeline = ResearchPipeline(test_settings, Storage(test_settings.database_path))

    assert pipeline.portfolio.strategy_id == f"{MODEL_VERSION}_fixed10_portfolio_v1"
    assert pipeline.portfolio.adaptive_enabled is False
    assert pipeline.paper_account.account_id == f"{MODEL_VERSION}_fixed10_paper_account_v1"
    assert pipeline.paper_account.account_id != LgbmPaperAccountService.ACCOUNT_ID
    assert "75/25" in pipeline.paper_account.strategy_label
