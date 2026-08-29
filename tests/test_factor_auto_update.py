from datetime import datetime, timezone

import pandas as pd

from app.config import Settings
from app.scheduler import build_scheduler
from scripts.update_factor_registry import rolling_factor_config


class _Pipeline:
    def run(self, mode, notify):
        return None


def test_rolling_windows_are_ordered_and_non_overlapping():
    dates = pd.bdate_range("2018-01-02", periods=1800)
    history = pd.DataFrame({"date": dates})

    config = rolling_factor_config(history)

    assert config.discovery.start < config.discovery.end < config.selection.start
    assert config.selection.start < config.selection.end < config.diagnostic.start
    assert config.diagnostic.start < config.diagnostic.end
    assert len(dates[(dates >= pd.Timestamp(config.diagnostic.start)) & (dates <= pd.Timestamp(config.diagnostic.end))]) == 252


def test_scheduler_registers_monthly_factor_update_when_enabled():
    settings = Settings(
        factor_auto_update_enabled=True,
        factor_update_day=2,
        factor_update_hour=20,
        factor_update_minute=15,
    )
    scheduler = build_scheduler(settings, _Pipeline())

    job = scheduler.get_job("monthly_factor_update")

    assert job is not None
    assert "day='2'" in str(job.trigger)
    assert "hour='20'" in str(job.trigger)
