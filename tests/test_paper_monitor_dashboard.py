import asyncio
from types import SimpleNamespace

import app.main as main_module


class MonitorStorage:
    def load_strategy_state(self, account_id):
        return {"last_snapshot": {"account_id": account_id, "nav": 1_000_000}}

    def strategy_decision_history(self, account_id, _limit):
        return [{
            "account_id": account_id,
            "signal_date": "2026-08-28",
            "initial_capital": 1_000_000,
            "nav": 1_000_000,
            "drawdown": 0.0,
            "sleeves": {
                "trend": {"initial_capital": 750_000, "nav": 750_000},
                "lgbm": {"initial_capital": 250_000, "nav": 250_000},
            },
        }]

    def latest(self):
        return None


def test_paper_monitor_api_exposes_independent_sleeves(monkeypatch):
    monkeypatch.setattr(main_module, "storage", MonitorStorage())
    monkeypatch.setattr(
        main_module,
        "pipeline",
        SimpleNamespace(sleeve_monitor=SimpleNamespace(account_id="monitor-v1")),
    )
    payload = asyncio.run(main_module.paper_monitor())
    assert payload["enabled"] is True
    assert payload["account_id"] == "monitor-v1"
    assert payload["history"][0]["equity"] == 1.0
    assert payload["history"][0]["trend_equity"] == 1.0
    assert payload["config"]["promotion_required_sessions"] == 126


def test_monitor_page_has_curve_sleeves_hedge_and_costs():
    template = (main_module.BASE_DIR / "templates" / "paper_monitor.html").read_text(
        encoding="utf-8"
    )
    script = (main_module.BASE_DIR / "static" / "paper_monitor.js").read_text(
        encoding="utf-8"
    )
    assert 'id="monitor-equity-chart"' in template
    assert 'id="trend-positions"' in template
    assert 'id="lgbm-positions"' in template
    assert 'id="hedge-active"' in template
    assert "renderChart(history)" in script
