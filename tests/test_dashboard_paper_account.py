import asyncio
from types import SimpleNamespace

import app.main as main_module


class FakeStorage:
    def __init__(self, account_id):
        self.account_id = account_id

    def latest(self):
        return {"run_id": "latest-run"}

    def load_strategy_state(self, state_id):
        if state_id == self.account_id:
            return {
                "account_id": state_id,
                "last_snapshot": {
                    "account_id": state_id,
                    "nav": 1_000_000,
                    "cash": 1_000_000,
                    "positions": [],
                },
            }
        return {"strategy_id": state_id, "pending_order": {"execution_date": "2026-08-31"}}

    def strategy_decision_history(self, *_args):
        return []


def test_dashboard_exposes_latest_paper_account_snapshot(monkeypatch):
    account_id = "blend_fixed10_paper_account_v1"
    fake_pipeline = SimpleNamespace(
        portfolio=SimpleNamespace(strategy_id="blend_fixed10_portfolio_v1"),
        paper_account=SimpleNamespace(account_id=account_id),
        index_hedge=None,
        strategy_signal=SimpleNamespace(model_status=lambda: {"model_version": "blend"}),
    )
    monkeypatch.setattr(main_module, "storage", FakeStorage(account_id))
    monkeypatch.setattr(main_module, "pipeline", fake_pipeline)
    monkeypatch.setattr(main_module, "scheduler", SimpleNamespace(running=False))

    payload = asyncio.run(main_module.dashboard())

    assert payload["strategy"]["paper_account"]["nav"] == 1_000_000
    assert payload["strategy"]["paper_account_state"]["account_id"] == account_id


def test_dashboard_page_contains_paper_account_renderer():
    template = (main_module.BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    script = (main_module.BASE_DIR / "static" / "app.js").read_text(encoding="utf-8")

    assert 'id="paper-account-content"' in template
    assert "renderPaperAccount(" in script


def test_dashboard_exposes_csi300_hedge_snapshot(monkeypatch):
    hedge_id = "blend_fixed10_csi300_hedged_hedge_overlay_v1"
    monkeypatch.setattr(main_module.pipeline, "index_hedge", SimpleNamespace(account_id=hedge_id))
    monkeypatch.setattr(
        main_module.storage,
        "load_strategy_state",
        lambda account_id: {
            "last_snapshot": {
                "account_id": account_id,
                "target_hedge_ratio": 0.75,
            }
        },
    )
    monkeypatch.setattr(main_module.storage, "strategy_decision_history", lambda *_: [])

    payload = asyncio.run(main_module.dashboard())

    assert payload["strategy"]["index_hedge_account_id"] == hedge_id
    assert payload["strategy"]["index_hedge"]["target_hedge_ratio"] == 0.75
