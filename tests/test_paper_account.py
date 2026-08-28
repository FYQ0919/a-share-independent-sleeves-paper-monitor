from datetime import date

import pandas as pd

from app.config import settings
from app.notifications import NotificationService
from app.paper_account import LgbmPaperAccountService
from app.storage import Storage


def history_for(dates, codes):
    rows = []
    for day_index, day in enumerate(dates):
        for code_index, code in enumerate(codes):
            rows.append({
                "date": pd.Timestamp(day),
                "code": code,
                "name": f"股票{code}",
                "open": 10.0 + code_index,
                "close": 10.5 + code_index + day_index,
                "tradestatus": 1,
            })
    return pd.DataFrame(rows)


def holdings(codes):
    return [
        {
            "code": code,
            "name": f"股票{code}",
            "target_weight": 1 / len(codes),
        }
        for code in codes
    ]


def test_paper_account_waits_for_next_open_and_is_idempotent(tmp_path):
    storage = Storage(tmp_path / "research.db")
    account = LgbmPaperAccountService(storage, initial_capital=1_000_000, cost_bps=12)
    codes = [f"60000{index}" for index in range(1, 6)]
    first_day = date(2026, 8, 24)
    second_day = date(2026, 8, 25)

    first = account.update(
        {
            "signal_date": first_day.isoformat(),
            "current_holdings": [],
            "executed_order": None,
        },
        history_for([first_day], codes),
    )
    assert first["nav"] == 1_000_000
    assert first["positions"] == []
    assert first["trades"] == []

    decision = {
        "signal_date": second_day.isoformat(),
        "current_holdings": holdings(codes),
        "executed_order": {
            "signal_date": first_day.isoformat(),
            "execution_date": second_day.isoformat(),
            "action_label": "主调仓",
        },
    }
    second = account.update(decision, history_for([first_day, second_day], codes))
    repeated = account.update(decision, history_for([first_day, second_day], codes))

    assert repeated == second
    assert len(second["trades"]) == 5
    assert {item["side"] for item in second["trades"]} == {"BUY"}
    assert all(item["execution_date"] == second_day.isoformat() for item in second["trades"])
    assert all(item["shares"] % 100 == 0 for item in second["trades"])
    assert second["transaction_cost_total"] > 0
    assert second["cash"] >= 0
    assert len(second["positions"]) == 5
    assert len(storage.strategy_decision_history(account.ACCOUNT_ID)) == 2


def test_paper_account_missing_open_price_fails_closed_for_symbol(tmp_path):
    storage = Storage(tmp_path / "research.db")
    account = LgbmPaperAccountService(storage)
    codes = ["600001", "600002"]
    first_day = date(2026, 8, 24)
    second_day = date(2026, 8, 25)
    account.update(
        {"signal_date": first_day.isoformat(), "current_holdings": [], "executed_order": None},
        history_for([first_day], codes),
    )
    second_history = history_for([first_day, second_day], codes)
    second_history = second_history[
        ~(
            second_history["date"].eq(pd.Timestamp(second_day))
            & second_history["code"].eq("600002")
        )
    ]
    snapshot = account.update(
        {
            "signal_date": second_day.isoformat(),
            "current_holdings": holdings(codes),
            "executed_order": {
                "signal_date": first_day.isoformat(),
                "execution_date": second_day.isoformat(),
                "action_label": "主调仓",
            },
        },
        second_history,
    )

    assert [item["code"] for item in snapshot["positions"]] == ["600001"]
    assert any("600002" in item and "缺少有效开盘价" in item for item in snapshot["warnings"])


def test_paper_accounts_with_different_ids_do_not_share_state(tmp_path):
    storage = Storage(tmp_path / "research.db")
    old_account = LgbmPaperAccountService(storage)
    blend_account = LgbmPaperAccountService(
        storage,
        account_id="lgbm_new_factors_blend_v1_candidate_paper_account_v1",
        strategy_label="75/25 Alpha158-Barra LGBM 模拟盘",
    )
    signal_date = date(2026, 8, 25)
    history = history_for([signal_date], ["600001"])

    old_account.update(
        {"signal_date": signal_date.isoformat(), "current_holdings": []}, history
    )
    blend_snapshot = blend_account.update(
        {"signal_date": signal_date.isoformat(), "current_holdings": []}, history
    )

    assert blend_snapshot["account_id"] == blend_account.account_id
    assert "75/25" in blend_snapshot["strategy"]
    assert storage.load_strategy_state(old_account.account_id)["account_id"] == old_account.account_id
    assert storage.load_strategy_state(blend_account.account_id)["account_id"] == blend_account.account_id


def test_wecom_message_contains_paper_account_results():
    message = NotificationService(settings)._compact_message(
        "LGBM模拟盘日报｜2026-08-25",
        "市场均衡。",
        [],
        [],
        [],
        {
            "model_version": "lgbm_ranker_top5_v1",
            "paper_account": {
                "signal_date": "2026-08-25",
                "nav": 1_012_345.67,
                "daily_pnl": 2_345.67,
                "daily_return": 0.00232,
                "cumulative_return": 0.01235,
                "drawdown": -0.01,
                "cash": 12_345.67,
                "exposure": 0.9878,
                "transaction_cost_total": 456.78,
                "trades": [{
                    "side": "BUY",
                    "name": "样例股票",
                    "code": "600001",
                    "shares": 1000,
                    "price": 10.0,
                }],
                "positions": [{
                    "name": "样例股票",
                    "code": "600001",
                    "shares": 1000,
                    "weight": 0.2,
                }],
                "warnings": [],
            },
        },
        {},
    )

    assert "LGBM 模拟盘" in message
    assert "净值 1,012,345.67" in message
    assert "累计 +1.23%" in message
    assert "买入样例股票 1000股@10.00" in message
    assert "仅为模拟盘，不连接券商" in message
