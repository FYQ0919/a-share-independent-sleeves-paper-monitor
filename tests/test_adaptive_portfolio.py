from datetime import date, timedelta

from app.adaptive_portfolio import AdaptivePortfolioService
from app.models import StrategyPick
from app.storage import Storage


def rankings(order, scores=None):
    scores = scores or {}
    return [
        StrategyPick(
            rank=index,
            code=code,
            name=f"股票{code}",
            score=scores.get(code, 100 - index * 5),
            price=10 + index,
            target_weight=0.2,
            factor_scores={"中期反转": 0.8},
            reason="中期反转",
        )
        for index, code in enumerate(order, start=1)
    ]


def test_adaptive_portfolio_persists_pending_and_is_idempotent(tmp_path):
    storage = Storage(tmp_path / "research.db")
    service = AdaptivePortfolioService(storage)
    signal = date(2026, 8, 25)
    calendar = [signal]
    ranked = rankings([f"60000{i}" for i in range(1, 9)])

    first = service.evaluate(ranked, signal, calendar)
    repeated = service.evaluate(ranked, signal, calendar)

    assert first == repeated
    assert first["action_label"] == "主调仓"
    assert first["trade_required"] is True
    assert len(first["target_holdings"]) == 5
    assert first["current_holdings"] == []
    assert len(storage.strategy_decision_history(service.STRATEGY_ID)) == 1


def test_adaptive_portfolio_executes_next_open_then_waits_minimum_hold(tmp_path):
    storage = Storage(tmp_path / "research.db")
    service = AdaptivePortfolioService(storage, min_hold_days=3, score_gap=20)
    first_day = date(2026, 8, 24)
    codes = [f"60000{i}" for i in range(1, 9)]
    base = rankings(codes)
    service.evaluate(base, first_day, [first_day])

    second_day = first_day + timedelta(days=1)
    changed = rankings([codes[5], *codes[:5], *codes[6:]], {codes[5]: 99, codes[4]: 40})
    decision = service.evaluate(changed, second_day, [first_day, second_day])

    assert decision["executed_order"]["execution_date"] == second_day.isoformat()
    assert len(decision["current_holdings"]) == 5
    assert decision["trade_required"] is False
    assert decision["action_label"] == "继续持有"
    assert "最少 3 个交易日" in decision["reason"]


def test_adaptive_portfolio_allows_one_early_replacement_per_cycle(tmp_path):
    storage = Storage(tmp_path / "research.db")
    service = AdaptivePortfolioService(storage, min_hold_days=3, rank_buffer=7, score_gap=20)
    start = date(2026, 8, 17)
    dates = [start + timedelta(days=index) for index in range(6)]
    codes = [f"60000{i}" for i in range(1, 10)]
    base = rankings(codes)
    service.evaluate(base, dates[0], dates[:1])
    for index in range(1, 4):
        service.evaluate(base, dates[index], dates[: index + 1])

    changed = rankings([codes[5], *codes[:4], codes[6], codes[7], codes[4], codes[8]], {codes[5]: 99, codes[4]: 30})
    early = service.evaluate(changed, dates[4], dates[:5])
    assert early["action_label"] == "提前换股"
    assert early["trade_required"] is True
    assert early["adaptive_used_in_cycle"] == 1
    assert [item["code"] for item in early["buys"]] == [codes[5]]
    assert [item["code"] for item in early["sells"]] == [codes[4]]

    second_challenge = rankings([codes[6], codes[5], *codes[:4], codes[7], codes[8], codes[4]], {codes[6]: 100, codes[3]: 20})
    held = service.evaluate(second_challenge, dates[5], dates)
    assert held["trade_required"] is False
    assert "已使用一次" in held["reason"]


def test_fixed_cycle_portfolio_does_not_replace_between_rebalances(tmp_path):
    storage = Storage(tmp_path / "research.db")
    service = AdaptivePortfolioService(
        storage,
        rebalance_days=10,
        min_hold_days=1,
        rank_buffer=5,
        score_gap=1,
        adaptive_enabled=False,
        strategy_id="fixed10_test",
    )
    start = date(2026, 8, 17)
    dates = [start + timedelta(days=index) for index in range(3)]
    codes = [f"60000{i}" for i in range(1, 9)]
    service.evaluate(rankings(codes), dates[0], dates[:1])
    service.evaluate(rankings(codes), dates[1], dates[:2])

    challenger = rankings(
        [codes[6], codes[7], *codes[:6]], {codes[6]: 100, codes[4]: 1}
    )
    decision = service.evaluate(challenger, dates[2], dates)

    assert decision["policy"] == "固定周期调仓"
    assert decision["trade_required"] is False
    assert decision["rules"]["adaptive_enabled"] is False
    assert "周期内不提前换股" in decision["reason"]


def test_risk_replacement_keeps_holdings_missing_from_reduced_ranking(tmp_path):
    storage = Storage(tmp_path / "research.db")
    service = AdaptivePortfolioService(storage)
    first_day = date(2026, 8, 25)
    original_codes = [f"60000{i}" for i in range(1, 6)]
    service.evaluate(rankings(original_codes), first_day, [first_day])

    second_day = first_day + timedelta(days=1)
    challenger = "600015"
    reduced_ranking = rankings([challenger, original_codes[1]])
    decision = service.evaluate(reduced_ranking, second_day, [first_day, second_day])

    assert decision["action_label"] == "风险退出"
    assert len(decision["target_holdings"]) == 5
    assert [item["code"] for item in decision["buys"]] == [challenger]
    assert [item["code"] for item in decision["sells"]] == [original_codes[0]]
    assert {item["code"] for item in decision["target_holdings"]} == {
        challenger,
        *original_codes[1:],
    }
    assert all(item["target_weight"] == 0.2 for item in decision["target_holdings"])
