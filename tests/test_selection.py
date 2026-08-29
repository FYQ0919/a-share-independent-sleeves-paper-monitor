import pandas as pd

from app.data.demo import DemoProvider
from app.models import Sector
from app.selection import SelectionEngine


WEIGHTS = {"momentum": 0.28, "sector": 0.22, "liquidity": 0.18, "value": 0.17, "risk": 0.15}


def test_demo_selection_is_ranked_and_filtered():
    stocks, sectors, _, _ = DemoProvider().load()
    engine = SelectionEngine(50_000_000, 2_000_000_000, WEIGHTS)
    candidates = engine.select(stocks, sectors, top_n=12)

    assert len(candidates) == 12
    assert [item.rank for item in candidates] == list(range(1, 13))
    assert all("ST" not in item.name.upper() for item in candidates)
    assert all(0 <= item.total_score <= 100 for item in candidates)
    assert all(set(item.factor_scores) == set(WEIGHTS) for item in candidates)
    assert all(candidates[index].total_score >= candidates[index + 1].total_score for index in range(11))


def test_stronger_sector_wins_for_equal_stocks():
    rows = []
    for code, industry in (("600001", "强势行业"), ("600002", "弱势行业")):
        rows.append({
            "code": code, "name": f"测试{code}", "price": 20, "pct_change": 2,
            "change_5m": 0.1, "pct_60d": 10, "pct_ytd": 15, "turnover_rate": 3,
            "volume_ratio": 1.4, "amount": 200_000_000, "amplitude": 4, "pe": 20,
            "pb": 2, "market_cap": 20_000_000_000, "industry": industry, "concept": "",
        })
    sectors = [
        Sector("强势行业", "行业", 4.2),
        Sector("弱势行业", "行业", -2.0),
    ]
    candidates = SelectionEngine(1, 1, WEIGHTS).select(pd.DataFrame(rows), sectors, 2)
    assert candidates[0].industry == "强势行业"
    assert candidates[0].factor_scores["sector"] > candidates[1].factor_scores["sector"]


def test_fifty_stock_pool_reserves_technology_candidates():
    stocks, sectors, _, _ = DemoProvider().load()
    engine = SelectionEngine(50_000_000, 2_000_000_000, WEIGHTS, technology_reserve=10)

    candidates = engine.select(stocks, sectors, top_n=50)

    assert len(candidates) == 50
    assert sum(item.is_technology for item in candidates) >= 10
    assert any(item.board == "科创板" for item in candidates)
    assert all(item.board for item in candidates)
