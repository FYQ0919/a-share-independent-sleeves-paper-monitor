from datetime import datetime, timedelta
import json

from app.macro_environment import MacroEnvironmentService


def test_macro_assessment_scores_three_dimensions_and_ignores_future_records():
    now = datetime(2026, 8, 27, 10, 30)
    result = MacroEnvironmentService.assess_inputs(
        now=now,
        breadth={"up": 3500, "down": 1400, "flat": 100},
        global_indices=[
            {"名称": "纳斯达克综合指数", "涨跌幅": 1.0, "行情时间": now - timedelta(hours=10)},
            {"名称": "标普500指数", "涨跌幅": -5.0, "行情时间": now + timedelta(hours=2)},
        ],
        pmi=[{"制造业-指数": 51.0, "非制造业-指数": 50.5}],
        lpr=[
            {"TRADE_DATE": "2026-07-20", "LPR1Y": 3.1},
            {"TRADE_DATE": "2026-08-20", "LPR1Y": 3.0},
        ],
        policy_news=[{"title": "加大支持科技创新并促进民营经济发展", "content": "扩大开放，提振信心"}],
        market_news=[
            {"标题": "市场反弹，经济预期改善", "摘要": "风险偏好回暖", "发布时间": now - timedelta(minutes=20)},
            {"标题": "未来坏消息", "摘要": "市场暴跌", "发布时间": now + timedelta(hours=1)},
        ],
    )

    assert result["dimensions"]["international"]["data_points"] == 1
    assert result["dimensions"]["international"]["score"] == 58.0
    assert result["dimensions"]["domestic_policy"]["score"] > 50
    assert result["dimensions"]["sentiment"]["score"] > 50
    assert result["dimensions"]["sentiment"]["future_records_ignored"] is True
    assert result["automatic_execution"] is False
    assert result["base_strategy_priority"] is True
    assert result["recommended_exposure"] == 1.0
    assert result["point_in_time"]["historical_replay_ready"] is False


def test_macro_assessment_fails_neutral_when_coverage_is_missing():
    now = datetime(2026, 8, 27, 10, 30)
    result = MacroEnvironmentService.assess_inputs(now=now, breadth={})

    assert result["score"] == 50.0
    assert result["confidence"] == 0.0
    assert result["recommended_exposure"] == 1.0
    assert result["base_strategy_priority"] is True
    assert "覆盖不足" in result["warnings"][0]


def test_macro_recommendation_only_reduces_two_percent_in_high_confidence_extreme_risk():
    assert MacroEnvironmentService._protected_exposure(37.9, 0.65) == 0.98
    assert MacroEnvironmentService._protected_exposure(37.9, 0.64) == 1.0
    assert MacroEnvironmentService._protected_exposure(38.0, 1.0) == 1.0


def test_old_macro_cache_is_rejected(tmp_path):
    now = datetime(2026, 8, 27, 10, 30)
    service = MacroEnvironmentService(tmp_path)
    service.cache_path.write_text(
        json.dumps({"version": "macro-overlay-v1", "fetched_at": now.isoformat()}),
        encoding="utf-8",
    )

    assert service._read_cache(now, timedelta(hours=2)) is None
