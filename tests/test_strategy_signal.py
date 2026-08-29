from datetime import date

from app.backtest_data import DemoHistoryProvider
from app.config import settings
from app.models import StrategyPick
from app.notifications import NotificationService
from app.strategy_signal import StrategySignalService


def test_strategy_signal_selects_five_equal_weight_picks(tmp_path):
    codes = [f"{600000 + index:06d}" for index in range(12)]
    names = {code: f"测试股票{index + 1}" for index, code in enumerate(codes)}
    history, _ = DemoHistoryProvider().load(codes, names, date(2026, 1, 1), date(2026, 8, 25))
    service = StrategySignalService(tmp_path, top_n=5, rebalance_days=10)

    picks, metadata = service.select_from_history(history, date(2026, 8, 25))

    assert len(picks) == 5
    assert [item.rank for item in picks] == [1, 2, 3, 4, 5]
    assert all(item.target_weight == 0.2 for item in picks)
    assert all(item.score >= picks[index + 1].score for index, item in enumerate(picks[:-1]))
    assert metadata["strategy"] == "逆向增强"
    assert metadata["rebalance_days"] == 10


def test_wecom_message_leads_with_daily_top_five_contract():
    picks = [
        StrategyPick(index, f"60000{index}", f"样例{index}", 80 - index, 10 + index, 0.2, {"中期反转": 0.8}, "中期反转")
        for index in range(1, 6)
    ]
    message = NotificationService(settings)._compact_message(
        "A股量化研究日报｜2026-08-25",
        "市场均衡。",
        [],
        [],
        picks,
        {
            "signal_date": "2026-08-25",
            "rebalance_days": 10,
            "decision": {
                "signal_date": "2026-08-25",
                "action_label": "继续持有",
                "execution_date": "无交易",
                "trade_required": False,
                "reason": "排名与分差均未触发换股",
                "cycle_day": 4,
                "next_scheduled_in": 6,
                "adaptive_used_in_cycle": 0,
                "current_holdings": [
                    {"code": item.code, "name": item.name, "target_weight": 0.2} for item in picks
                ],
                "target_holdings": [
                    {"code": item.code, "name": item.name, "target_weight": 0.2} for item in picks
                ],
                "buys": [],
                "sells": [],
                "rules": {"rebalance_days": 10},
            },
        },
        {
            "score": 42.0,
            "regime": "偏谨慎",
            "recommended_exposure": 0.6,
            "confidence": 0.8,
            "dimensions": {
                "international": {"score": 35.0},
                "domestic_policy": {"score": 52.0},
                "sentiment": {"score": 41.0},
            },
        },
    )

    assert "每日自适应交易决策" in message
    assert "宏观环境评测" in message
    assert "研究建议总仓位 60%" in message
    assert "尚未自动改写交易订单" in message
    assert "信号日：2026-08-25" in message
    assert "今日动作：继续持有" in message
    assert "今日不交易，继续持有" in message
    assert "今日观察 Top5（不等于交易指令）" in message
    assert "目标 20%" in message
    assert "样例5" in message
