from datetime import date

from app.backtest_data import DemoHistoryProvider
from app.backtest_engine import BacktestConfig, BacktestEngine


def _data():
    codes = [f"{600000 + index:06d}" for index in range(20)]
    history, warnings = DemoHistoryProvider().load(codes, {}, date(2024, 1, 1), date(2025, 12, 31))
    return codes, history, warnings


def test_backtest_produces_delayed_rebalances_and_metrics():
    codes, history, warnings = _data()
    config = BacktestConfig("demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 20, 1_000_000, 12)
    result = BacktestEngine().run(history, config, warnings)

    assert result["data"]["trading_days"] > 400
    assert result["metrics"]["rebalance_count"] > 10
    assert len(result["curves"]) == result["data"]["trading_days"]
    assert result["rebalances"][0]["execution_date"] > result["rebalances"][0]["signal_date"]
    assert len(result["rebalances"][0]["holdings"]) == 5
    assert -1 < result["metrics"]["max_drawdown"] <= 0
    assert result["data"]["execution"] == "收盘生成信号，下一交易日开盘生效"
    assert result["data"]["portfolio_accounting"] == "持仓权重随收益漂移，仅在调仓执行日交易并扣费"
    assert result["metrics"]["rebalance_count"] == len(result["rebalances"])


def test_transaction_cost_reduces_return():
    codes, history, warnings = _data()
    base = BacktestConfig("demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 10, 1_000_000, 0)
    costly = BacktestConfig("demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 10, 1_000_000, 50)
    no_cost = BacktestEngine().run(history, base, warnings)
    with_cost = BacktestEngine().run(history, costly, warnings)

    assert with_cost["metrics"]["total_return"] < no_cost["metrics"]["total_return"]
    assert with_cost["metrics"]["total_cost"] > 0


def test_contrarian_strategy_exposes_its_factor_contract():
    codes, history, warnings = _data()
    config = BacktestConfig(
        "demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 10, 1_000_000, 12, "contrarian"
    )
    result = BacktestEngine().run(history, config, warnings)

    assert result["data"]["strategy"] == "逆向增强"
    assert result["data"]["factor_weights"] == BacktestEngine.STRATEGIES["contrarian"]["weights"]
    assert result["config"]["strategy"] == "contrarian"


def test_interim_equal_weight_adjustment_is_counted_and_costed():
    codes, history, warnings = _data()
    base = BacktestConfig(
        "demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 20, 1_000_000, 12, "contrarian"
    )
    adjusted = BacktestConfig(
        "demo", date(2024, 1, 1), date(2025, 12, 31), codes, 5, 20, 1_000_000, 12,
        "contrarian", "daily_equal", 0.03,
    )
    baseline = BacktestEngine().run(history, base, warnings)
    result = BacktestEngine().run(history, adjusted, warnings)

    assert result["data"]["interim_adjustment_count"] > 0
    assert result["metrics"]["rebalance_count"] > len(result["rebalances"])
    assert result["metrics"]["total_cost"] > baseline["metrics"]["total_cost"]


def test_adaptive_policy_trades_only_after_close_for_next_open():
    codes, history, warnings = _data()
    config = BacktestConfig(
        mode="demo",
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        codes=codes,
        top_n=5,
        rebalance_days=10,
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
        rebalance_policy="adaptive",
        min_hold_days=3,
        rank_buffer=5,
        score_gap=0,
        max_replacements=1,
    )
    result = BacktestEngine().run(history, config, warnings)
    adaptive = [record for record in result["rebalances"] if record["trigger"] != "scheduled"]

    assert adaptive
    assert result["data"]["adaptive_rebalances"] == len(adaptive)
    assert all(record["execution_date"] > record["signal_date"] for record in adaptive)
    assert all(record["reason"] != "保持持仓" for record in adaptive)


def test_trend_volatility_risk_overlay_is_cash_preserving():
    codes, history, warnings = _data()
    config = BacktestConfig(
        mode="demo",
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        codes=codes,
        top_n=5,
        rebalance_days=10,
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
        rebalance_policy="adaptive",
        risk_overlay="trend_volatility",
    )
    result = BacktestEngine().run(history, config, warnings)

    assert result["data"]["risk_overlay"] == "trend_volatility"
    low, high = result["data"]["risk_exposure_range"]
    assert 0.20 <= low <= high <= 1.0
    assert all(0 < record["exposure"] <= 1 for record in result["rebalances"])
    assert all(sum(item["weight"] for item in record["holdings"]) <= 1.0 for record in result["rebalances"])


def test_drawdown_circuit_breaker_halts_and_reenters():
    codes, history, warnings = _data()
    config = BacktestConfig(
        mode="demo",
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        codes=codes,
        top_n=5,
        rebalance_days=10,
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
        rebalance_policy="adaptive",
        drawdown_limit=0.01,
        drawdown_cooldown_days=5,
    )
    result = BacktestEngine().run(history, config, warnings)

    assert result["data"]["risk_halt_count"] > 0
    assert result["data"]["drawdown_cooldown_days"] == 5
    assert result["metrics"]["total_cost"] > 0
