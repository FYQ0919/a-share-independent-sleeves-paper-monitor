from dataclasses import replace
from datetime import date
import json

from app.backtest_data import BaoStockHistoryProvider
from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR, settings
from app.storage import Storage
from analyze_strategy import period_summary, summarize_result


START = date(2016, 8, 25)
END = date(2026, 8, 25)
TRAIN_END = date(2021, 12, 31)
VALIDATION_START = date(2022, 1, 1)
OUTPUT_PATH = BASE_DIR / "data" / "top50_decade_backtest.json"


def technology_exposure(result, technology_codes):
    slots = [item for record in result["rebalances"] for item in record["holdings"]]
    if not slots:
        return 0.0
    return sum(item["code"] in technology_codes for item in slots) / len(slots)


def yearly_results(result):
    rows = []
    for year in range(START.year, END.year + 1):
        period_start = START if year == START.year else date(year, 1, 1)
        period_end = END if year == END.year else date(year, 12, 31)
        try:
            row = period_summary(result, period_start, period_end)
        except ValueError:
            continue
        row["year"] = str(year)
        rows.append(row)
    return rows


def main():
    storage = Storage(settings.database_path)
    latest = storage.latest()
    if not latest or len(latest.get("candidates", [])) < 50:
        raise RuntimeError("请先运行一次真实Top50选股")
    candidates = latest["candidates"][:50]
    codes = [item["code"] for item in candidates]
    names = {item["code"]: item["name"] for item in candidates}
    technology_codes = {item["code"] for item in candidates if item.get("is_technology")}
    provider = BaoStockHistoryProvider(settings.cache_dir)
    history, warnings = provider.load(codes, names, START, END)
    loaded_codes = sorted(history["code"].unique().tolist())
    base = BacktestConfig(
        mode="live",
        start_date=START,
        end_date=END,
        codes=loaded_codes,
        top_n=5,
        rebalance_days=10,
        initial_capital=1_000_000,
        cost_bps=12,
        strategy="contrarian",
        rebalance_policy="fixed",
    )
    engine = BacktestEngine()
    feature_frame = engine._features(history)
    engine._features = lambda _: feature_frame
    cases = [("固定10日", base)]
    for min_hold in (3, 5):
        for rank_buffer in (10, 15, 20, 25):
            for score_gap in (15.0, 20.0, 25.0):
                cases.append((
                    f"自适应 H{min_hold} B{rank_buffer} G{score_gap:.0f}",
                    replace(
                        base,
                        rebalance_policy="adaptive",
                        min_hold_days=min_hold,
                        rank_buffer=rank_buffer,
                        score_gap=score_gap,
                        max_replacements=1,
                        entry_rank=3,
                        max_adaptive_per_cycle=1,
                    ),
                ))

    evaluated = []
    results = {}
    for label, config in cases:
        result = engine.run(history, config, warnings)
        results[label] = result
        evaluated.append({
            "label": label,
            "config": {
                "rebalance_policy": config.rebalance_policy,
                "min_hold_days": config.min_hold_days,
                "rank_buffer": config.rank_buffer,
                "score_gap": config.score_gap,
            },
            "full": summarize_result(result, label),
            "training": period_summary(result, START, TRAIN_END),
            "validation": period_summary(result, VALIDATION_START, END),
            "scheduled_rebalances": result["data"]["scheduled_rebalances"],
            "adaptive_rebalances": result["data"]["adaptive_rebalances"],
            "technology_holding_exposure": technology_exposure(result, technology_codes),
        })

    fixed = evaluated[0]
    adaptive = evaluated[1:]
    adaptive.sort(
        key=lambda row: (
            min(row["training"]["excess_return"], row["validation"]["excess_return"]),
            row["validation"]["information_ratio"],
            -abs(row["full"]["max_drawdown"]),
        ),
        reverse=True,
    )
    selected = adaptive[0]
    fixed_result = results[fixed["label"]]
    selected_result = results[selected["label"]]
    storage.save_backtest(fixed_result)
    storage.save_backtest(selected_result)
    payload = {
        "generated_at": selected_result["created_at"],
        "window": {"start": START.isoformat(), "end": END.isoformat()},
        "universe": {
            "requested": len(codes),
            "loaded": len(loaded_codes),
            "technology_reserved": len(technology_codes),
            "technology_codes": sorted(technology_codes),
            "point_in_time_warning": "使用当前Top50回填历史，存在成分、上市与存续偏差",
        },
        "execution": "收盘信号，下一交易日开盘生效，单边成本12bp",
        "selection_rule": "优先最大化训练期与验证期中较低的超额收益，再比较验证期信息比率和回撤",
        "fixed": fixed,
        "selected_adaptive": selected,
        "yearly_fixed": yearly_results(fixed_result),
        "yearly_adaptive": yearly_results(selected_result),
        "adaptive_grid": adaptive,
        "warnings": warnings,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "universe_loaded": len(loaded_codes),
        "fixed": fixed,
        "selected_adaptive": selected,
        "yearly_adaptive": payload["yearly_adaptive"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
