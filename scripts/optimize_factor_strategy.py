from __future__ import annotations

from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest_engine import BacktestConfig, BacktestEngine
from app.config import BASE_DIR, settings
from app.storage import Storage
from analyze_strategy import period_summary, summarize_result


START = date(2016, 8, 25)
TRAIN_END = date(2020, 12, 31)
VALIDATION_START = date(2021, 1, 1)
VALIDATION_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
END = date(2026, 8, 25)
OUTPUT_PATH = BASE_DIR / "data" / "factor_optimization.json"
PROTOCOL_PATH = BASE_DIR / "data" / "research_protocol.json"
CACHE_DIR = settings.cache_dir / "backtest"

FACTOR_NAMES = [
    "medium_reversal",
    "inefficiency",
    "liquidity_cooling",
    "long_momentum",
    "trend_quality",
    "low_volatility",
    "downside_quality",
    "short_reversal",
    "price_position",
    "intraday_quality",
    "risk",
    "value",
]

MODELS = {
    "current_contrarian": {
        "medium_reversal": 0.35,
        "inefficiency": 0.25,
        "liquidity_cooling": 0.20,
        "risk": 0.10,
        "value": 0.10,
    },
    "balanced_reversal": {
        "medium_reversal": 0.20,
        "short_reversal": 0.10,
        "long_momentum": 0.15,
        "trend_quality": 0.10,
        "low_volatility": 0.20,
        "downside_quality": 0.10,
        "value": 0.10,
        "liquidity_cooling": 0.05,
    },
    "quality_momentum": {
        "long_momentum": 0.30,
        "trend_quality": 0.15,
        "low_volatility": 0.20,
        "downside_quality": 0.10,
        "short_reversal": 0.05,
        "value": 0.10,
        "risk": 0.10,
    },
    "defensive_momentum": {
        "long_momentum": 0.25,
        "trend_quality": 0.10,
        "low_volatility": 0.30,
        "downside_quality": 0.15,
        "value": 0.15,
        "short_reversal": 0.05,
    },
    "momentum_value": {
        "long_momentum": 0.35,
        "trend_quality": 0.15,
        "value": 0.20,
        "low_volatility": 0.15,
        "risk": 0.10,
        "short_reversal": 0.05,
    },
    "low_vol_value": {
        "low_volatility": 0.30,
        "downside_quality": 0.15,
        "value": 0.25,
        "long_momentum": 0.15,
        "short_reversal": 0.10,
        "liquidity_cooling": 0.05,
    },
    "reversal_quality": {
        "medium_reversal": 0.25,
        "short_reversal": 0.15,
        "low_volatility": 0.25,
        "downside_quality": 0.10,
        "value": 0.15,
        "liquidity_cooling": 0.10,
    },
    "trend_quality": {
        "long_momentum": 0.30,
        "trend_quality": 0.25,
        "price_position": 0.15,
        "low_volatility": 0.15,
        "value": 0.10,
        "risk": 0.05,
    },
    "price_strength": {
        "long_momentum": 0.25,
        "trend_quality": 0.20,
        "price_position": 0.25,
        "low_volatility": 0.15,
        "value": 0.10,
        "short_reversal": 0.05,
    },
    "intraday_quality": {
        "long_momentum": 0.20,
        "trend_quality": 0.15,
        "intraday_quality": 0.20,
        "low_volatility": 0.20,
        "downside_quality": 0.10,
        "value": 0.10,
        "short_reversal": 0.05,
    },
    "contrarian_quality_10": {
        "medium_reversal": 0.33,
        "inefficiency": 0.23,
        "liquidity_cooling": 0.19,
        "low_volatility": 0.05,
        "risk": 0.10,
        "value": 0.10,
    },
    "contrarian_quality_20": {
        "medium_reversal": 0.30,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.18,
        "low_volatility": 0.08,
        "downside_quality": 0.02,
        "risk": 0.10,
        "value": 0.10,
    },
    "contrarian_value_tilt": {
        "medium_reversal": 0.32,
        "inefficiency": 0.23,
        "liquidity_cooling": 0.17,
        "low_volatility": 0.05,
        "risk": 0.08,
        "value": 0.15,
    },
    "contrarian_low_vol_tilt": {
        "medium_reversal": 0.32,
        "inefficiency": 0.22,
        "liquidity_cooling": 0.17,
        "low_volatility": 0.12,
        "risk": 0.08,
        "value": 0.09,
    },
    "contrarian_stable_ic": {
        "medium_reversal": 0.30,
        "inefficiency": 0.20,
        "liquidity_cooling": 0.15,
        "low_volatility": 0.10,
        "downside_quality": 0.05,
        "risk": 0.05,
        "value": 0.15,
    },
    "contrarian_value_quality": {
        "medium_reversal": 0.35,
        "inefficiency": 0.20,
        "liquidity_cooling": 0.15,
        "low_volatility": 0.05,
        "risk": 0.10,
        "value": 0.15,
    },
    "contrarian_low_vol_quality": {
        "medium_reversal": 0.35,
        "inefficiency": 0.20,
        "liquidity_cooling": 0.15,
        "low_volatility": 0.15,
        "risk": 0.10,
        "value": 0.05,
    },
    "contrarian_less_liquidity": {
        "medium_reversal": 0.35,
        "inefficiency": 0.25,
        "liquidity_cooling": 0.10,
        "low_volatility": 0.08,
        "risk": 0.10,
        "value": 0.12,
    },
    "contrarian_no_liquidity": {
        "medium_reversal": 0.40,
        "inefficiency": 0.25,
        "low_volatility": 0.10,
        "risk": 0.10,
        "value": 0.15,
    },
}


def load_current_history(candidates=None):
    storage = Storage(settings.database_path)
    if candidates is None:
        eligible = [
            run for run in storage.history(200)
            if len(run.get("candidates", [])) >= 50
        ]
        latest = max(eligible, key=lambda run: run["created_at"]) if eligible else None
        if not latest:
            raise RuntimeError("请先运行一次真实Top50选股")
        candidates = latest["candidates"][:50]
    elif len(candidates) < 50:
        raise RuntimeError("冻结研究股票池不足50只")
    else:
        candidates = candidates[:50]
    requested_codes = [item["code"] for item in candidates]
    frames = []
    warnings = []
    for code in requested_codes:
        path = CACHE_DIR / f"{code}_{START.isoformat()}_{END.isoformat()}_qfq.csv"
        matches = sorted(
            CACHE_DIR.glob(f"{code}_*_qfq.csv"),
            key=lambda item: item.stat().st_size,
            reverse=True,
        )
        source = path if path.exists() else (matches[0] if matches else None)
        if source is None:
            warnings.append(f"{code} 没有历史缓存")
            continue
        frame = pd.read_csv(source, dtype={"code": str})
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame["date"] >= pd.Timestamp(START)) & (frame["date"] <= pd.Timestamp(END))]
        if len(frame) < 65:
            warnings.append(f"{code} 历史数据不足，已跳过")
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError("没有可用历史缓存")
    history = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["code", "date"])
        .drop_duplicates(["code", "date"], keep="last")
        .reset_index(drop=True)
    )
    return history, requested_codes, warnings


def valid_cross(engine, frame, signal_date):
    cross = frame[frame["date"] == signal_date].copy()
    cross = cross[
        (cross["tradestatus"] == 1)
        & (cross["is_st"] == 0)
        & (cross["close"] > 1)
        & (cross["amount_20"] >= 10_000_000)
    ].dropna(subset=[
        "ret_5", "ret_20", "ret_60", "volatility_20", "amount_20",
        "amount_60", "absolute_return_60", "ma_20",
    ])
    return engine._score(cross, {"risk": 1.0}) if not cross.empty else cross


def factor_diagnostics(engine, frame):
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))]
    opens = frame.pivot(index="date", columns="code", values="open").reindex(dates)
    forward = opens.shift(-11).div(opens.shift(-1)).sub(1)
    rows = []
    for signal_date in dates[::10]:
        cross = valid_cross(engine, frame, signal_date)
        if len(cross) < 8:
            continue
        future = forward.loc[signal_date]
        for factor in FACTOR_NAMES:
            joined = pd.concat([
                cross.set_index("code")[f"{factor}_score"],
                future.rename("forward"),
            ], axis=1).dropna()
            if len(joined) < 8:
                continue
            ranked = joined.rank(pct=True)
            rows.append({
                "date": signal_date,
                "factor": factor,
                "ic": ranked.iloc[:, 0].corr(ranked.iloc[:, 1]),
            })
    values = pd.DataFrame(rows)
    periods = {
        "training": (START, TRAIN_END),
        "validation": (VALIDATION_START, VALIDATION_END),
        "test": (TEST_START, END),
    }
    output = []
    for factor, group in values.groupby("factor"):
        result = {"factor": factor, "observations": int(group["ic"].notna().sum())}
        means = []
        for label, (period_start, period_end) in periods.items():
            sample = group[
                (group["date"] >= pd.Timestamp(period_start))
                & (group["date"] <= pd.Timestamp(period_end))
            ]["ic"].dropna()
            result[label] = {
                "mean_ic": float(sample.mean()) if not sample.empty else 0.0,
                "positive_rate": float((sample > 0).mean()) if not sample.empty else 0.0,
                "observations": int(len(sample)),
            }
            means.append(result[label]["mean_ic"])
        result["worst_period_ic"] = min(means)
        output.append(result)
    return sorted(output, key=lambda item: item["worst_period_ic"], reverse=True)


def evaluate_periods(result):
    return {
        "training": period_summary(result, START, TRAIN_END),
        "validation": period_summary(result, VALIDATION_START, VALIDATION_END),
        "test": period_summary(result, TEST_START, END),
        "full": summarize_result(result, result["data"]["strategy"]),
    }


def robustness_key(row):
    train = row["periods"]["training"]
    validation = row["periods"]["validation"]
    return (
        min(train["sharpe"], validation["sharpe"]),
        min(train["annual_return"], validation["annual_return"]),
        min(train["information_ratio"], validation["information_ratio"]),
        -abs(validation["max_drawdown"]),
    )


def validate_frozen_candidate_manifest(models):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    canonical = json.dumps(models, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected = protocol["candidate_manifest_sha256"]
    if actual != expected:
        raise RuntimeError(
            "留出集揭盲后候选集已改变；请先版本化 research_protocol.json，不得直接重选。"
        )
    return protocol, actual


def main():
    history, requested_codes, warnings = load_current_history()
    loaded_codes = sorted(history["code"].unique().tolist())
    engine = BacktestEngine()
    feature_frame = engine._features(history)
    engine._features = lambda _: feature_frame
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
        rebalance_policy="adaptive",
        min_hold_days=5,
        rank_buffer=20,
        score_gap=15,
        max_replacements=1,
        entry_rank=3,
        max_adaptive_per_cycle=1,
    )
    comparisons = []
    model_results = {}
    research_models = {
        name: weights
        for name, weights in MODELS.items()
        if name == "current_contrarian" or name.startswith("contrarian_")
    }
    protocol, candidate_manifest = validate_frozen_candidate_manifest(research_models)
    for name, weights in research_models.items():
        engine._factor_weights_override = weights
        result = engine.run(history, base, warnings)
        model_results[name] = result
        comparisons.append({
            "name": name,
            "weights": weights,
            "periods": evaluate_periods(result),
        })
    comparisons.sort(key=robustness_key, reverse=True)
    selected = comparisons[0]
    selected_result = model_results[selected["name"]]
    payload = {
        "generated_at": selected_result["created_at"],
        "window": {
            "start": START.isoformat(),
            "training_end": TRAIN_END.isoformat(),
            "validation_start": VALIDATION_START.isoformat(),
            "validation_end": VALIDATION_END.isoformat(),
            "test_start": TEST_START.isoformat(),
            "end": END.isoformat(),
        },
        "universe": {"requested": len(requested_codes), "loaded": len(loaded_codes)},
        "execution": "收盘信号，下一交易日开盘生效，单边成本12bp",
        "selection_rule": "仅按训练期和验证期的最差夏普、最差年化、最差信息比率排序；2024年后留出测试不参与选择",
        "research_protocol": {
            "candidate_manifest_sha256": candidate_manifest,
            "holdout_status": protocol["holdout"]["status"],
            "holdout_may_be_used_for_reselection": protocol["holdout"]["may_be_used_for_reselection"],
            "forward_paper_start": protocol["forward_paper"]["start"],
        },
        "factor_diagnostics": factor_diagnostics(engine, feature_frame),
        "selected": selected,
        "comparisons": comparisons,
        "warnings": warnings + ["使用当前Top50回填历史，存在成分、上市与存续偏差"],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "selected": selected,
        "top_factor_diagnostics": payload["factor_diagnostics"][:6],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
