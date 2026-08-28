from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import subprocess

import httpx
import numpy as np
import pandas as pd

from app.config import BASE_DIR, settings


for variable in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(variable, None)


COST_RATE = 12 / 10_000


@dataclass(frozen=True)
class PeriodConfig:
    key: str
    label: str
    start: date
    end: date
    base_curve: Path
    output_stem: Path
    nasdaq_from: date
    status: str


@dataclass(frozen=True)
class OverlayConfig:
    key: str
    label: str
    exposure_map: dict[str, float]
    base_strategy_priority: bool


PERIODS = {
    "validation": PeriodConfig(
        key="validation",
        label="验证集",
        start=date(2021, 1, 1),
        end=date(2023, 12, 31),
        base_curve=BASE_DIR / "reports" / "validation_strategy_vs_nasdaq.csv",
        output_stem=BASE_DIR / "reports" / "macro_validation_performance",
        nasdaq_from=date(2020, 12, 1),
        status="proxy_validation_only",
    ),
    "test": PeriodConfig(
        key="test",
        label="2024年至今已揭示留出集",
        start=date(2024, 1, 1),
        end=date(2026, 8, 25),
        base_curve=BASE_DIR / "reports" / "test_strategy_vs_nasdaq.csv",
        output_stem=BASE_DIR / "reports" / "macro_test_performance",
        nasdaq_from=date(2023, 12, 1),
        status="consumed_holdout_proxy_test",
    ),
}

OVERLAYS = {
    "protected": OverlayConfig(
        key="protected",
        label="基础优先宏观提示层",
        exposure_map={">=38": 1.0, "<38": 0.98},
        base_strategy_priority=True,
    ),
    "legacy": OverlayConfig(
        key="legacy",
        label="旧版宏观仓位覆盖层",
        exposure_map={">=62": 1.0, ">=48": 0.8, ">=38": 0.6, "<38": 0.4},
        base_strategy_priority=False,
    ),
}


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def metrics(curve: pd.Series, dates: pd.Series) -> dict:
    returns = curve.pct_change().dropna()
    elapsed_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    volatility = float(returns.std(ddof=0) * np.sqrt(252))
    annualized_mean = float(returns.mean() * 252)
    drawdown = curve.div(curve.cummax()).sub(1)
    return {
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "total_return": float(curve.iloc[-1] / curve.iloc[0] - 1),
        "annual_return": float((curve.iloc[-1] / curve.iloc[0]) ** (1 / elapsed_years) - 1),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": volatility,
        "sharpe_zero_rate": annualized_mean / volatility if volatility else 0.0,
    }


def load_frozen_strategy_curve(config: PeriodConfig) -> pd.DataFrame:
    if not config.base_curve.exists():
        raise RuntimeError(f"冻结策略曲线不存在: {config.base_curve}")
    frame = pd.read_csv(config.base_curve)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[["date", "strategy", "nasdaq"]].sort_values("date")


def load_period_calendar(config: PeriodConfig) -> pd.DatetimeIndex:
    paths = sorted((settings.cache_dir / "backtest").glob("*_20160328_20260825_qfq.csv"))
    if not paths:
        raise RuntimeError("缺少冻结研究期A股交易日历")
    sample = pd.read_csv(paths[0], usecols=["date"])
    dates = pd.to_datetime(sample["date"], errors="coerce").dropna().drop_duplicates()
    dates = dates[(dates >= pd.Timestamp(config.start)) & (dates <= pd.Timestamp(config.end))]
    return pd.DatetimeIndex(sorted(dates))


def load_frozen_universe_breadth(config: PeriodConfig) -> pd.DataFrame:
    paths = sorted((settings.cache_dir / "backtest").glob("*_20160328_20260825_qfq.csv"))
    rows = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["date", "code", "close"], dtype={"code": str})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame["return"] = frame["close"].pct_change()
        rows.append(frame[["date", "code", "return"]])
    history = pd.concat(rows, ignore_index=True).dropna(subset=["return"])
    history = history[
        (history["date"] >= pd.Timestamp(config.start) - pd.Timedelta(days=10))
        & (history["date"] <= pd.Timestamp(config.end))
    ]
    breadth = history.groupby("date").agg(
        up=("return", lambda values: int((values > 0).sum())),
        total=("return", "count"),
    ).reset_index()
    breadth["sentiment_score"] = breadth["up"].div(breadth["total"]).mul(100).clip(0, 100)
    return breadth


def fetch_nasdaq(config: PeriodConfig) -> pd.DataFrame:
    url = (
        "https://api.nasdaq.com/api/quote/COMP/historical"
        f"?assetclass=index&fromdate={config.nasdaq_from:%Y-%m-%d}"
        f"&todate={config.end:%Y-%m-%d}&limit=2000"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
    rows = response.json()["data"]["tradesTable"]["rows"]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["close"] = pd.to_numeric(
        frame["close"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    frame = frame[["date", "close"]].dropna().drop_duplicates("date").sort_values("date")
    frame["international_score"] = frame["close"].pct_change().mul(800).add(50).clip(0, 100)
    return frame.dropna(subset=["international_score"])


def fetch_domestic_policy() -> pd.DataFrame:
    import akshare as ak

    pmi = ak.macro_china_pmi().copy()
    pmi_rows = []
    for _, row in pmi.iterrows():
        matched = re.match(r"(\d{4})年(\d{2})月份", str(row.get("月份", "")))
        if not matched:
            continue
        year, month = map(int, matched.groups())
        period = pd.Period(f"{year:04d}-{month:02d}", freq="M")
        available_date = (period + 1).start_time.normalize()
        values = [
            pd.to_numeric(row.get("制造业-指数"), errors="coerce"),
            pd.to_numeric(row.get("非制造业-指数"), errors="coerce"),
        ]
        values = [float(value) for value in values if pd.notna(value)]
        if values:
            pmi_rows.append({
                "available_date": available_date,
                "pmi_score": clamp(50 + np.mean([value - 50 for value in values]) * 7),
                "pmi_values": "/".join(f"{value:.1f}" for value in values),
            })
    pmi_frame = pd.DataFrame(pmi_rows).sort_values("available_date")

    lpr = ak.macro_china_lpr().copy()
    lpr["available_date"] = pd.to_datetime(lpr["TRADE_DATE"], errors="coerce")
    lpr["LPR1Y"] = pd.to_numeric(lpr["LPR1Y"], errors="coerce")
    lpr = lpr.dropna(subset=["available_date", "LPR1Y"]).sort_values("available_date")
    lpr["lpr_change"] = lpr["LPR1Y"].diff()
    lpr["lpr_score"] = lpr["lpr_change"].mul(-100).add(50).clip(0, 100)
    lpr = lpr.dropna(subset=["lpr_score"])[["available_date", "LPR1Y", "lpr_score"]]

    events = pd.concat([
        pmi_frame.assign(kind="pmi"),
        lpr.assign(kind="lpr"),
    ], ignore_index=True, sort=False).sort_values("available_date")
    return events


def asof_column(target_dates: pd.Series, source: pd.DataFrame, value: str, strict: bool) -> pd.DataFrame:
    left = pd.DataFrame({"date": pd.to_datetime(target_dates)}).sort_values("date")
    source_date = "date" if "date" in source.columns else "available_date"
    right = source[[source_date, value]].dropna().sort_values(source_date).rename(
        columns={source_date: "source_date"}
    )
    return pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="source_date",
        direction="backward",
        allow_exact_matches=not strict,
    )


def build_macro_signals(
    trading_dates: pd.DatetimeIndex,
    config: PeriodConfig,
    overlay: OverlayConfig,
) -> pd.DataFrame:
    dates = pd.Series(trading_dates, name="date")
    international = asof_column(dates, fetch_nasdaq(config), "international_score", strict=True)
    breadth = asof_column(
        dates,
        load_frozen_universe_breadth(config),
        "sentiment_score",
        strict=True,
    )
    events = fetch_domestic_policy()
    pmi = asof_column(dates, events[events["kind"] == "pmi"], "pmi_score", strict=False)
    lpr = asof_column(dates, events[events["kind"] == "lpr"], "lpr_score", strict=False)

    signals = pd.DataFrame({"date": trading_dates})
    signals["international_score"] = international["international_score"].fillna(50).to_numpy()
    signals["international_source_date"] = international["source_date"].to_numpy()
    signals["pmi_score"] = pmi["pmi_score"].fillna(50).to_numpy()
    signals["pmi_source_date"] = pmi["source_date"].to_numpy()
    signals["lpr_score"] = lpr["lpr_score"].fillna(50).to_numpy()
    signals["lpr_source_date"] = lpr["source_date"].to_numpy()
    signals["policy_text_score"] = 50.0
    signals["domestic_policy_score"] = signals[["pmi_score", "lpr_score", "policy_text_score"]].mean(axis=1)
    signals["breadth_score"] = breadth["sentiment_score"].fillna(50).to_numpy()
    signals["breadth_source_date"] = breadth["source_date"].to_numpy()
    signals["news_sentiment_score"] = 50.0
    signals["sentiment_score"] = signals[["breadth_score", "news_sentiment_score"]].mean(axis=1)
    signals["macro_score"] = (
        signals["international_score"] * 0.40
        + signals["domestic_policy_score"] * 0.35
        + signals["sentiment_score"] * 0.25
    )
    if overlay.key == "protected":
        signals["exposure"] = np.where(signals["macro_score"] < 38, 0.98, 1.0)
    else:
        signals["exposure"] = np.select(
            [
                signals["macro_score"] >= 62,
                signals["macro_score"] >= 48,
                signals["macro_score"] >= 38,
            ],
            [1.0, 0.8, 0.6],
            default=0.4,
        )
    for column in ("international_source_date", "breadth_source_date"):
        valid = signals[column].notna()
        if not bool((signals.loc[valid, column] < signals.loc[valid, "date"]).all()):
            raise RuntimeError(f"{column} 存在非严格滞后记录")
    for column in ("pmi_source_date", "lpr_source_date"):
        valid = signals[column].notna()
        if not bool((signals.loc[valid, column] <= signals.loc[valid, "date"]).all()):
            raise RuntimeError(f"{column} 存在未来记录")
    return signals


def apply_overlay(base: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    curve = base.merge(signals, on="date", how="inner").sort_values("date").reset_index(drop=True)
    curve["base_return"] = curve["strategy"].pct_change().fillna(0)
    curve["exposure_turnover"] = curve["exposure"].diff().abs().fillna(0)
    curve["overlay_cost"] = curve["exposure_turnover"] * COST_RATE
    curve["macro_gross_return"] = curve["base_return"] * curve["exposure"]
    curve["macro_gross_strategy"] = (1 + curve["macro_gross_return"]).cumprod()
    curve["macro_return"] = curve["base_return"] * curve["exposure"] - curve["overlay_cost"]
    curve["macro_strategy"] = (1 + curve["macro_return"]).cumprod()
    curve["strategy"] = curve["strategy"].div(curve["strategy"].iloc[0])
    curve["nasdaq"] = curve["nasdaq"].div(curve["nasdaq"].iloc[0])
    curve["macro_strategy"] = curve["macro_strategy"].div(curve["macro_strategy"].iloc[0])
    curve["macro_gross_strategy"] = curve["macro_gross_strategy"].div(curve["macro_gross_strategy"].iloc[0])
    return curve


def render_plot(
    config: PeriodConfig,
    overlay: OverlayConfig,
    csv_path: Path,
    path: Path,
) -> bool:
    renderer = BASE_DIR / "scripts" / "render_macro_validation.ps1"
    subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(renderer),
            "-CsvPath",
            str(csv_path),
            "-OutputPath",
            str(path),
            "-Title",
            f"{config.label}：基础策略保护效果",
            "-Subtitle",
            f"{config.start} 至 {config.end} | {overlay.label} | 仓位变化另计单边12bp",
            "-AxisTitle",
            f"累计净值（{config.start} = 1）",
            "-SeriesLabel",
            f"{overlay.label}（含成本）",
        ],
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回测固定宏观覆盖层并与纳斯达克综合指数对比")
    parser.add_argument("--period", choices=sorted(PERIODS), default="validation")
    parser.add_argument("--overlay", choices=sorted(OVERLAYS), default="protected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PERIODS[args.period]
    overlay = OVERLAYS[args.overlay]
    base = load_frozen_strategy_curve(config)
    calendar = load_period_calendar(config)
    anchor = base[base["date"] <= pd.Timestamp(config.start)].tail(1)
    validation = base[base["date"].isin(calendar)]
    base = pd.concat([anchor, validation], ignore_index=True).drop_duplicates("date").sort_values("date")
    expected_last_session = calendar.max().date()
    if base.empty or base["date"].max().date() != expected_last_session:
        raise RuntimeError(
            "冻结策略曲线未覆盖区间内最后交易日，"
            f"预期={expected_last_session}，实际={base['date'].max().date() if not base.empty else None}"
        )
    signals = build_macro_signals(pd.DatetimeIndex(base["date"]), config, overlay)
    curve = apply_overlay(base, signals)

    base_metrics = metrics(curve["strategy"], curve["date"])
    macro_metrics = metrics(curve["macro_strategy"], curve["date"])
    macro_gross_metrics = metrics(curve["macro_gross_strategy"], curve["date"])
    nasdaq_metrics = metrics(curve["nasdaq"], curve["date"])
    exposure_distribution = {
        f"{int(level * 100)}%": int(count)
        for level, count in curve["exposure"].value_counts().sort_index(ascending=False).items()
    }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": config.status,
        "overlay": overlay.key,
        "window": {"start": config.start.isoformat(), "end": config.end.isoformat()},
        "execution": "基础策略次日开盘执行并含12bp成本；宏观仓位变化另计12bp",
        "macro_contract": {
            "weights": {"international": 0.40, "domestic_policy": 0.35, "sentiment": 0.25},
            "exposure_map": overlay.exposure_map,
            "base_strategy_priority": overlay.base_strategy_priority,
            "maximum_macro_reduction": 0.02 if overlay.base_strategy_priority else 0.60,
            "production_automatic_execution": False,
            "international": "严格使用交易日前一可用纳斯达克综合指数日收益",
            "domestic_policy": "PMI按下一月份首日可用，LPR按发布日期可用；历史政策文本缺失记50分",
            "sentiment": "严格使用前一A股交易日冻结股票池市场宽度；历史新闻文本缺失记50分",
            "thresholds_tuned_on_validation": False,
            "thresholds_tuned_on_2024_plus": False,
        },
        "metrics": {
            "base_strategy": base_metrics,
            "macro_proxy_strategy": macro_metrics,
            "macro_proxy_before_overlay_cost": macro_gross_metrics,
            "nasdaq_composite": nasdaq_metrics,
            "macro_minus_base": {
                "total_return": macro_metrics["total_return"] - base_metrics["total_return"],
                "annual_return": macro_metrics["annual_return"] - base_metrics["annual_return"],
                "max_drawdown": macro_metrics["max_drawdown"] - base_metrics["max_drawdown"],
                "sharpe_zero_rate": macro_metrics["sharpe_zero_rate"] - base_metrics["sharpe_zero_rate"],
            },
        },
        "diagnostics": {
            "sessions": int(len(curve) - 1),
            "average_macro_score": float(curve["macro_score"].mean()),
            "average_exposure": float(curve["exposure"].mean()),
            "exposure_distribution": exposure_distribution,
            "exposure_turnover": float(curve["exposure_turnover"].sum()),
            "additional_cost": float(curve["overlay_cost"].sum()),
            "international_mean": float(curve["international_score"].mean()),
            "domestic_policy_mean": float(curve["domestic_policy_score"].mean()),
            "sentiment_mean": float(curve["sentiment_score"].mean()),
        },
        "limitations": [
            "历史政策新闻和财经舆情没有逐日当时可见快照，本次固定为中性50分。",
            "市场宽度来自当前冻结股票池历史回填，仍有成分与存续偏差。",
            "基础价格为当前版本前复权数据，不满足严格as-of复权认证。",
            (
                "2024-2026区间此前已被查看，本次运行后属于已消耗留出集，"
                "不能再称为未触碰样本外测试集。"
                if config.key == "test"
                else "2021-2023是既有模型选择验证集，不是新的未触碰样本外测试集。"
            ),
            "完整宏观策略必须从2026-08-27开始积累实时快照后做前向检验。",
        ],
    }

    output_stem = config.output_stem
    if overlay.key == "protected":
        output_stem = output_stem.with_name(output_stem.name.replace("macro_", "macro_protected_", 1))
    csv_path = output_stem.with_suffix(".csv")
    json_path = output_stem.with_suffix(".json")
    md_path = output_stem.with_suffix(".md")
    png_path = output_stem.with_suffix(".png")
    output = curve.copy()
    for column in output.columns:
        if column == "date" or column.endswith("source_date"):
            output[column] = pd.to_datetime(output[column], errors="coerce").dt.strftime("%Y-%m-%d")
    output.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.8f")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_plot(config, overlay, csv_path, png_path)
    delta = payload["metrics"]["macro_minus_base"]
    md_path.write_text(
        f"""# {overlay.label}{config.label}回测

- 区间：`{config.start}` 至 `{config.end}`
- 状态：**{overlay.label}，生产订单仍不受宏观评测自动影响**
- 基础策略年化：{base_metrics['annual_return']:.2%}，夏普：{base_metrics['sharpe_zero_rate']:.3f}，最大回撤：{base_metrics['max_drawdown']:.2%}
- 宏观代理策略年化：{macro_metrics['annual_return']:.2%}，夏普：{macro_metrics['sharpe_zero_rate']:.3f}，最大回撤：{macro_metrics['max_drawdown']:.2%}
- 宏观代理毛值（不计额外仓位成本）年化：{macro_gross_metrics['annual_return']:.2%}，夏普：{macro_gross_metrics['sharpe_zero_rate']:.3f}，最大回撤：{macro_gross_metrics['max_drawdown']:.2%}
- 年化变化：{delta['annual_return']:+.2%}，夏普变化：{delta['sharpe_zero_rate']:+.3f}，回撤变化：{delta['max_drawdown']:+.2%}
- 平均总仓位：{payload['diagnostics']['average_exposure']:.1%}
- 宏观额外成本：{payload['diagnostics']['additional_cost']:.2%}

## 时点限制

""" + "\n".join(f"- {item}" for item in payload["limitations"]),
        encoding="utf-8",
    )
    print(json.dumps({
        "overlay": overlay.key,
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "base": base_metrics,
        "macro": macro_metrics,
        "macro_gross": macro_gross_metrics,
        "delta": delta,
        "diagnostics": payload["diagnostics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
