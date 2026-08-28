from __future__ import annotations

from datetime import date, datetime
import json

import httpx
import pandas as pd

from app.config import BASE_DIR
from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates, fetch_nasdaq_composite
from optimize_factor_strategy import load_current_history
from research_qlib_lgbm_ranker_v1 import (
    baseline_scores,
    blend_scores,
    build_alpha158_lite,
    fit_expanding_lgbm_predictions,
    normalized_curve,
    topk_dropout_backtest,
)


START = date(2020, 1, 1)
END = date(2026, 8, 25)
SOURCE_PATH = BASE_DIR / "data" / "factor_optimization.json"
CURVE_PATH = BASE_DIR / "reports" / "lgbm_validation_2020_2026_curve.csv"
REPORT_PATH = BASE_DIR / "reports" / "lgbm_validation_2020_2026.md"


def main() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history = history[history["date"] <= pd.Timestamp(END)].copy()

    frame, feature_columns = build_alpha158_lite(history)
    baseline = baseline_scores(frame, START, END)
    predictions, audit = fit_expanding_lgbm_predictions(
        frame, feature_columns, WINNER_CONFIG, START, END
    )
    scores = blend_scores(frame, predictions, baseline, WINNER_CONFIG["baseline_weight"])
    result = topk_dropout_backtest(history, scores, START, END)
    lgbm = normalized_curve(result, "strategy_return").rename("lgbm")
    lgbm.index.name = "date"

    nasdaq_url = (
        "https://api.nasdaq.com/api/quote/COMP/historical"
        f"?assetclass=index&fromdate=2019-12-30&todate={END.isoformat()}&limit=4000"
    )
    nasdaq = fetch_nasdaq_composite(nasdaq_url).set_index("date")["nasdaq_close"]
    common_start = lgbm.index.min()
    base = nasdaq[nasdaq.index <= common_start].iloc[-1]
    nasdaq = nasdaq.div(float(base)).rename("nasdaq")
    curve = pd.concat([lgbm, nasdaq], axis=1).sort_index().ffill().dropna().reset_index()
    curve = curve[curve["date"] >= common_start].copy()
    curve["date"] = curve["date"].dt.strftime("%Y-%m-%d")
    curve.to_csv(CURVE_PATH, index=False, encoding="utf-8-sig", float_format="%.8f")

    lgbm_total = float(curve["lgbm"].iloc[-1] - 1)
    nasdaq_total = float(curve["nasdaq"].iloc[-1] - 1)
    elapsed_years = max((pd.Timestamp(curve["date"].iloc[-1]) - pd.Timestamp(curve["date"].iloc[0])).days / 365.25, 1 / 365.25)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "research_only_not_promoted",
        "window": {"start": curve["date"].iloc[0], "end": curve["date"].iloc[-1]},
        "strategy": {
            "name": "LightGBM LambdaRank Top5",
            "config": WINNER_CONFIG,
            "total_return": lgbm_total,
            "end_value": float(curve["lgbm"].iloc[-1]),
            "annual_return": float((curve["lgbm"].iloc[-1] ** (1 / elapsed_years)) - 1),
        },
        "nasdaq": {
            "name": "Nasdaq Composite",
            "symbol": "COMP",
            "return_type": "price return, excluding dividends",
            "total_return": nasdaq_total,
            "end_value": float(curve["nasdaq"].iloc[-1]),
            "annual_return": float((curve["nasdaq"].iloc[-1] ** (1 / elapsed_years)) - 1),
            "source": nasdaq_url,
        },
        "universe_snapshot_run_id": snapshot_id,
        "loaded_codes": int(history["code"].nunique()),
        "refit_months": audit["refit_months"],
        "curve_data": str(CURVE_PATH.relative_to(BASE_DIR)),
        "warnings": warnings + [
            "两条曲线按共同首个有效交易日归一化为1.00",
            "当前Top50回填存在成分与存续偏差；纳指为价格指数，不含分红",
        ],
    }
    (BASE_DIR / "data" / "lgbm_validation_2020_2026.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_PATH.write_text(
        "\n".join([
            "# LightGBM 与纳斯达克综合指数：2020-2026",
            "",
            f"- 区间：{payload['window']['start']} 至 {payload['window']['end']}",
            f"- LightGBM期末净值：{payload['strategy']['end_value']:.2f}x，年化：{payload['strategy']['annual_return']:.2%}",
            f"- 纳斯达克期末净值：{payload['nasdaq']['end_value']:.2f}x，年化：{payload['nasdaq']['annual_return']:.2%}",
            f"- LightGBM相对纳指累计收益差：{lgbm_total - nasdaq_total:.2%}",
            "- 执行口径：收盘信号、下一交易日开盘、Top5、每10个交易日主调仓、单边12bp",
            "- 说明：研究型曲线，不自动修改生产策略；纳指为价格指数，不含分红、费用、税收和汇率影响",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
