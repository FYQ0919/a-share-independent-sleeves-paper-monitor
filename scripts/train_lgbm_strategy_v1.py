from __future__ import annotations

from datetime import timedelta
import json

from app.config import BASE_DIR, settings
from app.lgbm_strategy import (
    WINNER_CONFIG,
    LgbmStrategyModel,
    file_sha256,
    train_frozen_model,
)
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history

import pandas as pd


RESEARCH_RESULT = BASE_DIR / "data" / "qlib_lgbm_ranker_research_v1.json"
SOURCE_OPTIMIZATION = BASE_DIR / "data" / "factor_optimization.json"
SIGNAL_OUTPUT = BASE_DIR / "data" / "lgbm_forward_signal_v1.json"
REPORT_OUTPUT = BASE_DIR / "reports" / "lgbm_forward_strategy_v1.md"


def display_name(value, code) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return str(code)
    return str(value)


def main() -> None:
    research = json.loads(RESEARCH_RESULT.read_text(encoding="utf-8"))
    if research.get("model_backend") != "lightgbm" or not research.get("test_gate", {}).get("passed"):
        raise RuntimeError("LGBM 研究门禁未通过，拒绝训练每日策略模型")
    if research.get("selection_winner", {}).get("config") != WINNER_CONFIG:
        raise RuntimeError("研究冠军参数与冻结每日策略参数不一致")

    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, _, warnings = load_current_history(universe)
    latest_date = history["date"].max().date()
    training_as_of = latest_date + timedelta(days=1)
    metadata = train_frozen_model(
        history,
        settings.lgbm_model_dir,
        training_as_of,
        metadata_extra={
            "universe_snapshot_run_id": snapshot_id,
            "universe_codes": sorted(history["code"].astype(str).unique().tolist()),
            "research_result": str(RESEARCH_RESULT.relative_to(BASE_DIR)),
            "research_result_sha256": file_sha256(RESEARCH_RESULT),
            "research_selection_winner": research["selection_winner"]["name"],
            "research_test_gate": research["test_gate"],
            "research_2020_performance": research["winner_test"],
            "data_warnings": warnings,
        },
    )

    model = LgbmStrategyModel(settings.lgbm_model_dir)
    ranked, signal_metadata = model.rank(history, latest_date)
    top = ranked.head(5)
    picks = [
        {
            "rank": int(row["rank"]),
            "code": str(row["code"]),
            "name": display_name(row.get("name"), row["code"]),
            "price": round(float(row["close"]), 2),
            "score": round(float(row["score"]), 2),
            "lgbm_rank": round(float(row["lgbm_rank"]) * 100, 2),
            "baseline_rank": round(float(row["baseline_rank"]) * 100, 2),
            "reason": str(row["lgbm_reason"]),
            "target_weight": 0.2,
        }
        for _, row in top.iterrows()
    ]
    signal_payload = {
        "status": "forward_paper_ready",
        "model_version": metadata["model_version"],
        "model_sha256": metadata["model_sha256"],
        "signal": signal_metadata,
        "picks": picks,
        "production_change": False,
        "warnings": metadata["warnings"],
    }
    SIGNAL_OUTPUT.write_text(
        json.dumps(signal_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_OUTPUT.write_text(
        "\n".join([
            "# LGBM LambdaRank Top5 前向策略 v1",
            "",
            "- 状态：冻结模型已生成，可用于前向模拟盘",
            f"- 模型版本：`{metadata['model_version']}`",
            f"- 模型哈希：`{metadata['model_sha256']}`",
            f"- 训练信号区间：{metadata['training_signal_start']} 至 {metadata['training_signal_end']}",
            f"- 最大成熟标签结束日：{metadata['max_training_label_end_date']}",
            f"- 训练样本：{metadata['training_rows']} 行 / {metadata['training_dates']} 个交易日",
            f"- 最新信号日：{signal_metadata['signal_date']}",
            "- 执行：收盘生成信号，下一交易日开盘执行",
            "- 组合：Top5 等权，10日主调仓，自适应换股规则沿用",
            "- 生产修改：否；影子观察设置 `STRATEGY_MODEL_BACKEND=lgbm_shadow`，通过126个前向交易日后再评估是否启用 `lgbm_active`",
            "",
            "## 最新观察 Top5",
            "",
            *[
                f"- {item['rank']}. {item['name']}（{item['code']}）{item['score']:.2f} 分；{item['reason']}"
                for item in picks
            ],
            "",
            "## 风险边界",
            "",
            *[f"- {warning}" for warning in metadata["warnings"]],
        ]),
        encoding="utf-8",
    )
    print(json.dumps({
        "model_dir": str(settings.lgbm_model_dir),
        "metadata": metadata,
        "signal_output": str(SIGNAL_OUTPUT),
        "report": str(REPORT_OUTPUT),
        "picks": picks,
        "production_change": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
