from typing import Dict, List

import numpy as np
import pandas as pd

from app.models import Candidate, Sector


REQUIRED_COLUMNS = {
    "code", "name", "price", "pct_change", "change_5m", "pct_60d", "pct_ytd",
    "turnover_rate", "volume_ratio", "amount", "amplitude", "pe", "pb",
    "market_cap", "industry", "concept",
}

TECHNOLOGY_KEYWORDS = (
    "半导体", "芯片", "集成电路", "电子", "计算机", "软件", "通信",
    "人工智能", "机器人", "自动化", "云计算", "数据", "信息技术",
    "光学", "算力", "国产替代",
)


def _board(code: str) -> str:
    value = str(code).zfill(6)
    if value.startswith(("688", "689")):
        return "科创板"
    if value.startswith(("300", "301")):
        return "创业板"
    if value.startswith(("4", "8")):
        return "北交所"
    return "沪市主板" if value.startswith(("6", "9")) else "深市主板"


def _technology_mask(frame: pd.DataFrame) -> pd.Series:
    text = (
        frame["industry"].fillna("").astype(str)
        + " "
        + frame["concept"].fillna("").astype(str)
    )
    keyword_pattern = "|".join(TECHNOLOGY_KEYWORDS)
    star_market = frame["code"].astype(str).str.zfill(6).str.startswith(("688", "689"))
    return star_market | text.str.contains(keyword_pattern, regex=True, na=False)


def _rank(series: pd.Series, higher_is_better: bool = True, neutral: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() < 2:
        return pd.Series(neutral, index=series.index, dtype=float)
    ranked = numeric.rank(pct=True, ascending=higher_is_better)
    return ranked.fillna(neutral).clip(0, 1)


def _optimal_range(series: pd.Series, target: float, width: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(target)
    return (1 - (numeric - target).abs() / width).clip(0, 1)


def _sector_score_map(sectors: List[Sector], kind: str) -> Dict[str, float]:
    selected = [item for item in sectors if item.kind == kind]
    if not selected:
        return {}
    frame = pd.DataFrame({"name": [item.name for item in selected], "change": [item.pct_change for item in selected]})
    frame["score"] = _rank(frame["change"])
    return dict(zip(frame["name"], frame["score"]))


class SelectionEngine:
    def __init__(
        self,
        min_amount: float,
        min_market_cap: float,
        weights: Dict[str, float],
        technology_reserve: int = 0,
    ):
        self.min_amount = min_amount
        self.min_market_cap = min_market_cap
        self.weights = weights
        self.technology_reserve = max(0, technology_reserve)

    def select(self, stocks: pd.DataFrame, sectors: List[Sector], top_n: int) -> List[Candidate]:
        missing = REQUIRED_COLUMNS - set(stocks.columns)
        if missing:
            raise ValueError(f"行情数据缺少字段: {', '.join(sorted(missing))}")
        frame = stocks.copy()
        numeric_columns = REQUIRED_COLUMNS - {"code", "name", "industry", "concept"}
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        valid_name = ~frame["name"].astype(str).str.upper().str.contains(r"ST|退", regex=True, na=False)
        liquid = frame["amount"].fillna(0) >= self.min_amount
        sizable = frame["market_cap"].fillna(0) >= self.min_market_cap
        tradable = (frame["price"].fillna(0) > 1) & frame["pct_change"].between(-19.5, 19.5)
        frame = frame[valid_name & liquid & sizable & tradable].copy()
        if frame.empty:
            return []

        # Cross-sectional scores. These are transparent ranking signals, not a return forecast.
        chase_penalty = (1 - (frame["pct_change"].abs() - 5).clip(lower=0) / 12).clip(0.25, 1)
        momentum = (
            _rank(frame["pct_60d"]) * 0.38
            + _rank(frame["pct_ytd"]) * 0.20
            + _rank(frame["pct_change"]) * 0.22
            + _rank(frame["volume_ratio"]) * 0.20
        ) * chase_penalty

        industry_map = _sector_score_map(sectors, "行业")
        concept_map = _sector_score_map(sectors, "概念")
        industry_score = frame["industry"].map(industry_map).fillna(0.35)

        def concept_score(value: str) -> float:
            names = [item for item in str(value).split(",") if item]
            return max([concept_map.get(item, 0.35) for item in names] or [0.35])

        sector = industry_score * 0.7 + frame["concept"].apply(concept_score) * 0.3
        liquidity = (
            _rank(np.log1p(frame["amount"])) * 0.5
            + _optimal_range(frame["turnover_rate"], target=4.0, width=10.0) * 0.3
            + _optimal_range(frame["volume_ratio"], target=1.5, width=2.5) * 0.2
        )

        pe_valid = frame["pe"].where(frame["pe"].between(3, 100))
        pb_valid = frame["pb"].where(frame["pb"].between(0.2, 15))
        value = (1 - _rank(pe_valid)) * 0.6 + (1 - _rank(pb_valid)) * 0.4
        value = value.where(pe_valid.notna() | pb_valid.notna(), 0.35)

        risk = (
            (1 - _rank(frame["amplitude"])) * 0.5
            + (1 - _rank(frame["pct_change"].abs())) * 0.2
            + _rank(np.log1p(frame["market_cap"])) * 0.3
        )

        frame["momentum_score"] = momentum.clip(0, 1)
        frame["sector_score"] = sector.clip(0, 1)
        frame["liquidity_score"] = liquidity.clip(0, 1)
        frame["value_score"] = value.clip(0, 1)
        frame["risk_score"] = risk.clip(0, 1)
        frame["total_score"] = sum(
            frame[f"{name}_score"] * weight for name, weight in self.weights.items()
        ) * 100
        frame["board"] = frame["code"].astype(str).map(_board)
        frame["is_technology"] = _technology_mask(frame)
        ranked = frame.sort_values(["total_score", "amount"], ascending=[False, False])
        reserve = min(self.technology_reserve, top_n, int(ranked["is_technology"].sum()))
        core = ranked.head(max(0, top_n - reserve))
        technology = ranked[ranked["is_technology"] & ~ranked.index.isin(core.index)].head(reserve)
        selected_indices = list(core.index) + list(technology.index)
        if len(selected_indices) < top_n:
            fill = ranked[~ranked.index.isin(selected_indices)].head(top_n - len(selected_indices))
            selected_indices.extend(fill.index.tolist())
        frame = ranked.loc[selected_indices].sort_values(
            ["total_score", "amount"], ascending=[False, False]
        )
        factor_labels = {
            "momentum": "趋势动量",
            "sector": "板块强度",
            "liquidity": "流动性",
            "value": "估值",
            "risk": "风险质量",
        }
        candidates: List[Candidate] = []
        for rank, (_, row) in enumerate(frame.iterrows(), start=1):
            factor_scores = {name: round(float(row[f"{name}_score"] * 100), 1) for name in self.weights}
            strongest = sorted(factor_scores, key=factor_scores.get, reverse=True)[:2]
            reasons = [f"{factor_labels[name]}得分靠前" for name in strongest]
            if row["volume_ratio"] >= 1.3:
                reasons.append(f"量比 {row['volume_ratio']:.2f}，交易活跃度提升")
            if row["industry"] != "未分类":
                reasons.append(f"所属行业：{row['industry']}")
            if row["is_technology"]:
                reasons.append(f"{row['board']} / 科技专项候选")

            risks = []
            if row["amplitude"] > 8:
                risks.append("日内振幅偏高")
            if row["pct_60d"] > 35:
                risks.append("60日涨幅较大，注意追高风险")
            if row["pe"] <= 0 or row["pe"] > 80:
                risks.append("盈利或估值指标异常")
            if factor_scores["sector"] < 45:
                risks.append("板块协同较弱")
            if not risks:
                risks.append("需关注次日量价确认和大盘系统性风险")

            completeness_fields = ["pct_60d", "pct_ytd", "turnover_rate", "amount", "pe", "pb"]
            completeness = sum(pd.notna(row[name]) and float(row[name]) != 0 for name in completeness_fields) / len(completeness_fields)
            classified = 1.0 if row["industry"] != "未分类" else 0.6
            confidence = round((completeness * 0.75 + classified * 0.25) * 100, 1)
            candidates.append(Candidate(
                rank=rank,
                code=str(row["code"]).zfill(6),
                name=str(row["name"]),
                price=round(float(row["price"]), 2),
                pct_change=round(float(row["pct_change"]), 2),
                industry=str(row["industry"]),
                concept=str(row["concept"]),
                total_score=round(float(row["total_score"]), 1),
                factor_scores=factor_scores,
                confidence=confidence,
                reasons=reasons,
                risks=risks,
                metrics={
                    "60日涨跌幅": round(float(row["pct_60d"]), 2),
                    "年初至今": round(float(row["pct_ytd"]), 2),
                    "换手率": round(float(row["turnover_rate"]), 2),
                    "量比": round(float(row["volume_ratio"]), 2),
                    "成交额": round(float(row["amount"]), 2),
                    "市盈率": round(float(row["pe"]), 2),
                    "市净率": round(float(row["pb"]), 2),
                    "振幅": round(float(row["amplitude"]), 2),
                },
                board=str(row["board"]),
                is_technology=bool(row["is_technology"]),
            ))
        return candidates
