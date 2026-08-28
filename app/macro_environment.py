import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


POSITIVE_POLICY_WORDS = (
    "支持", "促进", "提振", "稳增长", "扩内需", "降准", "降息", "纾困",
    "科技创新", "民营经济", "资本市场", "改革", "扩大开放", "设备更新",
)
NEGATIVE_POLICY_WORDS = (
    "收紧", "处罚", "问责", "禁止", "债务风险", "房地产风险", "系统性风险",
    "专项整治", "严格监管", "退市风险",
)
POSITIVE_SENTIMENT_WORDS = (
    "上涨", "反弹", "走强", "增长", "改善", "突破", "新高", "回暖", "利好",
    "增持", "超预期", "复苏", "提振", "扩张",
)
NEGATIVE_SENTIMENT_WORDS = (
    "下跌", "走弱", "暴跌", "风险", "亏损", "违约", "减持", "低迷", "衰退",
    "冲突", "制裁", "关税", "收紧", "不及预期",
)
GLOBAL_INDEX_KEYWORDS = (
    "纳斯达克", "标普500", "标普 500", "道琼斯", "费城半导体", "日经225",
    "日经 225", "恒生指数", "韩国KOSPI", "德国DAX",
)


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _value(row: Dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _records(frame: Any) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, pd.DataFrame):
        return frame.to_dict(orient="records")
    return [dict(item) for item in frame]


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().replace(tzinfo=None)


def _label(score: float) -> str:
    if score >= 62:
        return "偏利多"
    if score >= 48:
        return "中性"
    if score >= 38:
        return "偏谨慎"
    return "高风险"


def _keyword_score(
    rows: Sequence[Dict[str, Any]],
    positive_words: Sequence[str],
    negative_words: Sequence[str],
    title_names: Sequence[str],
    summary_names: Sequence[str],
) -> Dict[str, Any]:
    polarities: List[float] = []
    examples: List[str] = []
    positive_hits = 0
    negative_hits = 0
    for row in rows:
        title = str(_value(row, title_names, "")).strip()
        summary = str(_value(row, summary_names, "")).strip()
        text = f"{title} {summary}"
        positive = sum(word in text for word in positive_words)
        negative = sum(word in text for word in negative_words)
        if positive or negative:
            polarities.append(_clamp(positive - negative, -2, 2))
            positive_hits += positive
            negative_hits += negative
            if title and len(examples) < 3:
                examples.append(title[:80])
    score = 50.0 if not polarities else _clamp(50 + sum(polarities) / len(polarities) * 10)
    return {
        "score": round(score, 1),
        "matched_articles": len(polarities),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "examples": examples,
    }


class MacroEnvironmentService:
    VERSION = "macro-overlay-v2-base-protected"
    MAX_MACRO_REDUCTION = 0.02
    EXTREME_RISK_SCORE = 38.0
    MIN_EXECUTION_CONFIDENCE = 0.65

    def __init__(self, cache_dir: Path, cache_minutes: int = 120):
        self.cache_path = Path(cache_dir) / "macro_environment.json"
        self.cache_minutes = cache_minutes

    def assess(self, now: datetime, breadth: Dict[str, Any], mode: str) -> Dict[str, Any]:
        if mode != "live":
            return self.neutral(now, "演示模式不请求实时宏观数据")
        cached = self._read_cache(now, timedelta(minutes=self.cache_minutes))
        if cached:
            cached["cache_status"] = "fresh"
            return cached

        warnings: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return self.neutral(now, f"AkShare 不可用: {exc}")

        global_indices = self._fetch(warnings, "全球指数", ak.index_global_spot_em)
        pmi = self._fetch(warnings, "中国 PMI", ak.macro_china_pmi)
        lpr = self._fetch(warnings, "中国 LPR", ak.macro_china_lpr)
        policy_news = self._fetch_policy_news(ak, now, warnings)
        market_news = self._fetch(warnings, "财经舆情", ak.stock_info_global_em)
        result = self.assess_inputs(
            now=now,
            breadth=breadth,
            global_indices=global_indices,
            pmi=pmi,
            lpr=lpr,
            policy_news=policy_news,
            market_news=market_news,
            warnings=warnings,
        )
        self._write_cache(result)
        return result

    @classmethod
    def assess_inputs(
        cls,
        now: datetime,
        breadth: Dict[str, Any],
        global_indices: Any = None,
        pmi: Any = None,
        lpr: Any = None,
        policy_news: Any = None,
        market_news: Any = None,
        warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        warning_list = list(warnings or [])
        international = cls._international(now, _records(global_indices))
        policy = cls._domestic_policy(_records(pmi), _records(lpr), _records(policy_news))
        sentiment = cls._sentiment(now, breadth, _records(market_news))
        dimensions = {
            "international": international,
            "domestic_policy": policy,
            "sentiment": sentiment,
        }
        weights = {"international": 0.40, "domestic_policy": 0.35, "sentiment": 0.25}
        score = sum(dimensions[name]["score"] * weight for name, weight in weights.items())
        confidence = sum(dimensions[name]["coverage"] * weight for name, weight in weights.items())
        if confidence < 0.35:
            warning_list.append("宏观数据覆盖不足，综合分按中性收缩，仅供观察")
            score = 50 + (score - 50) * confidence / 0.35
        score = round(_clamp(score), 1)
        exposure = cls._protected_exposure(score, confidence)
        warning_list.append("宏观评测仅作风险提示，基础策略订单不自动修改")
        return {
            "version": cls.VERSION,
            "as_of": now.isoformat(),
            "fetched_at": now.isoformat(),
            "score": score,
            "regime": _label(score),
            "recommended_exposure": exposure,
            "automatic_execution": False,
            "base_strategy_priority": True,
            "max_macro_reduction": cls.MAX_MACRO_REDUCTION,
            "dimensions": dimensions,
            "weights": weights,
            "confidence": round(_clamp(confidence, 0, 1), 3),
            "warnings": list(dict.fromkeys(warning_list)),
            "cache_status": "new",
            "point_in_time": {
                "availability_time": now.isoformat(),
                "future_records_ignored": True,
                "historical_replay_ready": False,
                "note": "当前快照只用于实时评测；进入回测前需保存逐日可用快照并按发布日期回放。",
            },
        }

    @classmethod
    def neutral(cls, now: datetime, warning: str) -> Dict[str, Any]:
        dimensions = {
            name: {"score": 50.0, "label": "中性", "coverage": 0.0, "signals": [], "sources": []}
            for name in ("international", "domestic_policy", "sentiment")
        }
        return {
            "version": cls.VERSION,
            "as_of": now.isoformat(),
            "fetched_at": now.isoformat(),
            "score": 50.0,
            "regime": "中性",
            "recommended_exposure": 1.0,
            "automatic_execution": False,
            "base_strategy_priority": True,
            "max_macro_reduction": cls.MAX_MACRO_REDUCTION,
            "dimensions": dimensions,
            "weights": {"international": 0.40, "domestic_policy": 0.35, "sentiment": 0.25},
            "confidence": 0.0,
            "warnings": [warning],
            "cache_status": "neutral",
            "point_in_time": {
                "availability_time": now.isoformat(),
                "future_records_ignored": True,
                "historical_replay_ready": False,
                "note": "无实时宏观输入，不改变交易指令。",
            },
        }

    @classmethod
    def _protected_exposure(cls, score: float, confidence: float) -> float:
        if score < cls.EXTREME_RISK_SCORE and confidence >= cls.MIN_EXECUTION_CONFIDENCE:
            return 1.0 - cls.MAX_MACRO_REDUCTION
        return 1.0

    @staticmethod
    def _international(now: datetime, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected = []
        for row in rows:
            name = str(_value(row, ("名称", "name", "指数名称"), ""))
            if not any(keyword in name for keyword in GLOBAL_INDEX_KEYWORDS):
                continue
            updated = _parse_time(_value(row, ("行情时间", "更新时间", "时间", "updated_at"), None))
            if updated and (updated > now + timedelta(minutes=5) or updated < now - timedelta(days=4)):
                continue
            change = _number(_value(row, ("涨跌幅", "pct_change", "change_pct"), None))
            if change is None:
                continue
            selected.append({"name": name, "pct_change": round(change, 2), "updated_at": updated.isoformat() if updated else ""})
        mean_change = sum(item["pct_change"] for item in selected) / len(selected) if selected else 0.0
        score = _clamp(50 + mean_change * 8) if selected else 50.0
        signals = [f"{item['name']} {item['pct_change']:+.2f}%" for item in selected[:8]]
        return {
            "score": round(score, 1),
            "label": _label(score),
            "coverage": round(min(1.0, len(selected) / 5), 3),
            "signals": signals,
            "sources": ["东方财富全球指数快照"] if selected else [],
            "data_points": len(selected),
        }

    @staticmethod
    def _domestic_policy(pmi_rows, lpr_rows, policy_rows) -> Dict[str, Any]:
        component_scores: List[float] = []
        signals: List[str] = []
        if pmi_rows:
            latest = pmi_rows[0]
            manufacturing = _number(_value(latest, ("制造业-指数", "manufacturing_pmi"), None))
            services = _number(_value(latest, ("非制造业-指数", "non_manufacturing_pmi"), None))
            values = [value for value in (manufacturing, services) if value is not None]
            if values:
                pmi_score = _clamp(50 + sum(value - 50 for value in values) / len(values) * 7)
                component_scores.append(pmi_score)
                signals.append("PMI " + " / ".join(f"{value:.1f}" for value in values))
        if len(lpr_rows) >= 2:
            ordered = sorted(
                lpr_rows,
                key=lambda row: _parse_time(_value(row, ("TRADE_DATE", "date"), None)) or datetime.min,
            )
            latest, previous = ordered[-1], ordered[-2]
            current = _number(_value(latest, ("LPR1Y", "lpr_1y"), None))
            prior = _number(_value(previous, ("LPR1Y", "lpr_1y"), None))
            if current is not None and prior is not None:
                component_scores.append(_clamp(50 - (current - prior) * 100))
                signals.append(f"1Y LPR {current:.2f}%（环比 {current - prior:+.2f}pct）")
        news_score = _keyword_score(
            policy_rows,
            POSITIVE_POLICY_WORDS,
            NEGATIVE_POLICY_WORDS,
            ("title", "标题"),
            ("content", "摘要", "summary"),
        )
        if policy_rows:
            component_scores.append(news_score["score"])
            signals.append(
                f"政策文本正向/负向词 {news_score['positive_hits']}/{news_score['negative_hits']}"
            )
            signals.extend(news_score["examples"][:2])
        score = sum(component_scores) / len(component_scores) if component_scores else 50.0
        coverage = min(1.0, len(component_scores) / 3)
        sources = []
        if pmi_rows:
            sources.append("国家统计口径 PMI（AkShare）")
        if lpr_rows:
            sources.append("LPR（AkShare）")
        if policy_rows:
            sources.append("央视新闻政策文本")
        return {
            "score": round(score, 1),
            "label": _label(score),
            "coverage": round(coverage, 3),
            "signals": signals[:8],
            "sources": sources,
            "data_points": len(component_scores),
        }

    @staticmethod
    def _sentiment(now: datetime, breadth: Dict[str, Any], news_rows) -> Dict[str, Any]:
        eligible = []
        for row in news_rows:
            published = _parse_time(_value(row, ("发布时间", "publish_time", "time"), None))
            if published and (published > now + timedelta(minutes=5) or published < now - timedelta(hours=36)):
                continue
            eligible.append(row)
        news_score = _keyword_score(
            eligible,
            POSITIVE_SENTIMENT_WORDS,
            NEGATIVE_SENTIMENT_WORDS,
            ("标题", "title"),
            ("摘要", "summary", "content"),
        )
        up = _number(breadth.get("up")) or 0.0
        down = _number(breadth.get("down")) or 0.0
        flat = _number(breadth.get("flat")) or 0.0
        total = up + down + flat
        breadth_score = up / total * 100 if total else None
        components = [news_score["score"]] if eligible else []
        if breadth_score is not None:
            components.append(breadth_score)
        score = sum(components) / len(components) if components else 50.0
        signals = []
        if eligible:
            signals.append(
                f"财经舆情正向/负向词 {news_score['positive_hits']}/{news_score['negative_hits']}"
            )
            signals.extend(news_score["examples"][:3])
        if breadth_score is not None:
            signals.append(f"A股上涨家数占比 {breadth_score:.1f}%")
        coverage = min(1.0, (0.6 if len(eligible) >= 10 else len(eligible) * 0.06) + (0.4 if total else 0.0))
        sources = (["东方财富财经新闻"] if eligible else []) + (["A股实时市场宽度"] if total else [])
        return {
            "score": round(score, 1),
            "label": _label(score),
            "coverage": round(coverage, 3),
            "signals": signals[:8],
            "sources": sources,
            "data_points": len(eligible) + (1 if total else 0),
            "future_records_ignored": len(eligible) < len(news_rows),
        }

    def _fetch_policy_news(self, ak, now: datetime, warnings: List[str]):
        for target in (now, now - timedelta(days=1)):
            result = self._fetch(warnings, f"央视新闻 {target:%Y-%m-%d}", ak.news_cctv, date=target.strftime("%Y%m%d"))
            if len(result):
                return result
        return []

    @staticmethod
    def _fetch(warnings: List[str], label: str, function, **kwargs):
        try:
            return function(**kwargs)
        except Exception as exc:
            warnings.append(f"{label}获取失败: {exc}")
            return []

    def _read_cache(self, now: datetime, max_age: timedelta):
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != self.VERSION:
                return None
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
            if fetched_at.date() != now.date() or now - fetched_at > max_age:
                return None
            return payload
        except Exception:
            return None

    def _write_cache(self, payload: Dict[str, Any]):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
