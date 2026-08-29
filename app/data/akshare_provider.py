import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from app.data.base import MarketDataProvider
from app.models import MarketIndex, Sector


def _number(frame: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _text(frame: pd.DataFrame, names: Iterable[str], default: str = "") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


class AkShareProvider(MarketDataProvider):
    name = "live"

    def __init__(self, cache_dir: Path, max_sectors: int = 10):
        self.cache_dir = cache_dir
        self.max_sectors = max_sectors
        self.cache_path = cache_dir / "akshare_snapshot.json"

    def load(self) -> Tuple[pd.DataFrame, List[Sector], List[MarketIndex], Dict]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._fetch_live()
        except Exception as exc:
            cached = self._read_cache(max_age=timedelta(hours=24))
            if cached:
                frame, sectors, indices, breadth = cached
                breadth["source"] = "实时源失败，使用最近缓存"
                breadth["provider_error"] = str(exc)
                return frame, sectors, indices, breadth
            raise RuntimeError(f"实时行情获取失败且没有可用缓存: {exc}") from exc

    def _fetch_live(self):
        import akshare as ak

        source = "AkShare / 东方财富"
        source_warnings = []
        try:
            raw = ak.stock_zh_a_spot_em()
            frame = self._normalize_eastmoney_snapshot(raw)
        except Exception as exc:
            raw = ak.stock_zh_a_spot_tx()
            frame = self._normalize_tencent_snapshot(raw)
            source = "AkShare / 腾讯行情（降级）"
            source_warnings.append("东方财富快照不可用，已切换腾讯行情")

        industry_members_available = True
        try:
            industries = self._fetch_sectors(ak.stock_board_industry_name_em(), "行业")
            concepts = self._fetch_sectors(ak.stock_board_concept_name_em(), "概念")
        except Exception as exc:
            industries = self._fetch_sina_industries(ak)
            concepts = []
            industry_members_available = False
            source_warnings.append("东方财富板块接口不可用，已切换新浪行业排行")

        if industry_members_available:
            self._assign_top_sector_members(ak, frame, industries, concepts)
        else:
            source_warnings.append("降级板块源不含稳定的全量成分映射，个股板块分项按中性值计算")
        sectors = sorted(industries + concepts, key=lambda item: item.pct_change, reverse=True)
        indices = self._fetch_indices(ak)
        if not indices:
            source_warnings.append("指数快照暂缺")
        breadth = {
            "up": int((frame["pct_change"] > 0).sum()),
            "down": int((frame["pct_change"] < 0).sum()),
            "flat": int((frame["pct_change"] == 0).sum()),
            "amount": float(frame["amount"].sum()),
            "source": source,
            "source_warnings": source_warnings,
        }
        self._write_cache(frame, sectors, indices, breadth)
        return frame, sectors, indices, breadth

    @staticmethod
    def _normalize_eastmoney_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame({
            "code": _text(raw, ["代码"]),
            "name": _text(raw, ["名称"]),
            "price": _number(raw, ["最新价"]),
            "pct_change": _number(raw, ["涨跌幅"]),
            "change_5m": _number(raw, ["5分钟涨跌"]),
            "pct_60d": _number(raw, ["60日涨跌幅"]),
            "pct_ytd": _number(raw, ["年初至今涨跌幅"]),
            "turnover_rate": _number(raw, ["换手率"]),
            "volume_ratio": _number(raw, ["量比"], 1.0),
            "amount": _number(raw, ["成交额"]),
            "amplitude": _number(raw, ["振幅"]),
            "pe": _number(raw, ["市盈率-动态", "市盈率"]),
            "pb": _number(raw, ["市净率"]),
            "market_cap": _number(raw, ["总市值"]),
            "industry": "未分类",
            "concept": "",
        })
        return frame

    @staticmethod
    def _normalize_tencent_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "code": _text(raw, ["code"]).str.replace(r"^(sh|sz|bj)", "", regex=True),
            "name": _text(raw, ["name"]),
            "price": _number(raw, ["zxj"]),
            "pct_change": _number(raw, ["zdf"]),
            "change_5m": _number(raw, ["speed"]),
            "pct_60d": _number(raw, ["zdf_d60"]),
            "pct_ytd": _number(raw, ["zdf_y"]),
            "turnover_rate": _number(raw, ["hsl"]),
            "volume_ratio": _number(raw, ["lb"], 1.0),
            # Tencent returns turnover in ten-thousand yuan and market cap in hundred-million yuan.
            "amount": _number(raw, ["turnover"]) * 10_000,
            "amplitude": _number(raw, ["zf"]),
            "pe": _number(raw, ["pe_ttm"]),
            "pb": 0.0,
            "market_cap": _number(raw, ["zsz"]) * 100_000_000,
            "industry": "未分类",
            "concept": "",
        })

    def _fetch_sectors(self, raw: pd.DataFrame, kind: str) -> List[Sector]:
        result = []
        for idx in raw.index:
            row = raw.loc[[idx]]
            result.append(Sector(
                name=str(_text(row, ["板块名称", "名称"]).iloc[0]),
                kind=kind,
                pct_change=float(_number(row, ["涨跌幅"]).iloc[0]),
                turnover_rate=float(_number(row, ["换手率"]).iloc[0]),
                main_inflow=float(_number(row, ["主力净流入-净额", "主力净流入"]).iloc[0]),
                up_count=int(_number(row, ["上涨家数"]).iloc[0]),
                down_count=int(_number(row, ["下跌家数"]).iloc[0]),
                leading_stock=str(_text(row, ["领涨股票"]).iloc[0]),
            ))
        return result

    def _fetch_sina_industries(self, ak) -> List[Sector]:
        raw = ak.stock_sector_spot(indicator="启明星行业")
        result = []
        for idx in raw.index:
            row = raw.loc[[idx]]
            result.append(Sector(
                name=str(_text(row, ["板块"]).iloc[0]),
                kind="行业",
                pct_change=float(_number(row, ["涨跌幅"]).iloc[0]),
                turnover_rate=0.0,
                main_inflow=0.0,
                up_count=0,
                down_count=0,
                leading_stock=str(_text(row, ["股票名称"]).iloc[0]),
            ))
        return result

    def _assign_top_sector_members(self, ak, frame, industries, concepts):
        for sector in sorted(industries, key=lambda x: x.pct_change, reverse=True)[: self.max_sectors]:
            try:
                members = ak.stock_board_industry_cons_em(symbol=sector.name)
                codes = set(_text(members, ["代码"]).tolist())
                frame.loc[frame["code"].isin(codes), "industry"] = sector.name
            except Exception:
                continue
        for sector in sorted(concepts, key=lambda x: x.pct_change, reverse=True)[: max(5, self.max_sectors // 2)]:
            try:
                members = ak.stock_board_concept_cons_em(symbol=sector.name)
                codes = set(_text(members, ["代码"]).tolist())
                mask = frame["code"].isin(codes)
                frame.loc[mask, "concept"] = frame.loc[mask, "concept"].apply(
                    lambda value: f"{value},{sector.name}".strip(",")
                )
            except Exception:
                continue

    def _fetch_indices(self, ak) -> List[MarketIndex]:
        try:
            raw = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
            wanted = {"上证指数", "深证成指", "创业板指", "沪深300"}
            output = []
            for idx in raw.index:
                row = raw.loc[[idx]]
                name = str(_text(row, ["名称"]).iloc[0])
                if name in wanted:
                    output.append(MarketIndex(
                        name=name,
                        code=str(_text(row, ["代码"]).iloc[0]),
                        price=float(_number(row, ["最新价"]).iloc[0]),
                        pct_change=float(_number(row, ["涨跌幅"]).iloc[0]),
                    ))
            return output
        except Exception:
            return []

    def _write_cache(self, frame, sectors, indices, breadth):
        payload = {
            "created_at": datetime.now().isoformat(),
            "stocks": frame.to_dict(orient="records"),
            "sectors": [item.__dict__ for item in sectors],
            "indices": [item.__dict__ for item in indices],
            "breadth": breadth,
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _read_cache(self, max_age: timedelta):
        if not self.cache_path.exists():
            return None
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        created_at = datetime.fromisoformat(payload["created_at"])
        if datetime.now() - created_at > max_age:
            return None
        return (
            pd.DataFrame(payload["stocks"]),
            [Sector(**item) for item in payload["sectors"]],
            [MarketIndex(**item) for item in payload["indices"]],
            payload["breadth"],
        )
