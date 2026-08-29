from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.data.base import MarketDataProvider
from app.models import MarketIndex, Sector


class DemoProvider(MarketDataProvider):
    name = "demo"

    INDUSTRIES = [
        "半导体", "电力设备", "工业软件", "创新药",
        "有色金属", "银行", "消费电子", "汽车零部件",
    ]
    CONCEPTS = ["人工智能", "机器人", "低空经济", "高股息", "算力", "国产替代"]

    def load(self) -> Tuple[pd.DataFrame, List[Sector], List[MarketIndex], Dict]:
        seed = int(datetime.now().strftime("%Y%m%d"))
        rng = np.random.default_rng(seed)
        count = 160
        industries = rng.choice(self.INDUSTRIES, count)
        concepts = rng.choice(self.CONCEPTS, count)
        sector_bias = {name: value for name, value in zip(self.INDUSTRIES, np.linspace(2.8, -1.5, 8))}
        daily = np.array([sector_bias[x] for x in industries]) + rng.normal(0, 2.1, count)
        frame = pd.DataFrame({
            "code": [
                f"{600000 + i:06d}" if i < 70
                else f"{i - 69:06d}" if i < 130
                else f"{688000 + i - 129:06d}"
                for i in range(count)
            ],
            "name": [f"样例股份{i + 1:03d}" for i in range(count)],
            "price": rng.uniform(4, 120, count),
            "pct_change": daily,
            "change_5m": rng.normal(0, 0.7, count),
            "pct_60d": daily * 3 + rng.normal(5, 12, count),
            "pct_ytd": daily * 4 + rng.normal(8, 18, count),
            "turnover_rate": rng.uniform(0.4, 12, count),
            "volume_ratio": rng.uniform(0.5, 2.8, count),
            "amount": rng.lognormal(19.4, 0.8, count),
            "amplitude": np.abs(daily) + rng.uniform(1, 5, count),
            "pe": rng.uniform(7, 75, count),
            "pb": rng.uniform(0.7, 8, count),
            "market_cap": rng.lognormal(23.5, 1.0, count),
            "industry": industries,
            "concept": concepts,
        })
        frame.loc[0, "name"] = "ST样例"
        frame.loc[1, "amount"] = 1_000_000

        sectors: List[Sector] = []
        for kind, names in (("行业", self.INDUSTRIES), ("概念", self.CONCEPTS)):
            for name in names:
                members = frame[frame["industry" if kind == "行业" else "concept"] == name]
                sectors.append(Sector(
                    name=name,
                    kind=kind,
                    pct_change=round(float(members["pct_change"].mean()), 2),
                    turnover_rate=round(float(members["turnover_rate"].mean()), 2),
                    main_inflow=float(members["amount"].sum() * members["pct_change"].mean() / 100),
                    up_count=int((members["pct_change"] > 0).sum()),
                    down_count=int((members["pct_change"] < 0).sum()),
                    leading_stock=str(members.sort_values("pct_change", ascending=False).iloc[0]["name"]),
                ))
        sectors.sort(key=lambda item: item.pct_change, reverse=True)
        indices = [
            MarketIndex("上证指数", "000001", 3388.42, 0.64),
            MarketIndex("深证成指", "399001", 10872.31, 1.08),
            MarketIndex("创业板指", "399006", 2219.65, 1.42),
            MarketIndex("沪深300", "000300", 3996.18, 0.76),
        ]
        breadth = {
            "up": int((frame["pct_change"] > 0).sum()),
            "down": int((frame["pct_change"] < 0).sum()),
            "flat": int((frame["pct_change"] == 0).sum()),
            "amount": float(frame["amount"].sum()),
            "source": "演示数据",
        }
        return frame, sectors, indices, breadth
