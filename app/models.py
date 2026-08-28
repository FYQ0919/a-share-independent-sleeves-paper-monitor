from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class MarketIndex:
    name: str
    code: str
    price: float
    pct_change: float


@dataclass
class Sector:
    name: str
    kind: str
    pct_change: float
    turnover_rate: float = 0.0
    main_inflow: float = 0.0
    up_count: int = 0
    down_count: int = 0
    leading_stock: str = ""


@dataclass
class Candidate:
    rank: int
    code: str
    name: str
    price: float
    pct_change: float
    industry: str
    concept: str
    total_score: float
    factor_scores: Dict[str, float]
    confidence: float
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    board: str = ""
    is_technology: bool = False


@dataclass
class StrategyPick:
    rank: int
    code: str
    name: str
    score: float
    price: float
    target_weight: float
    factor_scores: Dict[str, float]
    reason: str


@dataclass
class RunResult:
    run_id: str
    created_at: datetime
    mode: str
    market_status: str
    indices: List[MarketIndex]
    sectors: List[Sector]
    candidates: List[Candidate]
    market_breadth: Dict[str, Any]
    report_markdown: str
    report_summary: str
    strategy_picks: List[StrategyPick] = field(default_factory=list)
    strategy_signal: Dict[str, Any] = field(default_factory=dict)
    macro_environment: Dict[str, Any] = field(default_factory=dict)
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload
