from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.research_drl_adaptive_allocator_v1 as allocator


PROFILE_LABEL = "DRL regime-responsive allocator v3"
INITIAL_TREND_WEIGHT = 0.40
INITIAL_LGBM_WEIGHT = 0.60
TREND_WEIGHT_MIN = 0.25
TREND_WEIGHT_MAX = 0.90
PRIOR_TARGET_BLEND = 0.25
PRIOR_TREND_WEIGHT_START = 0.40
PRIOR_TREND_WEIGHT_END = 0.50
PRIOR_ANNEAL_EVENTS = 24
TREND_INCREASE_SMOOTHING_RISK_ON = 0.70
TREND_DECREASE_SMOOTHING = 0.25


def configure() -> None:
    allocator.PROFILE_LABEL = PROFILE_LABEL
    allocator.TREND_WEIGHT_MIN = TREND_WEIGHT_MIN
    allocator.TREND_WEIGHT_MAX = TREND_WEIGHT_MAX
    allocator.INITIAL_TREND_WEIGHT = INITIAL_TREND_WEIGHT
    allocator.INITIAL_HEDGE_RATIO = 0.75
    allocator.ACTION_SMOOTHING = 0.35
    allocator.ACTION_SMOOTHING_UP = TREND_INCREASE_SMOOTHING_RISK_ON
    allocator.ACTION_SMOOTHING_DOWN = TREND_DECREASE_SMOOTHING
    allocator.HEDGE_ACTION_SMOOTHING = 0.35
    allocator.PRIOR_TARGET_BLEND = PRIOR_TARGET_BLEND
    allocator.PRIOR_TREND_WEIGHT_START = PRIOR_TREND_WEIGHT_START
    allocator.PRIOR_TREND_WEIGHT_END = PRIOR_TREND_WEIGHT_END
    allocator.PRIOR_ANNEAL_EVENTS = PRIOR_ANNEAL_EVENTS
    allocator.OUTPUT_JSON = ROOT / "data" / "drl_regime_allocator_v3.json"
    allocator.OUTPUT_REPORT = ROOT / "reports" / "drl_regime_allocator_v3.md"
    allocator.OUTPUT_CURVES = ROOT / "reports" / "drl_regime_allocator_v3_curves.csv"
    allocator.OUTPUT_WEIGHTS = ROOT / "reports" / "drl_regime_allocator_v3_weights.csv"
    allocator.OUTPUT_CHART_HTML = ROOT / "reports" / "drl_regime_allocator_v3_chart.html"
    allocator.OUTPUT_CHART_PNG = ROOT / "reports" / "drl_regime_allocator_v3_chart.png"
    allocator.MODEL_DIR = ROOT / "data" / "models" / "drl_regime_allocator_v3"


def main() -> None:
    configure()
    allocator.main()


if __name__ == "__main__":
    main()
