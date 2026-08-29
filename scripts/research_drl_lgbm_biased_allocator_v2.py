from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.research_drl_adaptive_allocator_v1 as allocator


PROFILE_LABEL = "DRL LGBM-biased allocator v2"
INITIAL_TREND_WEIGHT = 0.40
INITIAL_LGBM_WEIGHT = 0.60
TREND_WEIGHT_MIN = 0.25
TREND_WEIGHT_MAX = 0.90


def configure() -> None:
    allocator.PROFILE_LABEL = PROFILE_LABEL
    allocator.TREND_WEIGHT_MIN = TREND_WEIGHT_MIN
    allocator.TREND_WEIGHT_MAX = TREND_WEIGHT_MAX
    allocator.INITIAL_TREND_WEIGHT = INITIAL_TREND_WEIGHT
    allocator.INITIAL_HEDGE_RATIO = 0.75
    allocator.OUTPUT_JSON = ROOT / "data" / "drl_lgbm_biased_allocator_v2.json"
    allocator.OUTPUT_REPORT = ROOT / "reports" / "drl_lgbm_biased_allocator_v2.md"
    allocator.OUTPUT_CURVES = ROOT / "reports" / "drl_lgbm_biased_allocator_v2_curves.csv"
    allocator.OUTPUT_WEIGHTS = ROOT / "reports" / "drl_lgbm_biased_allocator_v2_weights.csv"
    allocator.OUTPUT_CHART_HTML = ROOT / "reports" / "drl_lgbm_biased_allocator_v2_chart.html"
    allocator.OUTPUT_CHART_PNG = ROOT / "reports" / "drl_lgbm_biased_allocator_v2_chart.png"
    allocator.MODEL_DIR = ROOT / "data" / "models" / "drl_lgbm_biased_allocator_v2"


def main() -> None:
    configure()
    allocator.main()


if __name__ == "__main__":
    main()
