from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "reports" / "drl_adaptive_allocator_v1_curves.csv"
V2_PATH = ROOT / "reports" / "drl_lgbm_biased_allocator_v2_curves.csv"
V3_PATH = ROOT / "reports" / "drl_regime_allocator_v3_curves.csv"
OUTPUT_PATH = ROOT / "reports" / "drl_allocator_versions_comparison_curves.csv"


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def main() -> None:
    v1 = load(V1_PATH)
    v2 = load(V2_PATH)
    v3 = load(V3_PATH)
    index = v1.index.intersection(v2.index).intersection(v3.index)
    output = pd.DataFrame(
        {
            "drl_adaptive": v3.loc[index, "drl_adaptive"],
            "fixed_75_25_same_contract": v1.loc[index, "drl_adaptive"],
            "previous_75_25_independent_sleeves": v2.loc[index, "drl_adaptive"],
            "formal_lgbm": v3.loc[index, "fixed_75_25_same_contract"],
            "trend_weight": v3.loc[index, "trend_weight"],
            "hedge_ratio": v3.loc[index, "hedge_ratio"],
        },
        index=index,
    )
    output.index.name = "date"
    output.to_csv(OUTPUT_PATH, encoding="utf-8-sig", float_format="%.10f")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
