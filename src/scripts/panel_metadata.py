"""Print panel metadata for the two analysis samples:
1. Preference Stage (RQ1): determinant of willingness to WFH.
2. Realization Stage (RQ2): determinants of actual WFH among those willing.
"""

import pandas as pd
from src.data.utils import load_master
from src.data.descriptives import build_analysis_sample, build_realization_sample

def print_stats(name: str, df: pd.DataFrame, T: int):
    N = df["pid"].nunique()
    NT = len(df)
    obs_per_person = df.groupby("pid").size()
    
    print(f"\n--- {name} ---")
    print(f"  Total obs (NT)       : {NT:>8,}")
    print(f"  Unique individuals N : {N:>8,}")
    print(f"  Avg obs per person   : {NT/N:>8.2f}")
    print(f"  Min obs per person   : {obs_per_person.min():>8}")
    print(f"  Max obs per person   : {obs_per_person.max():>8}")
    
    wave_dist = obs_per_person.value_counts().sort_index()
    print(f"  Wave participation:")
    for n_waves, n_persons in wave_dist.items():
        print(f"    {n_waves} wave(s): {n_persons:>5,} persons  ({n_persons/N*100:.1f}%)")

    print(f"  Balanced ratio       : {NT/(N*T):.3f}")

def main():
    raw = load_master()
    rq1 = build_analysis_sample(raw)
    rq2 = build_realization_sample(raw)

    waves = sorted(raw["syear"].unique())
    T = len(waves)

    print("=" * 60)
    print("PANEL METADATA: ANALYSIS SUBSAMPLES (2009-2014)")
    print("=" * 60)
    print(f"Study period: {min(waves)}–{max(waves)} ({T} waves)")

    print_stats("Preference Stage (RQ1)", rq1, T)
    print_stats("Realization Stage (RQ2)", rq2, T)

if __name__ == "__main__":
    main()
