"""Print panel metadata for the analysis sample (full-time employees, 2009-2014)."""

import pandas as pd
from src.data.utils import load_master

df = load_master()

# Analysis sample: valid WFH response only (mirrors binary_choice.py)
analysis = df[df["plb0097"].isin([0, 1])].copy()

waves     = sorted(df["syear"].unique())
T         = len(waves)
N_raw     = df["pid"].nunique()
NT_raw    = len(df)
N_ana     = analysis["pid"].nunique()
NT_ana    = len(analysis)

obs_per_person = df.groupby("pid").size()
obs_per_person_ana = analysis.groupby("pid").size()

print("=" * 55)
print("PANEL METADATA  (full-time employees, pgemplst == 1)")
print("=" * 55)

print(f"\nStudy period : {min(waves)}–{max(waves)}  ({T} waves)")

print(f"\n--- Full master sample ---")
print(f"  Total obs (NT)       : {NT_raw:>8,}")
print(f"  Unique individuals N : {N_raw:>8,}")
print(f"  Avg obs per person   : {NT_raw/N_raw:>8.2f}")
print(f"  Min obs per person   : {obs_per_person.min():>8}")
print(f"  Max obs per person   : {obs_per_person.max():>8}")
print(f"  Obs per wave:")
for yr, cnt in df["syear"].value_counts().sort_index().items():
    share = cnt / NT_raw * 100
    print(f"    {yr}: {cnt:>6,}  ({share:.1f}%)")

print(f"\n--- Analysis sample (plb0097 ∈ {{0,1}}) ---")
print(f"  Total obs (NT)       : {NT_ana:>8,}")
print(f"  Unique individuals N : {N_ana:>8,}")
print(f"  Avg obs per person   : {NT_ana/N_ana:>8.2f}")
print(f"  Min obs per person   : {obs_per_person_ana.min():>8}")
print(f"  Max obs per person   : {obs_per_person_ana.max():>8}")

wave_dist = obs_per_person_ana.value_counts().sort_index()
print(f"  Wave participation:")
for n_waves, n_persons in wave_dist.items():
    print(f"    {n_waves} wave(s): {n_persons:>5,} persons  ({n_persons/N_ana*100:.1f}%)")

print(f"\n  Balanced if all had {T} obs: {N_ana * T:,}  (actual: {NT_ana:,}, ratio: {NT_ana/(N_ana*T):.3f})")
