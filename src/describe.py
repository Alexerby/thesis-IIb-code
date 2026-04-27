"""
Descriptive statistics for a SOEP-Core variable.

Usage:
    python src/describe.py --variable p11101
    python src/describe.py --variable p11101 --dataset pequiv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd


SOEP_MISSING = [-8, -7, -6, -5, -4, -3, -2, -1]


def load_config():
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def find_dataset(soep_dir: str, variable: str) -> str | None:
    for path in sorted(Path(soep_dir).glob("*_variables.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 4 and row[3].lower() == variable.lower():
                    return row[1]
    return None


def get_value_labels(soep_dir: str, variable: str, dataset: str) -> dict[int, str]:
    labels = {}
    path = Path(soep_dir) / f"{dataset}_values.csv"
    if not path.exists():
        return labels
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 6 and row[3].lower() == variable.lower():
                try:
                    val = int(row[4])
                    label = row[5] or (row[6] if len(row) > 6 else "")
                    if label:
                        labels[val] = label
                except ValueError:
                    pass
    return labels


def main():
    parser = argparse.ArgumentParser(description="Describe a SOEP-Core variable")
    parser.add_argument("--variable", "-v", required=True, help="Variable name (e.g. p11101)")
    parser.add_argument("--dataset", "-d", help="Dataset to load from (auto-detected if omitted)")
    args = parser.parse_args()

    config = load_config()
    soep_dir = config["data"]["soep_dir"]
    variable = args.variable.lower()

    # Check harmonized variables in config first, then fall back to codebook search
    harmonized = {h["name"].lower(): h["dataset"] for h in config.get("harmonize", [])}
    dataset = args.dataset or harmonized.get(variable) or find_dataset(soep_dir, variable)
    if not dataset:
        print(f"Variable '{args.variable}' not found in any dataset.")
        sys.exit(1)

    parquet_path = Path("output/data") / f"{dataset}.parquet"
    if parquet_path.exists():
        print(f"Loading {parquet_path.name} ...")
        try:
            df = pd.read_parquet(parquet_path, columns=["pid", "syear", variable])
        except Exception:
            print(f"  Column '{variable}' not in parquet — variable may not be in config.json variables list.")
            print(f"  Add it to config.json and re-run extract.py, or use --dataset to force CSV fallback.")
            sys.exit(1)
    else:
        csv_path = Path(soep_dir) / f"{dataset}.csv"
        if not csv_path.exists():
            print(f"Data file not found: {csv_path}")
            sys.exit(1)
        print(f"Loading {csv_path.name} (no parquet found — run extract.py first for large files) ...")
        df = pd.read_csv(csv_path, usecols=lambda c: c.lower() in {variable, "pid", "syear"}, low_memory=False)
        df.columns = df.columns.str.lower()

    if variable not in df.columns:
        print(f"Column '{variable}' not found in {dataset}.")
        sys.exit(1)

    df[variable] = pd.to_numeric(df[variable], errors="coerce")
    total_rows = len(df)
    df_valid = df[~df[variable].isin(SOEP_MISSING) & df[variable].notna()]

    print(f"\n{'='*60}")
    print(f"  Variable : {args.variable}  |  Dataset: {dataset}")
    print(f"{'='*60}")

    # Value labels for endpoint anchors
    labels = get_value_labels(soep_dir, variable, dataset)
    if labels:
        positive = {k: v for k, v in labels.items() if k >= 0}
        if positive:
            lo, hi = min(positive), max(positive)
            print(f"  Scale    : {lo} = {positive[lo]}  →  {hi} = {positive[hi]}")

    print(f"\n  Total rows in file : {total_rows:,}")
    print(f"  Valid observations : {len(df_valid):,}")
    print(f"  Missing / coded    : {total_rows - len(df_valid):,}  ({(total_rows - len(df_valid)) / total_rows * 100:.1f}%)")

    # Descriptive statistics
    print(f"\n--- Descriptive statistics (valid obs) ---")
    desc = df_valid[variable].describe(percentiles=[0.25, 0.5, 0.75])
    for stat, val in desc.items():
        print(f"  {stat:<8} {val:.4f}")

    # Observations by year
    if "syear" in df_valid.columns:
        print(f"\n--- Observations by year ---")
        by_year = (
            df_valid.groupby("syear")[variable]
            .agg(n="count", mean="mean", std="std")
            .reset_index()
        )
        print(f"  {'Year':<8} {'N':>8} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*36}")
        for _, row in by_year.iterrows():
            print(f"  {int(row['syear']):<8} {int(row['n']):>8,} {row['mean']:>8.3f} {row['std']:>8.3f}")


if __name__ == "__main__":
    main()
