"""
Builds the master analysis dataframe from extracted parquet files.

Merges all datasets defined in config.json on pid + syear,
applies sample restrictions, and writes output/data/master.parquet.

Run after extract.py whenever config or sample criteria change:
    python src/build_dataframe.py
"""

import json
from pathlib import Path

import pandas as pd

PARQUET_DIR = Path("output/data")
MASTER_PARQUET = PARQUET_DIR / "master.parquet"
MASTER_PARQUET = PARQUET_DIR / "master.parquet"


def load_config():
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def datasets_in_config(config: dict) -> dict[str, list[str]]:
    """Return {dataset: [varnames]} from all variable panels."""
    by_dataset: dict[str, list[str]] = {}
    for panel in config["variables"].values():
        for vdef in panel:
            by_dataset.setdefault(vdef["dataset"], []).append(vdef["name"].lower())
    return by_dataset


def main():
    config = load_config()
    study_years = config["study"]["pre_covid_years"] + config["study"]["post_covid_years"]
    by_dataset = datasets_in_config(config)

    print("Loading extracted parquets ...")
    frames = []
    for dataset in by_dataset.keys():
        if dataset == "derived":
            continue
        path = PARQUET_DIR / f"{dataset}.parquet"
        if not path.exists():
            print(f"  MISSING: {path.name} — run extract.py first")
            return
        
        # Load the whole file to ensure we get all extracted columns 
        # (including source variables for derived ones)
        df = pd.read_parquet(path)
        print(f"  {path.name}: {len(df.columns) - 2} variables")
        frames.append(df)

    # Merge all datasets on pid + syear
    print("Merging datasets ...")
    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on=["pid", "syear"], how="outer")

    # Handle derived variables
    print("Computing derived variables ...")
    for ddef in config.get("derived", []):
        if ddef["name"] == "age":
            # gebjahr is in ppathl
            if "gebjahr" in master.columns:
                master["age"] = master["syear"] - master["gebjahr"]
                print("  derived age = syear - gebjahr")

    # --- Sample Restrictions ---
    print("\nApplying sample restrictions ...")
    initial_rows = len(master)
    
    # 1. Study years
    master = master[master["syear"].isin(study_years)].copy()
    print(f"  Filtered to study years {min(study_years)}–{max(study_years)}: -{initial_rows - len(master):,} rows")

    print(f"\nFinal Master dataframe: {len(master):,} rows × {len(master.columns)} columns")
    print(f"Years covered: {sorted(master['syear'].unique().tolist())}")
    print(f"Unique individuals: {master['pid'].nunique():,}")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_PARQUET, index=False)
    master.to_csv(MASTER_PARQUET.with_suffix(".csv"), index=False)
    print(f"\nWrote {MASTER_PARQUET}")


if __name__ == "__main__":
    main()
