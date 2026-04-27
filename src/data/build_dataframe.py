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
MASTER_CSV = PARQUET_DIR / "master.csv"


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def datasets_in_config(config: dict) -> dict[str, list[str]]:
    """Return {dataset: [varnames]} from variable panels, plus derived source vars."""
    by_dataset: dict[str, list[str]] = {}
    for panel in config["variables"].values():
        for vdef in panel:
            if vdef["dataset"] == "derived":
                continue
            by_dataset.setdefault(vdef["dataset"], []).append(vdef["name"].lower())
    for ddef in config.get("derived", []):
        for src in ddef.get("source_vars", []):
            ds, name = src["dataset"], src["name"].lower()
            if name not in by_dataset.get(ds, []):
                by_dataset.setdefault(ds, []).append(name)
    return by_dataset


def compute_derived(master: pd.DataFrame, config: dict) -> pd.DataFrame:
    all_var_names = {
        vdef["name"].lower()
        for panel in config["variables"].values()
        for vdef in panel
        if vdef["dataset"] != "derived"
    }
    for ddef in config.get("derived", []):
        name = ddef["name"].lower()
        master[name] = master.eval(ddef["compute"])
        print(f"  computed '{name}' from: {ddef['compute']}")
    # Drop source-only columns not needed elsewhere
    source_cols = {
        src["name"].lower()
        for ddef in config.get("derived", [])
        for src in ddef.get("source_vars", [])
    }
    drop = source_cols - all_var_names
    master = master.drop(columns=[c for c in drop if c in master.columns])
    return master


def main():
    config = load_config()
    years = config["study"]["years_of_interest"]
    by_dataset = datasets_in_config(config)

    print("Loading extracted parquets ...")
    frames = []
    for dataset, varnames in by_dataset.items():
        path = PARQUET_DIR / f"{dataset}.parquet"
        if not path.exists():
            print(f"  MISSING: {path.name} — run extract.py first")
            return
        cols = ["pid", "syear"] + [v for v in varnames if v not in {"pid", "syear"}]
        # only request columns that actually exist in the file
        available = pd.read_parquet(path, columns=["pid"]).pipe(lambda _: pd.read_parquet(path)).columns.tolist()
        cols = [c for c in cols if c in available]
        print(f"  {path.name}: {len(cols)-2} variables")
        frames.append(pd.read_parquet(path, columns=cols))

    # Merge all datasets on pid + syear
    print("Merging datasets ...")
    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on=["pid", "syear"], how="outer")

    # Filter to study years
    print(f"Filtering to study years {min(years)}–{max(years)} ...")
    master = master[master["syear"].isin(years)].copy()

    # Compute derived variables (e.g. age = syear - gebjahr)
    print("Computing derived variables ...")
    master = compute_derived(master, config)

    # Require non-missing outcome variable
    outcome = config["variables"]["outcome_primary"][0]["name"].lower()
    n_before = len(master)
    master = master[master[outcome].notna()].copy()
    print(f"Dropping {n_before - len(master):,} rows missing primary outcome ({outcome})")

    print(f"\nMaster dataframe: {len(master):,} rows × {len(master.columns)} columns")
    print(f"Years covered: {sorted(master['syear'].unique().tolist())}")
    print(f"Unique individuals: {master['pid'].nunique():,}")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_PARQUET, index=False)
    master.to_csv(MASTER_CSV, index=False)
    print(f"\nWrote {MASTER_PARQUET} and {MASTER_CSV}")


if __name__ == "__main__":
    main()
