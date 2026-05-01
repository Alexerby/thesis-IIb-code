import json
from pathlib import Path
import pandas as pd

def load_config(root_dir="."):
    config_path = Path(root_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)

def load_parquet_datasets(config, parquet_dir="output/data"):
    """
    Loads all parquet files defined in config variables.
    Returns (person_frames, household_frames) — person frames are keyed by
    (pid, syear); household frames by (hid, syear) and merged separately.
    """
    from src.data.utils import HOUSEHOLD_DATASETS
    parquet_dir = Path(parquet_dir)
    by_dataset = {}
    for panel in config["variables"].values():
        for vdef in panel:
            by_dataset.setdefault(vdef["dataset"], set()).add(vdef["name"].lower())

    person_frames, household_frames = [], []
    for dataset in by_dataset.keys():
        if dataset == "derived":
            continue
        path = parquet_dir / f"{dataset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} - run extract.py first")
        df = pd.read_parquet(path)
        if dataset in HOUSEHOLD_DATASETS:
            household_frames.append(df)
        else:
            person_frames.append(df)
    return person_frames, household_frames

def save_master(df, output_path="output/data/master.parquet"):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    complete = df.dropna()
    xlsx_path = out_path.with_suffix(".xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Full", index=False)
        complete.to_excel(writer, sheet_name="Complete cases", index=False)

    print(f"\nFinal Master dataframe: {len(df):,} rows ({len(complete):,} complete cases)")
    print(f"Wrote {out_path} and {xlsx_path.name}")
    return df
