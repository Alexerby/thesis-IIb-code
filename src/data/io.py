import json
from pathlib import Path
import pandas as pd

def load_config(root_dir="."):
    config_path = Path(root_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)

def load_parquet_datasets(config, parquet_dir="output/data"):
    """Loads all parquet files defined in config variables."""
    parquet_dir = Path(parquet_dir)
    by_dataset = {}
    for panel in config["variables"].values():
        for vdef in panel:
            by_dataset.setdefault(vdef["dataset"], set()).add(vdef["name"].lower())
            
    frames = []
    for dataset in by_dataset.keys():
        if dataset == "derived": continue
        path = parquet_dir / f"{dataset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} - run extract.py first")
        df = pd.read_parquet(path)
        frames.append(df)
    return frames

def save_master(df, output_path="output/data/master.parquet"):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"\nFinal Master dataframe: {len(df):,} rows")
    print(f"Wrote {out_path}")
    return df
