"""
One-time extraction: reads large SOEP CSVs in chunks and writes slim Parquet
files containing only the variables defined in config.json.
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.io import load_config
from src.data.utils import SOEP_MISSING

CHUNKSIZE = 50_000

def collect_columns(config: dict) -> dict[str, set[str]]:
    """Return {dataset: {columns}} needed, including harmonize and derived sources."""
    harmonized_names = {h["name"] for h in config.get("harmonize", [])}
    by_dataset: dict[str, set[str]] = {}
    for panel in config["variables"].values():
        for vdef in panel:
            if vdef["dataset"] == "derived":
                continue
            if vdef["name"] in harmonized_names:
                continue
            by_dataset.setdefault(vdef["dataset"], set()).add(vdef["name"])
    for hdef in config.get("harmonize", []):
        for src in hdef["sources"]:
            by_dataset.setdefault(hdef["dataset"], set()).add(src["variable"])
    for ddef in config.get("derived", []):
        for src in ddef.get("source_vars", []):
            by_dataset.setdefault(src["dataset"], set()).add(src["name"])
    return by_dataset

def apply_harmonize(df: pd.DataFrame, harmonize: list[dict], dataset: str) -> pd.DataFrame:
    """Compute harmonized columns in-place and return the dataframe."""
    for hdef in harmonize:
        if hdef["dataset"] != dataset:
            continue
        name = hdef["name"].lower()
        result = pd.Series(pd.NA, index=df.index, dtype="Float64")
        for src in hdef["sources"]:
            col = src["variable"].lower()
            if col not in df.columns:
                continue
            recode = {int(k): v for k, v in src["recode"].items()}
            recoded = df[col].map(recode)
            result = result.combine_first(recoded)
        df[name] = result
        print(f"    harmonized {name} from {[s['variable'] for s in hdef['sources']]}")
    return df

def extract(soep_dir: str, dataset: str, columns: set[str], harmonize: list[dict], out_path: Path):
    csv_path = Path(soep_dir) / f"{dataset}.csv"
    cols_needed = {"pid", "syear"} | {c.lower() for c in columns}

    print(f"  {dataset}.csv → {out_path.name}")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        usecols=lambda c: c.lower() in cols_needed,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )):
        chunk.columns = chunk.columns.str.lower()
        for col in chunk.columns:
            if col in {"pid", "syear"}:
                continue
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk.loc[chunk[col].isin(SOEP_MISSING), col] = pd.NA
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"    ... {(i + 1) * CHUNKSIZE:,} rows processed")

    df = pd.concat(chunks, ignore_index=True)
    df = apply_harmonize(df, harmonize, dataset)

    harmonize_sources = {
        s["variable"].lower()
        for h in harmonize if h["dataset"] == dataset
        for s in h["sources"]
    }
    keep = {"pid", "syear"} | ({c.lower() for c in columns} - harmonize_sources) | \
           {h["name"].lower() for h in harmonize if h["dataset"] == dataset}
    df = df[[c for c in df.columns if c in keep]]

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)
    print(f"    done.")

def main():
    config = load_config()
    soep_dir = config["data"]["soep_dir"]
    out_dir = Path("output/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = collect_columns(config)
    harmonize = config.get("harmonize", [])

    print("Extracting datasets to Parquet ...")
    for dataset, columns in by_dataset.items():
        out_path = out_dir / f"{dataset}.parquet"
        if out_path.exists():
            print(f"  {out_path.name} already exists — skipping")
            continue
        extract(soep_dir, dataset, columns, harmonize, out_path)

if __name__ == "__main__":
    main()
