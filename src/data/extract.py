"""
One-time extraction: reads large SOEP CSVs in chunks and writes slim Parquet
files containing only the variables defined in config.json.
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.io import load_config
from src.data.utils import SOEP_MISSING, HOUSEHOLD_DATASETS

CHUNKSIZE = 50_000


def collect_columns(config: dict) -> dict[str, set[str]]:
    """
    Build a mapping of dataset → required column names from config.

    Includes direct variables, harmonize source columns, and derived
    source variables. Variables flagged as ``"derived"`` dataset are
    excluded since they are computed in-memory during the pipeline.

    Parameters
    ----------
    config : dict
        Project config containing ``"variables"``, ``"harmonize"``, and
        ``"derived"`` sections.

    Returns
    -------
    dict[str, set[str]]
        Mapping ``{dataset_name: {column_name, ...}}`` for every dataset
        that needs to be extracted.
    """
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


def expected_parquet_columns(config: dict, dataset: str) -> set[str]:
    """
    Compute the set of columns that should exist in a dataset's Parquet file.

    Used by ``ensure_datasets`` to detect whether a cached Parquet is
    stale and needs re-extraction.

    Parameters
    ----------
    config : dict
        Project config containing ``"variables"`` and ``"harmonize"``
        sections.
    dataset : str
        Name of the SOEP dataset (e.g. ``"pl"``, ``"pgen"``).

    Returns
    -------
    set[str]
        Expected column names including the ID key (``pid`` or ``hid``),
        ``syear``, direct variables, and harmonized output columns.
    """
    harmonized_names = {h["name"] for h in config.get("harmonize", [])}
    harmonized_for_dataset = {h["name"] for h in config.get("harmonize", []) if h["dataset"] == dataset}
    direct = set()
    for panel in config["variables"].values():
        for vdef in panel:
            if vdef["dataset"] == dataset and vdef["name"] not in harmonized_names:
                direct.add(vdef["name"])
    for ddef in config.get("derived", []):
        for src in ddef.get("source_vars", []):
            if src["dataset"] == dataset:
                direct.add(src["name"])
    id_col = "hid" if dataset in HOUSEHOLD_DATASETS else "pid"
    return {id_col, "syear"} | direct | harmonized_for_dataset


def ensure_datasets(config: dict, parquet_dir: str = "output/data") -> None:
    """
    Re-extract any dataset whose Parquet is missing or has stale columns.

    Compares each dataset's existing Parquet schema against the columns
    required by config. Missing or incomplete files are deleted and
    re-extracted from the raw SOEP CSVs.

    Parameters
    ----------
    config : dict
        Project config used to determine required datasets and columns.
    parquet_dir : str, optional
        Directory where Parquet files are stored. Defaults to
        ``"output/data"``.
    """
    out_dir = Path(parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_dataset = collect_columns(config)
    harmonize = config.get("harmonize", [])
    soep_dir = config["data"]["soep_dir"]

    for dataset, columns in by_dataset.items():
        out_path = out_dir / f"{dataset}.parquet"
        expected = expected_parquet_columns(config, dataset)

        if out_path.exists():
            existing = set(pq.read_schema(out_path).names)
            missing = expected - existing
            if not missing:
                continue
            print(f"  {dataset}.parquet missing columns {missing} — re-extracting")
            out_path.unlink()
        else:
            print(f"  {dataset}.parquet not found — extracting")

        extract(soep_dir, dataset, columns, harmonize, out_path)


def apply_harmonize(chunk: pd.DataFrame, harmonize: list[dict], dataset: str) -> pd.DataFrame:
    """
    Compute harmonized columns by coalescing source variables row-wise.

    Each harmonize definition specifies a target column name and one or more
    source variables with optional value recoding. Sources are applied in
    order: the first non-null value wins (``fillna`` chaining). Safe to call
    per-chunk during streaming extraction.

    Parameters
    ----------
    chunk : pd.DataFrame
        A single chunk of a SOEP CSV containing the source columns for
        this dataset's harmonize definitions.
    harmonize : list[dict]
        Full harmonize config list; entries not matching ``dataset`` are
        skipped.
    dataset : str
        Name of the dataset being processed (e.g. ``"pl"``).

    Returns
    -------
    pd.DataFrame
        Chunk with harmonized output columns added in-place.
    """
    for hdef in harmonize:
        if hdef["dataset"] != dataset:
            continue
        name = hdef["name"].lower()
        result = pd.Series(pd.NA, index=chunk.index, dtype="Float64")
        for src in hdef["sources"]:
            col = src["variable"].lower()
            if col not in chunk.columns:
                continue
            if "recode" in src:
                recode = {int(k): v for k, v in src["recode"].items()}
                recoded = chunk[col].map(recode)  # type: ignore[arg-type]
            else:
                recoded = chunk[col]
            result = result.fillna(recoded)
        chunk[name] = result
    return chunk


def extract(
    soep_dir: str,
    dataset: str,
    columns: set[str],
    harmonize: list[dict],
    out_path: Path,
) -> None:
    """
    Stream a SOEP CSV in chunks and write a slim Parquet file.

    Only the columns required by config are kept. SOEP missing-value codes
    are converted to ``pd.NA`` during chunked reading. Harmonized columns
    are computed per-chunk via ``apply_harmonize`` before writing.

    Parameters
    ----------
    soep_dir : str
        Root directory containing the SOEP CSV files.
    dataset : str
        Dataset name (e.g. ``"pl"``); the CSV is expected at
        ``<soep_dir>/<dataset>.csv``.
    columns : set[str]
        Set of raw column names to read from the CSV (before harmonize
        source removal).
    harmonize : list[dict]
        Full harmonize config list; only entries matching ``dataset`` are
        applied.
    out_path : Path
        Destination path for the output Parquet file.
    """
    csv_path = Path(soep_dir) / f"{dataset}.csv"
    id_col = "hid" if dataset in HOUSEHOLD_DATASETS else "pid"
    cols_needed = {id_col, "syear"} | {c.lower() for c in columns}

    harmonize_sources = {
        s["variable"].lower()
        for h in harmonize if h["dataset"] == dataset
        for s in h["sources"]
    }
    keep = {id_col, "syear"} | ({c.lower() for c in columns} - harmonize_sources) | \
           {h["name"].lower() for h in harmonize if h["dataset"] == dataset}

    harmonized_names = [h["name"].lower() for h in harmonize if h["dataset"] == dataset]
    if harmonized_names:
        print(f"  {dataset}.csv → {out_path.name}  (harmonizing: {harmonized_names})")
    else:
        print(f"  {dataset}.csv → {out_path.name}")

    writer = None
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        usecols=lambda c: c.lower() in cols_needed,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )):
        chunk.columns = chunk.columns.str.lower()
        for col in chunk.columns:
            if col in {id_col, "syear"}:
                continue
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk.loc[chunk[col].isin(SOEP_MISSING), col] = pd.NA

        chunk = apply_harmonize(chunk, harmonize, dataset)
        chunk = chunk[[c for c in chunk.columns if c in keep]]

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        if (i + 1) % 10 == 0:
            print(f"    ... {(i + 1) * CHUNKSIZE:,} rows processed")

    if writer:
        writer.close()
    print(f"    done.")


def main() -> None:
    """Extract all datasets defined in config to Parquet, skipping existing files."""
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
