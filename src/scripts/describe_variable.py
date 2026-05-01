"""
Descriptive statistics for a SOEP-Core variable.

Usage:
    python src/scripts/describe_variable.py --variable p11101
    python src/scripts/describe_variable.py --variable p11101 --dataset pequiv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd


SOEP_MISSING = {-8, -7, -6, -5, -4, -3, -2, -1}
CHUNKSIZE = 50_000


def load_config() -> dict:
    """
    Load the project configuration from ``config.json``.

    Returns
    -------
    dict
        Parsed JSON configuration located at the project root (three
        directories above this script).
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def find_dataset_in_codebook(
    soep_dir: str,
    variable: str,
    prefer: set[str] | None = None,
) -> str | None:
    """
    Search SOEP codebook CSVs for the dataset that contains a variable.

    Preferred datasets (e.g. key annual files) are searched first so that
    the most relevant dataset is returned when a variable appears in
    multiple files.

    Parameters
    ----------
    soep_dir : str
        Root directory containing the SOEP ``*_variables.csv`` codebook
        files.
    variable : str
        Variable name to search for (case-insensitive).
    prefer : set[str] or None, optional
        Dataset names to search before all others. Defaults to None
        (alphabetical order).

    Returns
    -------
    str or None
        Dataset name (e.g. ``"pl"``) if found, otherwise ``None``.
    """
    all_paths = sorted(Path(soep_dir).glob("*_variables.csv"))
    # Search preferred datasets first, then the rest
    if prefer:
        ordered = [p for p in all_paths if p.stem.replace("_variables", "") in prefer]
        ordered += [p for p in all_paths if p not in ordered]
    else:
        ordered = all_paths
    for path in ordered:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 4 and row[3].lower() == variable.lower():
                    return row[1]
    return None


def find_csv_path(config: dict, dataset: str) -> Path | None:
    """
    Locate the raw CSV file for a SOEP dataset.

    Checks ``config["data"]["key_files"]`` for an explicit filename
    override before falling back to ``<dataset>.csv``.

    Parameters
    ----------
    config : dict
        Project config containing ``config["data"]["soep_dir"]`` and
        optionally ``config["data"]["key_files"]``.
    dataset : str
        Dataset name (e.g. ``"pl"``).

    Returns
    -------
    Path or None
        Absolute path to the CSV if it exists, otherwise ``None``.
    """
    soep_dir = Path(config["data"]["soep_dir"])
    # Try key_files mapping first, then fall back to <dataset>.csv
    filename = config["data"].get("key_files", {}).get(dataset, f"{dataset}.csv")
    path = soep_dir / filename
    return path if path.exists() else None


def get_value_labels(soep_dir: str, variable: str, dataset: str) -> dict[int, str]:
    """
    Extract value labels for a variable from the SOEP codebook.

    Parameters
    ----------
    soep_dir : str
        Root directory containing the SOEP ``*_values.csv`` codebook
        files.
    variable : str
        Variable name to look up (case-insensitive).
    dataset : str
        Dataset name; only ``<dataset>_values.csv`` is searched.

    Returns
    -------
    dict[int, str]
        Mapping of integer code → English label string. Returns an empty
        dict if the values file does not exist or no labels are found.
    """
    labels: dict[int, str] = {}
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


def read_variable(csv_path: Path, variable: str) -> pd.DataFrame:
    """
    Read a single variable from a large SOEP CSV in chunks.

    Reads ``pid``, ``syear``, and ``variable`` columns only. SOEP
    system-missing codes are converted to ``NaN``.

    Parameters
    ----------
    csv_path : Path
        Path to the SOEP dataset CSV file.
    variable : str
        Variable name to extract (case-insensitive).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[pid, syear, variable]`` containing all
        rows concatenated across chunks.
    """
    cols = ["pid", "syear", variable]
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c.lower() in {v.lower() for v in cols},
                             chunksize=CHUNKSIZE, low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        if variable not in chunk.columns:
            return pd.DataFrame(columns=list(chunk.columns))
        chunk[variable] = pd.to_numeric(chunk[variable], errors="coerce")
        chunk.loc[chunk[variable].isin(SOEP_MISSING), variable] = float("nan")
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=cols)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    """CLI entry point: print descriptive statistics for a SOEP variable."""
    parser = argparse.ArgumentParser(description="Describe a SOEP-Core variable")
    parser.add_argument("--variable", "-v", required=True, help="Variable name (e.g. p11101)")
    parser.add_argument("--dataset", "-d", help="Dataset to load from (auto-detected if omitted)")
    args = parser.parse_args()

    config = load_config()
    soep_dir = config["data"]["soep_dir"]
    variable = args.variable.lower()

    key_datasets = set(config["data"].get("key_files", {}).keys())
    dataset = args.dataset or find_dataset_in_codebook(soep_dir, variable, prefer=key_datasets)
    if not dataset:
        print(f"Could not find '{args.variable}' in any SOEP codebook. Pass --dataset explicitly.")
        sys.exit(1)

    csv_path = find_csv_path(config, dataset)
    if not csv_path:
        print(f"CSV for dataset '{dataset}' not found. Check config.json data paths.")
        sys.exit(1)

    print(f"Reading '{variable}' from {csv_path.name} ...")
    try:
        df = read_variable(csv_path, variable)
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    if variable not in df.columns:
        print(f"Column '{variable}' not found in {csv_path.name}.")
        sys.exit(1)

    df_valid = df[df[variable].notna()]

    print(f"\n{'='*60}")
    print(f"  Variable : {args.variable}  |  Dataset: {dataset}")
    print(f"{'='*60}")

    labels = get_value_labels(soep_dir, variable, dataset)
    if labels:
        positive = {k: v for k, v in labels.items() if k >= 0}
        if positive:
            lo, hi = min(positive), max(positive)
            print(f"  Scale    : {lo} = {positive[lo]}  →  {hi} = {positive[hi]}")

    total_rows = len(df)
    print(f"\n  Total rows in file : {total_rows:,}")
    print(f"  Valid observations : {len(df_valid):,}")
    print(f"  Missing / coded    : {total_rows - len(df_valid):,}  ({(total_rows - len(df_valid)) / total_rows * 100:.1f}%)")

    print(f"\n--- Descriptive statistics (valid obs) ---")
    desc = df_valid[variable].describe(percentiles=[0.25, 0.5, 0.75])
    for stat, val in desc.items():
        print(f"  {stat:<8} {val:.4f}")

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
