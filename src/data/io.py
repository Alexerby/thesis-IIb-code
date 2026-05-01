import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


def load_config(root_dir: str = ".") -> dict:
    """
    Load the project configuration from ``config.json``.

    Parameters
    ----------
    root_dir : str, optional
        Directory containing ``config.json``. Defaults to the current
        working directory.

    Returns
    -------
    dict
        Parsed JSON configuration.
    """
    config_path = Path(root_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def load_parquet_datasets(
    config: dict, parquet_dir: str = "output/data"
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Load all Parquet datasets declared in config variables.

    Datasets are split into person-level (keyed by ``pid`` + ``syear``) and
    household-level (keyed by ``hid`` + ``syear``), which are merged
    separately in the pipeline.

    Parameters
    ----------
    config : dict
        Project config with a ``"variables"`` block mapping panel keys to
        lists of variable definitions, each having ``dataset`` and ``name``
        fields.
    parquet_dir : str, optional
        Directory containing the Parquet files. Defaults to
        ``"output/data"``.

    Returns
    -------
    tuple[list[pd.DataFrame], list[pd.DataFrame]]
        A 2-tuple ``(person_frames, household_frames)`` where each element
        is a list of DataFrames ready for merging.

    Raises
    ------
    FileNotFoundError
        If a required Parquet file is missing (run ``extract.py`` first).
    """
    from src.data.utils import HOUSEHOLD_DATASETS

    parquet_path = Path(parquet_dir)
    by_dataset = {}
    for panel in config["variables"].values():
        for vdef in panel:
            by_dataset.setdefault(vdef["dataset"], set()).add(vdef["name"].lower())
    for ddef in config.get("derived", []):
        for src in ddef.get("source_vars", []):
            by_dataset.setdefault(src["dataset"], set()).add(src["name"].lower())

    person_frames, household_frames = [], []
    for dataset, col_names in by_dataset.items():
        if dataset == "derived":
            continue
        path = parquet_path / f"{dataset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} - run extract.py first")
        id_col = "hid" if dataset in HOUSEHOLD_DATASETS else "pid"
        wanted = {id_col, "syear"} | {c.lower() for c in col_names}
        available = set(pq.read_schema(path).names)
        load_cols = [
            c
            for c in [id_col, "syear"] + sorted(wanted - {id_col, "syear"})
            if c in available
        ]
        df = pd.read_parquet(path, columns=load_cols)
        if dataset in HOUSEHOLD_DATASETS:
            household_frames.append(df)
        else:
            person_frames.append(df)
    return person_frames, household_frames


def save_master(
    df: pd.DataFrame, output_path: str = "output/data/master.parquet"
) -> pd.DataFrame:
    """
    Persist the master dataframe to Parquet and a multi-sheet Excel workbook.

    Three Excel sheets are written:

    - ``Full`` — entire dataframe.
    - ``Complete cases`` — rows with no missing values.
    - ``Eligible`` — rows where ``plb0097`` is 1 or 2 (excludes "Not possible").

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe to save.
    output_path : str, optional
        Destination path for the Parquet file. The Excel file is written
        to the same location with a ``.xlsx`` extension. Defaults to
        ``"output/data/master.parquet"``.

    Returns
    -------
    pd.DataFrame
        The input dataframe unchanged, enabling use in a ``.pipe()`` chain.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    xlsx_path = out_path.with_suffix(".xlsx")
    complete = df.dropna()
    eligible = df[df["plb0097"].isin([1, 2])]  # Filter out plb0097=3
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Full", index=False)
        complete.to_excel(writer, sheet_name="Complete cases", index=False)
        eligible.to_excel(writer, sheet_name="Eligible", index=False)
    complete_count = len(complete)
    print(
        f"\nFinal Master dataframe: {len(df):,} rows ({complete_count:,} complete cases)"
    )
    print(f"Wrote {out_path} and {xlsx_path.name}")
    return df
