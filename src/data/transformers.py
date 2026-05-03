from typing import Any
import pandas as pd

# types
Recodes = dict[str, dict[Any, Any]]


def merge_datasets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-merge all person-level DataFrames on ``pid`` and ``syear``.

    An outer join preserves every observation that appears in any individual
    dataset, filling missing columns with NaN.

    Parameters
    ----------
    frames : list[pd.DataFrame]
        Person-level DataFrames, each containing ``pid`` and ``syear``
        columns.

    Returns
    -------
    pd.DataFrame
        Single merged DataFrame with one row per unique ``(pid, syear)``
        combination across all input frames.
    """
    print("Merging datasets ...")
    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on=["pid", "syear"], how="outer")
    return master


def merge_household_data(df: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Left-join household-level DataFrames onto the master person DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Master person-level DataFrame containing ``hid`` and ``syear``.
    frames : list[pd.DataFrame]
        Household-level DataFrames, each keyed by ``(hid, syear)``.

    Returns
    -------
    pd.DataFrame
        Master DataFrame with household columns appended. Person rows
        without a matching household record receive NaN.
    """
    if not frames:
        return df
    print("Merging household datasets ...")
    for hdf in frames:
        df = df.merge(hdf, on=["hid", "syear"], how="left")
    return df


def compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive respondent age and drop the birth-year column.

    Age is computed as ``syear - gebjahr`` and stored in a new ``age``
    column. The source column ``gebjahr`` is then removed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``syear`` (survey year) and ``gebjahr``
        (birth year) columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``age`` added and ``gebjahr`` dropped.
    """
    print("Computing age ...")
    df["age"] = df["syear"] - df["gebjahr"]
    df = df.drop(columns=["gebjahr"])
    return df


def compute_sector(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Map NACE industry codes to a common sector identifier and drop source columns.

    Both NACE revisions use incompatible numeric codes, so each is mapped
    separately before coalescing — Rev.1 (``p_nace``) takes priority where
    both are present. The raw source columns are dropped after mapping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``p_nace`` (NACE Rev.1) and/or ``p_nace2``
        (NACE Rev.2) columns.
    config : dict
        Project config with a ``"nace_sectors"`` list; each entry must
        have ``id``, ``nace_rev1`` (list of int codes), and ``nace_rev2``
        (list of int codes).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new nullable ``sector`` (Int64) column and
        ``p_nace`` / ``p_nace2`` removed.
    """
    print("Computing industry sector ...")

    rev1_map = {}
    rev2_map = {}
    for s in config.get("nace_sectors", []):
        for code in s["nace_rev1"]:
            rev1_map[code] = s["id"]
        for code in s["nace_rev2"]:
            rev2_map[code] = s["id"]

    sector = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "p_nace" in df.columns:
        sector = sector.fillna(df["p_nace"].map(rev1_map))
    if "p_nace2" in df.columns:
        sector = sector.fillna(df["p_nace2"].map(rev2_map))

    df["sector"] = sector
    df = df.drop(columns=[c for c in ["p_nace", "p_nace2"] if c in df.columns])
    print(f"  {sector.notna().sum():,} non-null values across {sector.nunique()} sectors")
    return df


def compute_sqm_per_head(df: pd.DataFrame) -> pd.DataFrame:
    """Derive living space per person by dividing hlf0019_h by d11106."""
    print("Computing sqm per head ...")
    hh_size = df["d11106"].where(df["d11106"] > 0)
    df["sqm_per_head"] = df["hlf0019_h"] / hh_size
    return df


def compute_migback_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the ``migback`` column with two binary dummy columns.

    Reference category is 1 (no migration background). The source column
    is dropped after the dummies are created.
    """
    print("Computing migration background dummies ...")
    df["migback_direct"]   = df["migback"].eq(2).astype("Int8")
    df["migback_indirect"] = df["migback"].eq(3).astype("Int8")
    df = df.drop(columns=["migback"])
    return df


def recode_variables(df: pd.DataFrame, recodes: Recodes) -> pd.DataFrame:
    """
    Apply value remappings to one or more columns in a single pass.

    Each entry in ``recodes`` maps a column name to a ``{old: new}`` dict
    that is forwarded to ``pd.Series.replace``. Columns absent from the
    dataframe are silently skipped, so you can declare recodes for optional
    variables without guard clauses.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe after merging.
    recodes : dict[str, dict[Any, Any]]
        Mapping of ``{column_name: {old_value: new_value}}``. Values of
        ``None`` become ``NaN`` in the output column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified values replaced in-place.
    """
    for col, mapping in recodes.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df
