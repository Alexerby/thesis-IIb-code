import pandas as pd

def merge_datasets(frames):
    """Outer merge of all dataframes on pid and syear."""
    print("Merging datasets ...")
    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on=["pid", "syear"], how="outer")
    return master


def merge_household_data(df, frames):
    """Left-joins household-level datasets on (hid, syear)."""
    if not frames:
        return df
    print("Merging household datasets ...")
    for hdf in frames:
        df = df.merge(hdf, on=["hid", "syear"], how="left")
    return df

def compute_age(df):
    """Adds age = syear - gebjahr."""
    print("Computing age ...")
    df = df.copy()
    df["age"] = df["syear"] - df["gebjahr"]
    return df

def compute_sector(df, config):
    """
    Maps p_nace (NACE Rev.1) and p_nace2 (NACE Rev.2) to a common sector id
    using the nace_sectors mapping in config, then drops the source columns.

    The two revisions use incompatible numeric codes, so each is mapped
    separately before coalescing (Rev.1 takes priority where both exist).
    To change groupings, edit config['nace_sectors'] and re-run.
    """
    print("Computing industry sector ...")
    df = df.copy()

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
