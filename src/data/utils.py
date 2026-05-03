from pathlib import Path
import pandas as pd


SOEP_MISSING = {-8, -7, -6, -5, -4, -3, -2, -1}

# Datasets keyed by (hid, syear) rather than (pid, syear)
HOUSEHOLD_DATASETS = {"hl", "hgen"}


def clean_series(series: pd.Series) -> pd.Series:
    """
    Convert a Series to numeric and remove SOEP system-missing codes.

    Parameters
    ----------
    series : pd.Series
        Raw series, possibly containing SOEP missing-value codes (-1 to -8).

    Returns
    -------
    pd.Series
        Numeric series with SOEP missing codes replaced by NaN and dropped.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s[~s.isin(SOEP_MISSING)]  # type: ignore[index]


def study_period_label(config: dict, latex: bool = False) -> str:
    """
    Build a formatted study-period string from config study_years.

    Parameters
    ----------
    config : dict
        Project config containing ``config["study"]["study_years"]``.
    latex : bool, optional
        If True, uses LaTeX en-dash ``--``; otherwise uses Unicode ``–``.
        Default is False.

    Returns
    -------
    str
        Period label, e.g. ``"2009–2014"`` or ``"2009--2014"`` for LaTeX.
    """
    dash = "--" if latex else "–"
    years = config["study"]["study_years"]
    return f"{min(years)}{dash}{max(years)}"


MASTER_PATH = Path("output/data/master.parquet")


def load_master() -> pd.DataFrame:
    """
    Load the master analysis dataframe from Parquet.

    Returns
    -------
    pd.DataFrame
        Master dataframe produced by ``build_dataframe.py``.

    Raises
    ------
    FileNotFoundError
        If the master Parquet file does not exist.
    """
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_PATH} not found — run build_dataframe.py first"
        )
    return pd.read_parquet(MASTER_PATH)
