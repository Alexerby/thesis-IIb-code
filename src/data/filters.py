import pandas as pd


def filter_study_years(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Keep only rows whose survey year falls within the configured study period.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe containing a ``syear`` column.
    config : dict
        Project config containing ``config["study"]["study_years"]`` as a
        list of integer years.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe restricted to the declared study years.
    """
    print("\nFiltering to study years ...")
    initial_rows = len(df)
    study_years = config["study"]["study_years"]
    df = df[df["syear"].isin(study_years)]
    print(f"  Kept {min(study_years)}–{max(study_years)}: -{initial_rows - len(df):,} rows")
    return df
