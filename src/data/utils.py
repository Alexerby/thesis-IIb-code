import pandas as pd

SOEP_MISSING = {-8, -7, -6, -5, -4, -3, -2, -1}

def clean_series(series: pd.Series) -> pd.Series:
    """Converts to numeric and removes SOEP missing codes."""
    s = pd.to_numeric(series, errors="coerce")
    return s[~s.isin(SOEP_MISSING)]
