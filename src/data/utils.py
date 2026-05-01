import pandas as pd

SOEP_MISSING = {-8, -7, -6, -5, -4, -3, -2, -1}

# Datasets keyed by (hid, syear) rather than (pid, syear)
HOUSEHOLD_DATASETS = {"hl", "hgen"}

def clean_series(series: pd.Series) -> pd.Series:
    """Converts to numeric and removes SOEP missing codes."""
    s = pd.to_numeric(series, errors="coerce")
    return s[~s.isin(SOEP_MISSING)]

def regime_labels(config: dict, latex: bool = False) -> tuple[str, str]:
    """
    Returns (r1_label, r2_label) derived from config study years.
    Use latex=True for LaTeX output (-- dash), False for display (– en dash).
    """
    dash = "--" if latex else "–"
    pre  = config["study"]["pre_covid_years"]
    post = config["study"]["post_covid_years"]
    return (
        f"Regime 1 ({min(pre)}{dash}{max(pre)})",
        f"Regime 2 ({min(post)}{dash}{max(post)})",
    )
