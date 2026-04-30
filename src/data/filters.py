def filter_study_years(df, config):
    """Keeps only rows whose syear falls in the configured pre- and post-covid years."""
    print("\nFiltering to study years ...")
    initial_rows = len(df)
    study_years = config["study"]["pre_covid_years"] + config["study"]["post_covid_years"]
    df = df[df["syear"].isin(study_years)].copy()
    print(f"  Kept years {min(study_years)}–{max(study_years)}: -{initial_rows - len(df):,} rows")
    return df
