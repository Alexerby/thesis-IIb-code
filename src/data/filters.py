def apply_sample_restrictions(df, config):
    """Filters data based on study years defined in config."""
    print("\nApplying sample restrictions ...")
    initial_rows = len(df)
    
    pre = config["study"]["pre_covid_years"]
    post = config["study"]["post_covid_years"]
    study_years = pre + post
    
    df = df[df["syear"].isin(study_years)].copy()
    print(f"  Filtered to study years {min(study_years)}–{max(study_years)}: -{initial_rows - len(df):,} rows")
    return df
