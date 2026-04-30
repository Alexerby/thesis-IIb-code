def merge_datasets(frames):
    """Outer merge of all dataframes on pid and syear."""
    print("Merging datasets ...")
    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on=["pid", "syear"], how="outer")
    return master


def compute_age(df):
    """Adds age = syear - gebjahr."""
    print("Computing age ...")
    df = df.copy()
    df["age"] = df["syear"] - df["gebjahr"]
    return df
