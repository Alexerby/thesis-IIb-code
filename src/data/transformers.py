import pandas as pd

def merge_datasets(frames):
    """Outer merge of all dataframes on pid and syear."""
    print("Merging datasets ...")
    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on=["pid", "syear"], how="outer")
    return master

def compute_derived_variables(df, config):
    """Computes variables defined in config['derived']."""
    print("Computing derived variables ...")
    df = df.copy()
    for ddef in config.get("derived", []):
        if ddef["name"] == "age":
            if "gebjahr" in df.columns:
                df["age"] = df["syear"] - df["gebjahr"]
                print("  derived: age = syear - gebjahr")
    return df
