"""
Master Pipeline: Builds the analysis dataframe by chaining modular steps.

Each .pipe() step does exactly one thing to the dataframe.
To add a new step, write a function that takes a df (and optionally config)
and returns a df, then add it as a new .pipe() line below.

Note: harmonized variables (config['harmonize']) are computed during extraction
in extract.py, not here. Re-run extract.py when you add new harmonized variables.
"""

from src.data.io import load_config, load_parquet_datasets, save_master
from src.data.transformers import merge_datasets, compute_age
from src.data.filters import filter_study_years

def main():
    config = load_config()

    print("Starting master data pipeline ...")

    datasets = load_parquet_datasets(config)

    (
        merge_datasets(datasets)
        .pipe(compute_age)
        .pipe(filter_study_years, config=config)
        .pipe(save_master)
    )

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
