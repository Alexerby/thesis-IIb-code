"""
Master Pipeline: Builds the analysis dataframe by chaining modular steps.

Each .pipe() step does exactly one thing to the dataframe.
To add a new step, write a function that takes a df (and optionally config)
and returns a df, then add it as a new .pipe() line below.

Note: harmonized variables (config['harmonize']) are computed during extraction
in extract.py, not here. Re-run extract.py when you add new harmonized variables.
"""

from src.data.io import load_config, load_parquet_datasets, save_master
from src.data.transformers import merge_datasets, merge_household_data, compute_age, compute_sector
from src.data.filters import filter_study_years
from src.data.extract import ensure_datasets


def main():
    config = load_config()

    print("Starting master data pipeline ...")

    ensure_datasets(config)
    person_frames, household_frames = load_parquet_datasets(config)

    (
        merge_datasets(person_frames)
        .pipe(merge_household_data,  frames=household_frames)
        .pipe(compute_age)
        .pipe(compute_sector, config=config)
        .pipe(filter_study_years, config=config)
        .pipe(save_master)
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
