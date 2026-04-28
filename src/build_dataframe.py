"""
Master Pipeline: Builds the analysis dataframe by chaining modular steps.
"""

from src.data.io import load_config, load_parquet_datasets, save_master
from src.data.transformers import merge_datasets, compute_derived_variables
from src.data.filters import apply_sample_restrictions

def main():
    # 1. Setup
    config = load_config()
    
    # 2. Pipeline Execution using .pipe() pattern
    print("Starting master data pipeline ...")
    
    # Load separate parquets
    datasets = load_parquet_datasets(config)
    
    # Chain processing steps
    master = (
        merge_datasets(datasets)
        .pipe(compute_derived_variables, config=config)
        .pipe(apply_sample_restrictions, config=config)
        .pipe(save_master)
    )

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
