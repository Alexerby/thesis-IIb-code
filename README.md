# Thesis IIb: Remote Work Determinants (2009-2022)

This repository contains the data pipeline and analysis code for a thesis on the shifting determinants of remote work willingness and access using German SOEP data.

## 🚀 Quick Start

The entire analysis can be run via the main orchestrator:

```bash
python3 src/main.py
```

### Main Entry Points
- `src/main.py`: The master orchestrator (runs everything in order).
- `src/data/extract.py`: One-time extraction from raw SOEP CSV files to slim Parquet files.
- `src/build_dataframe.py`: Master pipeline that merges, cleans, and filters the data into `master.parquet`.

## 📂 Project Structure

### Data Package (`src/data/`)
The data processing logic is modularized into a sub-package:
- `io.py`: Loading and saving data (Config, Parquet).
- `transformers.py`: Data merging and derived variable logic (e.g., `age`).
- `filters.py`: Sample restrictions (filtering by study years).
- `utils.py`: Shared SOEP utilities (missing code handling).

### Analysis & Outputs
- `src/data/descriptives.py`: Generates `descriptives_main.tex` and `descriptives_appendix.tex`.
- `src/data/map_plot.py`: Generates the Pre/Post-COVID comparison map.
- `src/data/plots.py`: Generates trend line plots.
- `src/models/`: Contains OLS and Selection models.

## 🛠 Configuration
All study parameters, variable definitions, and period ranges are defined in `config.json`.
- `pre_covid_years`: [2009, 2010, 2011, 2012, 2013, 2014]
- `post_covid_years`: [2021, 2022]

## 📊 Data Pipeline
The master dataframe is built using a functional pipeline in `src/build_dataframe.py`:
```python
master = (
    merge_datasets(datasets)
    .pipe(compute_derived_variables, config=config)
    .pipe(apply_sample_restrictions, config=config)
    .pipe(save_master)
)
```
