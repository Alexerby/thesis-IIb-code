"""
Main entry point for the thesis-IIb analysis pipeline.
Orchestrates data extraction, dataframe building, descriptives, and modeling.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path: str, args: list[str] | None = None) -> None:
    """
    Run a pipeline sub-script as a subprocess and exit on failure.

    The current project root is added to ``PYTHONPATH`` so that
    ``src.*`` imports resolve correctly in each sub-script.

    Parameters
    ----------
    script_path : str
        Absolute or relative path to the Python script to execute.
    args : list[str] or None, optional
        Additional command-line arguments forwarded to the script.
        Defaults to None (no extra arguments).

    Raises
    ------
    SystemExit
        Propagates the sub-script's non-zero exit code on failure.
    """
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)

    cmd = [sys.executable, script_path] + (args or [])
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    """Parse CLI flags and run the full analysis pipeline in order."""
    parser = argparse.ArgumentParser(description="Thesis IIb Analysis Pipeline")
    parser.add_argument(
        "--extract", action="store_true", help="Run data extraction from raw SOEP CSVs"
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete existing parquets and force re-extraction",
    )
    parser.add_argument(
        "--no-model", action="store_true", help="Skip running the OLS model"
    )
    args = parser.parse_args()

    # Define paths
    src_dir = Path(__file__).parent
    data_dir = src_dir / "data"
    model_dir = src_dir / "models"
    parquet_dir = Path("output/data")

    # Extraction
    if args.force_extract:
        print("Cleaning up old parquet files for forced re-extraction...")
        for p in parquet_dir.glob("*.parquet"):
            p.unlink()
        run_script(str(data_dir / "extract.py"))
    elif args.extract:
        run_script(str(data_dir / "extract.py"))
    else:
        # Check if any parquets exist; if not, suggest extraction
        if not any(parquet_dir.glob("*.parquet")):
            print("No extracted data found. Running extraction...")
            run_script(str(data_dir / "extract.py"))

    # Build Master Dataframe
    run_script(str(src_dir / "build_dataframe.py"))

    # Generate Descriptive Statistics
    run_script(str(data_dir / "descriptives.py"))

    # Generate Plots
    run_script(str(data_dir / "plots.py"))

    print("\n" + "=" * 40)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
