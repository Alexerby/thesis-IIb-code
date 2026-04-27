"""
Generate a LaTeX descriptive statistics table for all study variables.

Usage:
    python src/descriptives.py                        # study years only (from config)
    python src/descriptives.py --all-years            # full panel
    python src/descriptives.py --out output/tables/descriptives.tex
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SOEP_MISSING = {-8, -7, -6, -5, -4, -3, -2, -1}


def load_config():
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


PARQUET_DIR = Path("output/data")


def load_dataset(soep_dir: str, dataset: str, variables: list[str]) -> pd.DataFrame:
    parquet_path = PARQUET_DIR / f"{dataset}.parquet"
    cols_needed = ["pid", "syear"] + [v.lower() for v in variables]
    if parquet_path.exists():
        print(f"  Loading {dataset}.parquet ...")
        return pd.read_parquet(parquet_path, columns=cols_needed)
    # Fallback to CSV — slow on large files
    path = Path(soep_dir) / f"{dataset}.csv"
    print(
        f"  Loading {dataset}.csv (no parquet found — run extract.py first for large files) ..."
    )
    df = pd.read_csv(
        path, usecols=lambda c: c.lower() in set(cols_needed), low_memory=False
    )
    df.columns = df.columns.str.lower()
    return df


def clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s[~s.isin(SOEP_MISSING)]


def compute_stats(df: pd.DataFrame, varname: str, years: list[int] | None) -> dict:
    sub = df if years is None else df[df["syear"].isin(years)]
    s = clean(sub[varname])
    if s.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_by_year": {},
        }
    by_year = {}
    for yr, grp in sub.groupby("syear"):
        s_yr = clean(grp[varname])
        if not s_yr.empty:
            by_year[int(yr)] = int(s_yr.count())
    return {
        "n": int(s.count()),
        "mean": float(s.mean()),
        "sd": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "n_by_year": by_year,
    }


PANEL_LABELS = {
    "outcome_primary": "Panel A: Primary Outcome",
    "outcome_secondary": "Panel B: Secondary Outcomes",
    "treatment": "Panel C: Treatment Variables",
    "mechanisms": "Panel D: Mechanisms",
}

PANEL_ORDER = ["outcome_primary", "outcome_secondary", "treatment", "mechanisms"]


def build_latex(all_stats: dict, var_meta: dict, years: list[int] | None, harmonized: set[str]) -> str:
    year_cols = sorted(
        {yr for stats in all_stats.values() for yr in stats["n_by_year"]}
    )
    if years:
        year_cols = [y for y in year_cols if y in years]

    col_spec = "llr" + "r" * len(year_cols) + "rrrrr"
    year_header = " & ".join(str(y) for y in year_cols)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive Statistics}",
        r"\label{tab:descriptives}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Variable & Description & Scale & {year_header} & $N$ & Mean & SD & Min & Max \\",
        r"\midrule",
    ]

    for panel_key in PANEL_ORDER:
        if panel_key not in var_meta:
            continue
        lines.append(
            rf"\multicolumn{{{3 + len(year_cols) + 5}}}{{l}}{{\textit{{{PANEL_LABELS[panel_key]}}}}} \\"
        )
        for vdef in var_meta[panel_key]:
            name = vdef["name"]
            stats = all_stats.get(name)
            if stats is None or stats["n"] == 0:
                continue
            n_cells = " & ".join(
                f"{stats['n_by_year'][y]:,}" if y in stats["n_by_year"] else "0"
                for y in year_cols
            )
            scale = vdef.get("scale", "")
            tex_name = name.replace("_", r"\_")
            dagger = r"$^\dagger$" if name in harmonized else ""
            lines.append(
                rf"\texttt{{{tex_name}}}{dagger} & {vdef['label']} & {scale} & "
                rf"{n_cells} & "
                rf"{stats['n']:,} & {stats['mean']:.3f} & {stats['sd']:.3f} & "
                rf"{stats['min']:.0f} & {stats['max']:.0f} \\"
            )
        lines.append(r"\addlinespace")

    year_note = f"{min(year_cols)}--{max(year_cols)}" if year_cols else "all years"
    harmonize_note = (
        r" $^\dagger$Manually harmonized from multiple survey versions."
        if harmonized else ""
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\\\small Note: Valid observations only (SOEP missing codes $-1$ to $-8$ excluded). Study period: {year_note}.{harmonize_note}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-years", action="store_true", help="Use full panel instead of study years"
    )
    parser.add_argument("--out", default="output/tables/descriptives.tex")
    args = parser.parse_args()

    config = load_config()
    soep_dir = config["data"]["soep_dir"]
    years = None if args.all_years else config["study"]["years_of_interest"]
    var_meta = config["variables"]

    # Group variables by dataset so each file is loaded only once
    by_dataset: dict[str, list[str]] = {}
    for panel in var_meta.values():
        for vdef in panel:
            by_dataset.setdefault(vdef["dataset"], []).append(vdef["name"])

    print("Loading datasets ...")
    frames: dict[str, pd.DataFrame] = {}
    for dataset, varnames in by_dataset.items():
        frames[dataset] = load_dataset(soep_dir, dataset, varnames)

    print("Computing statistics ...")
    all_stats: dict[str, dict] = {}
    for panel in var_meta.values():
        for vdef in panel:
            name = vdef["name"]
            df = frames[vdef["dataset"]]
            if name.lower() not in df.columns:
                print(
                    f"  WARNING: {name} not found in {vdef['dataset']}.csv — skipping"
                )
                continue
            all_stats[name] = compute_stats(df, name.lower(), years)

    harmonized = {h["name"] for h in config.get("harmonize", [])}
    latex = build_latex(all_stats, var_meta, years, harmonized)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
