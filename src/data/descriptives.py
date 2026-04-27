"""
Generate LaTeX descriptive statistics tables for study variables.
Outputs a main comparison table and a detailed appendix table.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

SOEP_MISSING = {-8, -7, -6, -5, -4, -3, -2, -1}

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)

MASTER_PATH = Path("output/data/master.parquet")

def load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"{MASTER_PATH} not found — run build_dataframe.py first")
    return pd.read_parquet(MASTER_PATH)

def clean(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s[~s.isin(SOEP_MISSING)]

def compute_stats(df: pd.DataFrame, varname: str, pre_years: list[int], post_years: list[int]) -> dict:
    # Full Sample
    s_all = clean(df[varname])
    
    # Pre-COVID
    s_pre = clean(df[df["syear"].isin(pre_years)][varname])
    
    # Post-COVID
    s_post = clean(df[df["syear"].isin(post_years)][varname])
    
    # Yearly N for Appendix
    by_year = {}
    for yr, grp in df.groupby("syear"):
        s_yr = clean(grp[varname])
        if not s_yr.empty:
            by_year[int(yr)] = int(s_yr.count())
            
    return {
        "full": {"n": int(s_all.count()), "mean": s_all.mean(), "sd": s_all.std(), "min": s_all.min(), "max": s_all.max()},
        "pre":  {"n": int(s_pre.count()), "mean": s_pre.mean(), "sd": s_pre.std()},
        "post": {"n": int(s_post.count()), "mean": s_post.mean(), "sd": s_post.std()},
        "n_by_year": by_year,
    }

def build_main_table(all_stats: dict, var_meta: dict, panels: list[dict], harmonized: set[str]) -> str:
    """Builds a table comparing Pre and Post COVID means/SDs."""
    col_spec = "ll l rr rr r" # Var, Desc, Scale, Pre(M/SD), Post(M/SD), N_Total
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive Statistics: Pre- vs. Post-COVID Comparison}",
        r"\label{tab:descriptives_main}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Pre-COVID} & \multicolumn{2}{c}{Post-COVID} & \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Variable & Description & Scale & Mean & SD & Mean & SD & $N$ \\",
        r"\midrule",
    ]

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta: continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{panel_label}}}}} \\")
        for vdef in var_meta[pk]:
            name = vdef["name"]
            s = all_stats.get(name)
            if not s or s["full"]["n"] == 0: continue
            
            scale = vdef.get("scale", "")
            tex_name = name.replace("_", r"\_")
            dagger = r"$^\dagger$" if name in harmonized else ""
            
            if scale.upper() == "ID":
                pre_cells = "-- & --"
                post_cells = "-- & --"
            else:
                pre_cells = f"{s['pre']['mean']:.2f} & {s['pre']['sd']:.2f}"
                post_cells = f"{s['post']['mean']:.2f} & {s['post']['sd']:.2f}"

            lines.append(
                rf"\texttt{{{tex_name}}}{dagger} & {vdef['label']} & {scale} & "
                rf"{pre_cells} & {post_cells} & {s['full']['n']:,} \\"
            )
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)

def build_appendix_table(all_stats: dict, var_meta: dict, panels: list[dict], years: list[int]) -> str:
    """The original detailed table with yearly N and full descriptives."""
    year_cols = sorted(years)
    col_spec = "llr" + "r" * len(year_cols) + "rrrrr"
    year_header = " & ".join(str(y) for y in year_cols)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detailed Descriptive Statistics (Appendix)}",
        r"\label{tab:descriptives_appendix}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Variable & Description & Scale & {year_header} & $N$ & Mean & SD & Min & Max \\",
        r"\midrule",
    ]

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta: continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(rf"\multicolumn{{{3 + len(year_cols) + 5}}}{{l}}{{\textit{{{panel_label}}}}} \\")
        for vdef in var_meta[pk]:
            name = vdef["name"]
            s = all_stats.get(name)
            if not s or s["full"]["n"] == 0: continue
            
            n_cells = " & ".join(f"{s['n_by_year'].get(y, 0):,}" for y in year_cols)
            scale = vdef.get("scale", "")
            tex_name = name.replace("_", r"\_")
            
            if scale.upper() == "ID":
                stats_cells = "& -- & -- & -- & --"
            else:
                stats_cells = rf"& {s['full']['mean']:.3f} & {s['full']['sd']:.3f} & {s['full']['min']:.0f} & {s['full']['max']:.0f}"

            lines.append(rf"\texttt{{{tex_name}}} & {vdef['label']} & {scale} & {n_cells} & {s['full']['n']:,} {stats_cells} \\")
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines)

def main():
    config = load_config()
    pre_years = config["study"]["pre_covid_years"]
    post_years = config["study"]["post_covid_years"]
    all_study_years = pre_years + post_years
    
    master = load_master()
    var_meta = config["variables"]
    all_stats = {}

    print("Computing stats for all variables ...")
    for panel in var_meta.values():
        for vdef in panel:
            name = vdef["name"]
            if name.lower() not in master.columns: continue
            all_stats[name] = compute_stats(master, name.lower(), pre_years, post_years)

    harmonized = {h["name"] for h in config.get("harmonize", [])}
    panels = config.get("panels", [])

    # Generate Main Table
    main_tex = build_main_table(all_stats, var_meta, panels, harmonized)
    main_path = Path("output/tables/descriptives_main.tex")
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.write_text(main_tex)
    print(f"Wrote {main_path}")

    # Generate Appendix Table
    app_tex = build_appendix_table(all_stats, var_meta, panels, all_study_years)
    app_path = Path("output/tables/descriptives_appendix.tex")
    app_path.write_text(app_tex)
    print(f"Wrote {app_path}")

if __name__ == "__main__":
    main()
