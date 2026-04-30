"""
Generate LaTeX descriptive statistics tables for study variables.
Outputs a main comparison table and a detailed appendix table.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data.io import load_config
from src.data.utils import clean_series

MASTER_PATH = Path("output/data/master.parquet")

def load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"{MASTER_PATH} not found — run build_dataframe.py first")
    return pd.read_parquet(MASTER_PATH)

def compute_stats(df: pd.DataFrame, varname: str, pre_years: list[int], post_years: list[int], vtype: str) -> dict:
    s_all = clean_series(df[varname])
    s_pre = clean_series(df[df["syear"].isin(pre_years)][varname])
    s_post = clean_series(df[df["syear"].isin(post_years)][varname])
    
    by_year = {}
    for yr, grp in df.groupby("syear"):
        s_yr = clean_series(grp[varname])
        if not s_yr.empty:
            by_year[int(yr)] = int(s_yr.count())
            
    res = {
        "full": {"n": int(s_all.count()), "mean": s_all.mean(), "sd": s_all.std(), "min": s_all.min(), "max": s_all.max()},
        "pre":  {"n": int(s_pre.count()), "mean": s_pre.mean(), "sd": s_pre.std()},
        "post": {"n": int(s_post.count()), "mean": s_post.mean(), "sd": s_post.std()},
        "n_by_year": by_year,
    }

    if vtype == "categorical":
        all_cats = sorted(s_all.unique())
        if len(all_cats) <= 5:
            def get_dist_str(s):
                if s.empty: return "--"
                c = s.value_counts(normalize=True).reindex(all_cats, fill_value=0)
                return " / ".join([f"{v*100:.1f}" for v in c])
            
            res["full"]["dist"] = get_dist_str(s_all)
            res["pre"]["dist"] = get_dist_str(s_pre)
            res["post"]["dist"] = get_dist_str(s_post)
        else:
            res["full"]["dist"] = "Dist. (>5)"
            res["pre"]["dist"] = "--"
            res["post"]["dist"] = "--"
    return res

def build_main_table(all_stats: dict, var_meta: dict, panels: list[dict], harmonized: set[str]) -> str:
    col_spec = "ll l rr rr r"
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
            vtype = vdef.get("type", "continuous").lower()
            tex_name = name.replace("_", r"\_")
            dagger = r"$^\dagger$" if name in harmonized else ""
            
            if vtype == "id":
                pre_cells = "-- & --"
                post_cells = "-- & --"
            elif vtype == "categorical":
                dist_pre = s["pre"].get("dist", "--")
                dist_post = s["post"].get("dist", "--")
                pre_cells = rf"\small {dist_pre} & --"
                post_cells = rf"\small {dist_post} & --"
            else:
                pre_cells = f"{s['pre']['mean']:.2f} & {s['pre']['sd']:.2f}"
                post_cells = f"{s['post']['mean']:.2f} & {s['post']['sd']:.2f}"

            lines.append(
                rf"\texttt{{{tex_name}}}{dagger} & {vdef['label']} & {scale} & "
                rf"{pre_cells} & {post_cells} & {s['full']['n']:,} \\"
            )
        lines.append(r"\addlinespace")

    notes = [
        r"For categorical variables with $\le 5$ categories, the 'Mean' column shows the percentage distribution (e.g., Cat 1 / Cat 2 / ...).",
    ]
    if harmonized:
        notes.append(r"$^\dagger$Manually harmonized from multiple survey versions.")
    
    formatted_notes = "\n".join([rf"\small Note {i+1}: {text} \\\\" for i, text in enumerate(notes)])

    lines += [
        r"\bottomrule", 
        r"\end{tabular}%", 
        r"}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        formatted_notes,
        r"}",
        r"\end{table}"
    ]
    return "\n".join(lines)
def build_appendix_table(all_stats: dict, var_meta: dict, panels: list[dict], years: list[int]) -> str:
    """The original detailed table with yearly N and full descriptives."""
    year_cols = sorted(years)
    col_spec = "llr" + "r" * len(year_cols) + "rrr"
    year_header = " & ".join(str(y) for y in year_cols)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detailed Descriptive Statistics (Appendix)}",
        r"\label{tab:descriptives_appendix}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Variable & Description & Scale & {year_header} & $N$ & Mean & SD \\",
        r"\midrule",
    ]

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta: continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(rf"\multicolumn{{{3 + len(year_cols) + 3}}}{{l}}{{\textit{{{panel_label}}}}} \\")
        for vdef in var_meta[pk]:
            name = vdef["name"]
            s = all_stats.get(name)
            if not s or s["full"]["n"] == 0: continue
            
            n_cells = " & ".join(f"{s['n_by_year'].get(y, 0):,}" for y in year_cols)
            scale = vdef.get("scale", "")
            vtype = vdef.get("type", "continuous").lower()
            tex_name = name.replace("_", r"\_")
            
            if vtype == "id":
                stats_cells = "& -- & --"
            elif vtype == "categorical":
                dist_full = s["full"].get("dist", "--")
                stats_cells = rf"& \small {dist_full} & --"
            else:
                stats_cells = rf"& {s['full']['mean']:.3f} & {s['full']['sd']:.3f}"

            lines.append(rf"\texttt{{{tex_name}}} & {vdef['label']} & {scale} & {n_cells} & {s['full']['n']:,} {stats_cells} \\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule", 
        r"\end{tabular}%", 
        r"}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        r"\small Note 1: For categorical variables with $\le 5$ categories, the 'Mean' column shows the percentage distribution. \\",
        r"}",
        r"\end{table}"
    ]
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
            vtype = vdef.get("type", "continuous").lower()
            if name.lower() not in master.columns: continue
            all_stats[name] = compute_stats(master, name.lower(), pre_years, post_years, vtype)

    harmonized = {h["name"] for h in config.get("harmonize", [])}
    panels = config.get("panels", [])

    main_tex = build_main_table(all_stats, var_meta, panels, harmonized)
    Path("output/tables/descriptives_main.tex").write_text(main_tex)
    
    app_tex = build_appendix_table(all_stats, var_meta, panels, all_study_years)
    Path("output/tables/descriptives_appendix.tex").write_text(app_tex)
    print("Descriptive tables generated.")

if __name__ == "__main__":
    main()
