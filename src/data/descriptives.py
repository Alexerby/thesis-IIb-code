"""
Generate LaTeX descriptive statistics tables for study variables.
Single study period: 2009-2014.
Two outcome variables: plb0097 (willingness to WFH) and plb0095_v1 (actually WFH).
"""

from pathlib import Path
import pandas as pd

from src.data.io import load_config
from src.data.utils import clean_series, load_master, study_period_label



def compute_stats(df: pd.DataFrame, varname: str, vtype: str) -> dict:
    """
    Compute descriptive statistics for a single variable across the full sample.

    For categorical variables with ≤ 5 unique values, a percentage
    distribution string (e.g. ``"45 / 35 / 20"``) is added under the
    ``"dist"`` key.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe containing ``varname`` and ``syear`` columns.
    varname : str
        Column name of the variable to summarise.
    vtype : str
        Variable type; ``"categorical"`` triggers distribution output,
        anything else is treated as continuous.

    Returns
    -------
    dict
        Dictionary with keys ``n``, ``mean``, ``sd``, ``min``, ``max``,
        ``n_by_year`` (dict mapping year → count), and optionally ``dist``
        for categorical variables.
    """
    s_all = clean_series(df[varname])  # type: ignore[arg-type]

    by_year = {}
    for yr, grp in df.groupby("syear"):
        s_yr = clean_series(grp[varname])  # type: ignore[arg-type]
        if not s_yr.empty:
            by_year[int(yr)] = int(s_yr.count())  # type: ignore[arg-type]

    res: dict = {
        "n": int(s_all.count()),
        "mean": s_all.mean(),
        "sd": s_all.std(),
        "min": s_all.min(),
        "max": s_all.max(),
        "n_by_year": by_year,
    }

    if vtype == "categorical":
        all_cats = sorted(s_all.dropna().unique())
        if len(all_cats) <= 5:
            c = s_all.value_counts(normalize=True).reindex(all_cats, fill_value=0)
            res["dist"] = " / ".join(f"{int(cat)}:{v * 100:.0f}\\%" for cat, v in zip(all_cats, c))
        else:
            res["dist"] = "--"

    return res


def build_main_table(
    stats_rq1: dict,
    stats_rq2: dict,
    var_meta: dict,
    panels: list[dict],
    harmonized: set[str],
    period_label: str,
) -> str:
    """
    Build a high-density, landscape-oriented descriptive statistics table.
    """
    n_rq1 = max((s["n"] for s in stats_rq1.values() if s["n"] > 0), default=0)
    n_rq2 = max((s["n"] for s in stats_rq2.values() if s["n"] > 0), default=0)

    col_spec = r"@{}l p{4.5cm} p{4.5cm} rrrr"
    lines = [
        r"\begin{landscape}",
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\centering",
        rf"\caption{{Descriptive Statistics: Preference and Realization Samples ({period_label})}}",
        r"\label{tab:descriptives_main}",
        r"\scriptsize",
        rf"\begin{{tabular*}}{{\linewidth}}{{ @{{\extracolsep{{\fill}}}} {col_spec} @{{}} }}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Preference Stage} & \multicolumn{2}{c}{Realization Stage} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Variable & Label & Definition/Scale & Mean & SD & Mean & SD \\",
        r"\midrule",
    ]

    def _cells(s1, s2, vtype):
        def _pair(s, vt):
            if s is None or s["n"] == 0:
                return "-- & --"
            if vt == "id":
                return "-- & --"
            if vt == "categorical":
                return rf"{s.get('dist', '--')} & --"
            return f"{s['mean']:.2f} & {s['sd']:.2f}"
        return f"{_pair(s1, vtype)} & {_pair(s2, vtype)}"

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta:
            continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{panel_label}}}}} \\")

        for vdef in var_meta[pk]:
            name = vdef["name"]
            s1 = stats_rq1.get(name)
            s2 = stats_rq2.get(name)
            if (not s1 or s1["n"] == 0) and (not s2 or s2["n"] == 0):
                continue

            vtype = vdef.get("type", "continuous").lower()
            safe_name = name.replace('_', r'\_')
            tex_name = rf"\texttt{{{safe_name}}}"
            label = vdef['label']
            scale = vdef.get("scale", "--").replace('_', r'\_')
            
            dagger = r"$^\dagger$" if name in harmonized else ""
            note_text = vdef.get("note", "")
            note_marker = "*" if note_text else ""

            lines.append(
                rf"{tex_name} & {label} & {scale}{dagger}{note_marker} & {_cells(s1, s2, vtype)} \\"
            )
        lines.append(r"\addlinespace")

    lines += [
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{$N$ (Observations)}} & \multicolumn{{2}}{{r}}{{{n_rq1:,}}} & \multicolumn{{2}}{{r}}{{{n_rq2:,}}} \\",
        r"\bottomrule",
        r"\vspace{1.5pt}",
        r"\end{tabular*}",
        r"\smallskip",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"Note: For categorical variables, the `Mean' column shows the percentage distribution across categories.\par",
    ]

    if harmonized:
        lines.append(r"$^\dagger$ Manually harmonized from multiple survey versions.\par")
    
    manual_notes = []
    for p in var_meta.values():
        for v in p:
            n = v.get("note", "")
            if n and n not in manual_notes:
                manual_notes.append(n)
    
    for n in manual_notes:
        safe_n = n.replace('_', r'\_')
        lines.append(rf"* {safe_n}\par")

    lines += [
        r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ]
    return "\n".join(lines)


def build_appendix_table(
    all_stats: dict, var_meta: dict, panels: list[dict], years: list[int]
) -> str:
    """
    Build the appendix descriptive statistics LaTeX table with per-year N columns.
    """
    year_cols = sorted(years)
    max_n_by_year = {}
    for y in year_cols:
        vals = [s["n_by_year"].get(y, 0) for s in all_stats.values()]
        max_n_by_year[y] = max(vals) if vals else 0
    max_n_total = max((s["n"] for s in all_stats.values() if s["n"] > 0), default=0)

    col_spec = r"@{}l p{4.5cm}" + "r" * len(year_cols) + "rr" + r"@{} "
    year_header = " & ".join(str(y) for y in year_cols)

    lines = [
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\centering",
        r"\caption{Detailed Descriptive Statistics (Appendix)}",
        r"\label{tab:descriptives_appendix}",
        r"\resizebox{\linewidth}{!}{%",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Variable & Label & {year_header} & Mean & SD \\",
        r"\midrule",
    ]

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta:
            continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(
            rf"\multicolumn{{{2 + len(year_cols) + 2}}}{{@{{}}l}}{{\textbf{{{panel_label}}}}} \\"
        )
        for vdef in var_meta[pk]:
            name = vdef["name"]
            s = all_stats.get(name)
            if not s or s["n"] == 0:
                continue

            n_cells = " & ".join(f"{s['n_by_year'].get(y, 0):,}" for y in year_cols)
            vtype = vdef.get("type", "continuous").lower()
            tex_name = name.replace("_", r"\_")
            
            label = vdef['label']
            scale = vdef.get("scale", "")
            if vtype == "categorical" and scale and scale != "--":
                safe_scale = scale.replace('_', r'\_')
                label = rf"{label} \tiny ({safe_scale})"

            if vtype == "id":
                stats_cells = "-- & --"
            elif vtype == "categorical":
                stats_cells = rf"\footnotesize {s.get('dist', '--')} & --"
            else:
                stats_cells = rf"{s['mean']:.3f} & {s['sd']:.3f}"

            lines.append(
                rf"\texttt{{{tex_name}}} & {label} & {n_cells} & {stats_cells} \\"
            )
        lines.append(r"\addlinespace")

    n_year_cells = " & ".join(f"{max_n_by_year[y]:,}" for y in year_cols)
    lines += [
        r"\midrule",
        rf"$N$ & & {n_year_cells} & \multicolumn{{2}}{{r}}{{{max_n_total:,}}} \\",
        r"\bottomrule",
        r"\vspace{1.5pt}",
        r"\end{tabular}%",
        r"}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        r"\footnotesize Note: For categorical variables, the `Mean' column shows the percentage distribution.",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _fmt_codes(codes: list[int]) -> str:
    """
    Format a list of integer codes into a compact range string.

    Consecutive integers are collapsed into ``start--end`` ranges.

    Parameters
    ----------
    codes : list[int]
        Unsorted list of integer NACE codes.

    Returns
    -------
    str
        Comma-separated range string, e.g. ``"1--5, 7, 10--12"``.
        Returns an empty string if ``codes`` is empty.
    """
    if not codes:
        return ""
    s = sorted(codes)
    ranges, start, end = [], s[0], s[0]
    for c in s[1:]:
        if c == end + 1:
            end = c
        else:
            ranges.append((start, end))
            start = end = c
    ranges.append((start, end))
    return ", ".join(str(a) if a == b else f"{a}--{b}" for a, b in ranges)


def build_nace_sector_table(config: dict) -> str:
    """
    Build a LaTeX table mapping NACE sector IDs to their Rev.1 and Rev.2 codes.

    Parameters
    ----------
    config : dict
        Project config containing a ``"nace_sectors"`` list where each
        entry has ``id``, ``label``, ``nace_rev1`` (list of int), and
        ``nace_rev2`` (list of int).

    Returns
    -------
    str
        Complete LaTeX source for the NACE sector classification table.
    """
    sectors = config.get("nace_sectors", [])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{NACE Industry Sector Classification}",
        r"\label{tab:nace_sectors}",
        r"\small",
        r"\begin{tabular}{rlp{4.5cm}p{4cm}}",
        r"\toprule",
        r"\# & Sector & NACE Rev.1 codes & NACE Rev.2 codes \\",
        r"\midrule",
    ]
    for s in sectors:
        label = s["label"].replace("&", r"\&")
        rev1 = _fmt_codes(s["nace_rev1"])
        rev2 = _fmt_codes(s["nace_rev2"])
        lines.append(rf"{s['id']} & {label} & {rev1} & {rev2} \\")
    lines += [
        r"\bottomrule",
        r"\vspace{1.5pt}",
        r"\end{tabular}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        r"\small Note: Rev.1 and Rev.2 codes are mapped separately before coalescing. "
        r"Rev.1 takes priority where both are observed.",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _add_derived_cols(df):
    """Add model-derived columns in-place (does not copy)."""
    import numpy as np
    df["work_hrs_100"]     = df["e11101"] / 100
    df["has_children"]     = (df["d11107"] > 0).where(df["d11107"].notna()).astype("float")
    df["log_income"]       = np.log(df["i11102"].clip(lower=1))
    df["migback_direct"]   = (df["migback"] == 2).where(df["migback"].notna()).astype("float")
    df["migback_indirect"] = (df["migback"] == 3).where(df["migback"].notna()).astype("float")
    # Partner in HH: values 1 (spouse/registered) and 2 (partner) are certain.
    # 3 (probably spouse) and 4 (probably partner) are also included.
    df["has_partner"]      = df["partner"].isin([1, 2, 3, 4]).astype("float").where(df["partner"].notna())
    return df


_MODEL_COLS_BASE = [
    "pid", "syear",
    "age", "sex", "migback_direct", "migback_indirect", "has_partner",
    "pgisced97", "log_income", "has_children", "sqm_per_head",
    "work_hrs_100", "plb0193_h", "plh0173", "sector",
]


def build_analysis_sample(master: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the exact RQ1 analysis sample (preference stage).

    Filters rows to match the model input (plb0097∈{0,1}, no missing on any
    model variable) but keeps all master columns so that descriptive stats
    can be computed for every config variable.
    """
    df = _add_derived_cols(master.copy())
    model_cols = ["plb0097"] + _MODEL_COLS_BASE
    valid_idx = (
        df[df["plb0097"].isin([0, 1])][model_cols]
        .dropna()
        .index
    )
    return df.loc[valid_idx]


def build_realization_sample(master: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the exact RQ2 analysis sample (realization stage).

    Filters to willing workers (plb0097==1) with a valid actual-WFH response
    (plb0095_v1∈{0,1}) and no missing on any model variable. Keeps all master
    columns so that descriptive stats can be computed for every config variable.
    """
    df = _add_derived_cols(master.copy())
    model_cols = ["plb0095_v1"] + _MODEL_COLS_BASE
    valid_idx = (
        df[(df["plb0097"] == 1) & df["plb0095_v1"].isin([0, 1])][model_cols]
        .dropna()
        .index
    )
    return df.loc[valid_idx]


def main() -> None:
    """Compute descriptive stats for both stage samples and write LaTeX tables."""
    config = load_config()
    period_label = study_period_label(config, latex=True)

    raw = load_master()
    rq1 = build_analysis_sample(raw)
    rq2 = build_realization_sample(raw)
    print(f"RQ1 sample: {len(rq1):,} obs, {rq1['pid'].nunique():,} individuals")
    print(f"RQ2 sample: {len(rq2):,} obs, {rq2['pid'].nunique():,} individuals")

    var_meta = config["variables"]

    def _compute_all(df):
        stats: dict = {}
        for panel in var_meta.values():
            for vdef in panel:
                name = vdef["name"]
                vtype = vdef.get("type", "continuous").lower()
                if name.lower() not in df.columns:
                    continue
                stats[name] = compute_stats(df, name.lower(), vtype)
        return stats

    print("Computing stats ...")
    stats_rq1 = _compute_all(rq1)
    stats_rq2 = _compute_all(rq2)

    harmonized = {h["name"] for h in config.get("harmonize", [])}
    panels = config.get("panels", [])

    Path("output/tables").mkdir(parents=True, exist_ok=True)

    main_tex = build_main_table(stats_rq1, stats_rq2, var_meta, panels, harmonized, period_label)
    Path("output/tables/descriptives_main.tex").write_text(main_tex)

    nace_tex = build_nace_sector_table(config)
    Path("output/tables/nace_sectors.tex").write_text(nace_tex)

    print("Descriptive tables written to output/tables/")


if __name__ == "__main__":
    main()
