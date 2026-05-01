"""
Generate LaTeX descriptive statistics tables for study variables.
Single study period: 2009-2014.
Two outcome variables: plb0097 (willingness to WFH) and plb0095_v1 (actually WFH).
"""

from pathlib import Path
import pandas as pd

from src.data.io import load_config
from src.data.utils import clean_series, study_period_label

MASTER_PATH = Path("output/data/master.parquet")


def load_master() -> pd.DataFrame:
    """
    Load the master analysis dataframe from Parquet.

    Returns
    -------
    pd.DataFrame
        Master dataframe produced by ``build_dataframe.py``.

    Raises
    ------
    FileNotFoundError
        If the master Parquet file does not exist.
    """
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_PATH} not found — run build_dataframe.py first"
        )
    return pd.read_parquet(MASTER_PATH)


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
            res["dist"] = " / ".join(f"{v * 100:.0f}" for v in c)
        else:
            res["dist"] = "--"

    return res


def build_main_table(
    all_stats: dict,
    var_meta: dict,
    panels: list[dict],
    harmonized: set[str],
    period_label: str,
) -> str:
    """
    Build the main descriptive statistics LaTeX table.

    Outputs a ``booktabs``-style table with one row per variable showing
    label, measurement scale, mean, SD, and N. Categorical variables show
    a percentage distribution instead of mean/SD. Harmonized variables are
    marked with a dagger footnote.

    Parameters
    ----------
    all_stats : dict
        Mapping of variable name → stats dict as returned by
        ``compute_stats``.
    var_meta : dict
        Config ``"variables"`` block mapping panel keys to lists of
        variable definition dicts (each having ``name``, ``label``,
        ``scale``, ``type`` fields).
    panels : list[dict]
        Ordered list of panel dicts from config (each having ``key`` and
        ``label``), controlling row grouping and order.
    harmonized : set[str]
        Set of variable names that were manually harmonized; these receive
        a dagger marker.
    period_label : str
        Formatted study period for the table caption (e.g.
        ``"2009--2014"``).

    Returns
    -------
    str
        Complete LaTeX source for the table, ready to ``\\input`` into a
        document.
    """
    col_spec = r"l p{3.5cm} p{2.0cm} rrr"
    lines = [
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\centering",
        rf"\caption{{Descriptive Statistics ({period_label})}}",
        r"\label{tab:descriptives_main}",
        r"\resizebox{\linewidth}{!}{%",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Variable & Label & Scale & Mean & SD & $N$ \\",
        r"\midrule",
    ]

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta:
            continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{panel_label}}}}} \\")

        for vdef in var_meta[pk]:
            name = vdef["name"]
            s = all_stats.get(name)
            if not s or s["n"] == 0:
                continue

            scale = vdef.get("scale", "")
            vtype = vdef.get("type", "continuous").lower()
            tex_name = name.replace("_", r"\_")
            dagger = r"$^\dagger$" if name in harmonized else ""

            if vtype == "id":
                mean_sd = "-- & --"
            elif vtype == "categorical":
                dist = s.get("dist", "--")
                mean_sd = rf"\footnotesize {dist} & --"
            else:
                mean_sd = f"{s['mean']:.2f} & {s['sd']:.2f}"

            lines.append(
                rf"\texttt{{{tex_name}}}{dagger} & {vdef['label']} & {scale} & "
                rf"{mean_sd} & {s['n']:,} \\"
            )
        lines.append(r"\addlinespace")

    notes = [
        r"For categorical variables with $\le 5$ categories, the `Mean' column shows the percentage distribution across categories.",
    ]
    if harmonized:
        notes.append(r"$^\dagger$Manually harmonized from multiple survey versions.")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        "\n".join(rf"\footnotesize Note {i+1}: {t} \\" for i, t in enumerate(notes)),
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_appendix_table(
    all_stats: dict, var_meta: dict, panels: list[dict], years: list[int]
) -> str:
    """
    Build the appendix descriptive statistics LaTeX table with per-year N columns.

    Extends the main table with one column per study year showing the
    observation count, enabling readers to see panel attrition over time.

    Parameters
    ----------
    all_stats : dict
        Mapping of variable name → stats dict as returned by
        ``compute_stats``.
    var_meta : dict
        Config ``"variables"`` block mapping panel keys to variable
        definition lists.
    panels : list[dict]
        Ordered list of panel dicts controlling row grouping and order.
    years : list[int]
        Study years to generate per-year N columns for.

    Returns
    -------
    str
        Complete LaTeX source for the appendix table.
    """
    year_cols = sorted(years)
    col_spec = r"l p{3.5cm} p{2.0cm}" + "r" * len(year_cols) + "rrr"
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
        rf"Variable & Label & Scale & {year_header} & $N$ & Mean & SD \\",
        r"\midrule",
    ]

    for panel in panels:
        pk = panel["key"]
        if pk not in var_meta:
            continue
        panel_label = panel["label"].replace("&", r"\&")
        lines.append(
            rf"\multicolumn{{{3 + len(year_cols) + 3}}}{{l}}{{\textit{{{panel_label}}}}} \\"
        )
        for vdef in var_meta[pk]:
            name = vdef["name"]
            s = all_stats.get(name)
            if not s or s["n"] == 0:
                continue

            n_cells = " & ".join(f"{s['n_by_year'].get(y, 0):,}" for y in year_cols)
            scale = vdef.get("scale", "")
            vtype = vdef.get("type", "continuous").lower()
            tex_name = name.replace("_", r"\_")

            if vtype == "id":
                stats_cells = "& -- & --"
            elif vtype == "categorical":
                stats_cells = rf"& \footnotesize {s.get('dist', '--')} & --"
            else:
                stats_cells = rf"& {s['mean']:.3f} & {s['sd']:.3f}"

            lines.append(
                rf"\texttt{{{tex_name}}} & {vdef['label']} & {scale} & {n_cells} & {s['n']:,} {stats_cells} \\"
            )
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        r"\footnotesize Note 1: For categorical variables with $\le 5$ categories, the 'Mean' column shows the percentage distribution. \\",
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
        r"\end{tabular}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        r"\small Note: Rev.1 and Rev.2 codes are mapped separately before coalescing. "
        r"Rev.1 takes priority where both are observed.",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Compute descriptive stats for all config variables and write LaTeX tables."""
    config = load_config()
    study_years = config["study"]["study_years"]
    period_label = study_period_label(config, latex=True)

    master = load_master()
    var_meta = config["variables"]
    all_stats: dict = {}

    print("Computing stats for all variables ...")
    for panel in var_meta.values():
        for vdef in panel:
            name = vdef["name"]
            vtype = vdef.get("type", "continuous").lower()
            if name.lower() not in master.columns:
                print(f"  Warning: {name} not in master, skipping")
                continue
            all_stats[name] = compute_stats(master, name.lower(), vtype)

    harmonized = {h["name"] for h in config.get("harmonize", [])}
    panels = config.get("panels", [])

    Path("output/tables").mkdir(parents=True, exist_ok=True)

    main_tex = build_main_table(all_stats, var_meta, panels, harmonized, period_label)
    Path("output/tables/descriptives_main.tex").write_text(main_tex)

    app_tex = build_appendix_table(all_stats, var_meta, panels, study_years)
    Path("output/tables/descriptives_appendix.tex").write_text(app_tex)

    nace_tex = build_nace_sector_table(config)
    Path("output/tables/nace_sectors.tex").write_text(nace_tex)

    print("Descriptive tables written to output/tables/")


if __name__ == "__main__":
    main()
