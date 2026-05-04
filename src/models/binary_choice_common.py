"""Shared constants, data prep, model fitting, and output functions for
the preference stage (RQ1) and realization stage (RQ2) binary choice models.
"""

from itertools import groupby
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.utils import load_master

CONTROLS = [
    "age", "sex", "migback_direct", "migback_indirect",
    "pgisced97", "log_income", "has_children", "sqm_per_head",
    "work_hrs_100", "plb0193_h", "plh0173",
]

# Time-varying only — sex and migback are constant within person so X_{i·} = X_{it},
# making them uninformative in the Mundlak auxiliary regression α_i = X_{i·}π + w_i.
TIME_VARYING = [
    "age", "pgisced97", "log_income", "has_children",
    "sqm_per_head", "work_hrs_100", "plb0193_h", "plh0173",
]

LABELS: dict[str, str] = {
    "Intercept":           "Constant",
    "age":                 "Age",
    "sex":                 "Female",
    "migback_direct":      "Direct Migr. Background",
    "migback_indirect":    "Indirect Migr. Background",
    "pgisced97":           "Education (ISCED-97)",
    "log_income":          r"HH Post-Gov. Income$^\dagger$",
    "has_children":        "Has Children in HH",
    "sqm_per_head":        "Living Space (sqm/person)",
    "work_hrs_100":        "Annual Work Hours (per 100h)",
    "plb0193_h":           "Works Overtime",
    "plh0173":             "Work Satisfaction",
    # Mundlak means
    "age_mean":            r"$\bar{\text{Age}}$",
    "pgisced97_mean":      r"$\bar{\text{Education}}$",
    "log_income_mean":     r"$\overline{\text{HH Income}}^\dagger$",
    "has_children_mean":   r"$\bar{\text{Has Children}}$",
    "sqm_per_head_mean":   r"$\overline{\text{Living Space}}$",
    "work_hrs_100_mean":   r"$\overline{\text{Work Hours}}$",
    "plb0193_h_mean":      r"$\bar{\text{Overtime}}$",
    "plh0173_mean":        r"$\overline{\text{Work Satisfaction}}$",
}

GROUPS = [
    ("Demographic Characteristics", ["age", "sex", "migback_direct", "migback_indirect"]),
    ("Socioeconomic Determinants",  ["pgisced97", "log_income", "has_children", "sqm_per_head"]),
    ("Labour Determinants",         ["work_hrs_100", "plb0193_h", "plh0173"]),
]

PLOT_LABELS = {
    "age":              "Age",
    "sex":              "Female",
    "migback_direct":   "Direct Migr. Background",
    "migback_indirect": "Indirect Migr. Background",
    "pgisced97":        "Education (ISCED-97)",
    "log_income":       "HH Post-Gov. Income (log)",
    "has_children":     "Has Children in HH",
    "sqm_per_head":     "Living Space (sqm/person)",
    "work_hrs_100":     "Annual Work Hours (per 100h)",
    "plb0193_h":        "Works Overtime",
    "plh0173":          "Work Satisfaction",
}


def prep_raw() -> pd.DataFrame:
    """Load master and compute all derived variables. No sample filter applied."""
    df = load_master().copy()
    df["work_hrs_100"]     = df["e11101"] / 100
    df["has_children"]     = (df["d11107"] > 0).where(df["d11107"].notna()).astype("float")
    df["log_income"]       = np.log(df["i11102"].clip(lower=1))
    df["migback_direct"]   = (df["migback"] == 2).where(df["migback"].notna()).astype("float")
    df["migback_indirect"] = (df["migback"] == 3).where(df["migback"].notna()).astype("float")
    df["sector"]           = df["sector"].astype("float")
    return df


def augment(data: pd.DataFrame) -> pd.DataFrame:
    """Append X_{i·} = (1/T)Σ_t X_{it} per individual (Mundlak 1978, eq. 2.4)."""
    means = (
        data.groupby("pid")[TIME_VARYING]
        .transform("mean")
        .rename(columns={v: f"{v}_mean" for v in TIME_VARYING})  # pyright: ignore
    )
    return pd.concat([data, means], axis=1)


def fit_pooled(estimator, formula: str, data: pd.DataFrame):
    return estimator(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pid"]},
        disp=False,
    )


def fit_cre(formula: str, data: pd.DataFrame):
    """Pooled probit on (X_{it}, X_{i·}) with cluster-robust SEs by pid."""
    return smf.probit(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pid"]},
        maxiter=200,
        disp=False,
    )


def _stars(p: float) -> str:
    if p < 0.01: return "{***}"
    if p < 0.05: return "{**}"
    if p < 0.10: return "{*}"
    return ""


def _fmt_coef(c: float, p: float) -> str:
    stars = _stars(p)
    return f"${c:.4f}^{stars}$" if stars else f"${c:.4f}$"


def get_ame(res) -> pd.DataFrame:
    frame = res.get_margeff(at="overall").summary_frame()
    return frame[frame.index.isin(CONTROLS)].reindex([c for c in CONTROLS if c in frame.index])


def var_rows(var: str, ame_frames: list, label: str) -> list[str]:
    coef_cells, se_cells = [], []
    for ame in ame_frames:
        if var in ame.index:
            coef_cells.append(_fmt_coef(ame.loc[var, "dy/dx"], ame.loc[var, "Pr(>|z|)"]))
            se_cells.append(f"$({ame.loc[var, 'Std. Err.']:.4f})$")
        else:
            coef_cells.append("---")
            se_cells.append("")
    return [
        f"    {label} & {' & '.join(coef_cells)} \\\\",
        f"    \\multicolumn{{1}}{{l}}{{}} & {' & '.join(se_cells)} \\\\",
    ]


def _mundlak_wald_p(res) -> str:
    """Joint Wald test H_0: π = 0. Rejection means RE assumption fails."""
    mean_vars = [v for v in res.params.index if v.endswith("_mean")]
    if not mean_vars:
        return "---"
    idx = [list(res.params.index).index(v) for v in mean_vars]
    r = np.zeros((len(idx), len(res.params)))
    for row_i, col_i in enumerate(idx):
        r[row_i, col_i] = 1.0
    p = float(res.wald_test(r, use_f=False, scalar=False).pvalue)
    return f"${p:.3f}$"


def save_latex_table(models: list, path: str, caption: str, label: str) -> None:
    n_models  = len(models)
    col_spec  = "@{}l" + "c" * n_models + "@{}"
    col_types = [m[1] for m in models]
    col_nums  = [m[0] for m in models]
    fe_keys   = list(models[0][3].keys())

    type_row_parts, rule_row_parts, current_col = [], [], 2
    for t, grp in groupby(col_types):
        count = len(list(grp))
        type_row_parts.append(rf"\multicolumn{{{count}}}{{c}}{{\textbf{{{t}}}}}")
        rule_row_parts.append(rf"\cmidrule(lr){{{current_col}-{current_col + count - 1}}}")
        current_col += count

    rows = [
        r"\begin{landscape}",
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\footnotesize",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular*}}{{\linewidth}}{{ @{{\extracolsep{{\fill}}}} {col_spec} @{{}} }}",
        r"\toprule",
        "  & " + " & ".join(type_row_parts) + r" \\",
        "  " + " ".join(rule_row_parts),
        "  & " + " & ".join(col_nums) + r" \\",
        r"\midrule",
    ]

    ame_frames = [get_ame(m[2]) for m in models]

    for i, (group_label, vars_in_group) in enumerate(GROUPS):
        present = [v for v in vars_in_group if any(v in ame.index for ame in ame_frames)]
        if not present:
            continue
        
        # Add bold subheader for the group
        rows.append(rf"  \multicolumn{{{1 + n_models}}}{{@{{}}l}}{{\textbf{{{group_label}}}}} \\")
        
        for var in present:
            rows += var_rows(var, ame_frames, LABELS.get(var, var))
        rows.append(r"  \addlinespace[2pt]")

    rows.append(r"  \midrule")
    for fe_key in fe_keys:
        rows.append(f"  {fe_key} & {' & '.join(m[3][fe_key] for m in models)} \\\\")

    rows.append(r"  \midrule")
    n_obs = int(models[0][2].nobs)
    rows += [
        r"  $N$ (all models) & " + f"{n_obs:,}" + " & " * (n_models - 1) + r" \\",
        r"  McFadden $R^2$ & "       + " & ".join(f"{m[2].prsquared:.4f}" for m in models) + r" \\",
        r"  Log-Likelihood & "       + " & ".join(f"{m[2].llf:.1f}"       for m in models) + r" \\",
        r"  Mundlak means ($p$) & "  + " & ".join(_mundlak_wald_p(m[2])   for m in models) + r" \\",
        r"  \bottomrule",
        r"  \vspace{1.5pt}",
        r"\end{tabular*}",
        r"\smallskip",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} Average marginal effects (AME) reported. Cluster-robust delta-method standard errors "
        r"in parentheses, clustered at the individual level. *** $p<0.01$, ** $p<0.05$, * $p<0.1$. "
        r"$^\dagger$ Variable entered as natural logarithm. "
        r"CRE Probit includes individual time-means of time-varying regressors (Mundlak correction). "
        r"Sector fixed effects: NACE sectors 1--15 (reference: Agriculture).",
        r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ]

    Path(path).write_text("\n".join(rows) + "\n")
    print(f"Saved LaTeX table → {path}")


def plot_ame_heatmap(models: list, path: str) -> None:
    base = [(num, label, res) for num, label, res, fe in models if fe["Sector FE"] == "No"]
    col_labels = [f"{num} {label}" for num, label, _ in base]

    ame_cols, p_cols = {}, {}
    for col, (_, _, res) in zip(col_labels, base):
        frame = res.get_margeff(at="overall").summary_frame()
        frame = frame[frame.index.isin(CONTROLS)].reindex(CONTROLS).dropna()
        ame_cols[col] = frame["dy/dx"]
        p_cols[col]   = frame["Pr(>|z|)"]

    ame_df = pd.DataFrame(ame_cols)
    p_df   = pd.DataFrame(p_cols)
    ame_df.index = [PLOT_LABELS.get(str(v), str(v)) for v in ame_df.index]
    p_df.index   = ame_df.index

    def _plot_stars(p: float) -> str:
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    annot = pd.DataFrame(
        [[f"{ame_df.iloc[i, j]:.3f}{_plot_stars(p_df.iloc[i, j])}"
          for j in range(len(ame_df.columns))]
         for i in range(len(ame_df))],
        index=ame_df.index, columns=ame_df.columns,
    )

    vmax = float(ame_df.abs().max().max())

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
        "axes.spines.right":  False,
        "axes.spines.top":    False,
    }):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            ame_df.abs(), annot=annot, fmt="", cmap="Greys",
            vmin=0, vmax=vmax, linewidths=0.5, linecolor="#e0e0e0", ax=ax,
            cbar_kws={"label": "|AME|", "shrink": 0.7, "aspect": 20},
        )
        threshold = vmax * 0.55
        for text, val in zip(ax.texts, ame_df.abs().values.flat):
            text.set_color("white" if val > threshold else "black")
            text.set_fontsize(8.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap → {path}")
