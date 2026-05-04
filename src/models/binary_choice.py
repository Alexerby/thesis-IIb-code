"""Pooled binary choice models (Logit, Probit) and Correlated Random Effects
Probit (Mundlak correction) for willingness to work from home.

Produces a single combined landscape LaTeX table (output/tables/binary_choice.tex).
"""

from itertools import groupby
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.data.utils import load_master

#  Variable lists 
CONTROLS = [
    "age", "sex", "migback_direct", "migback_indirect",
    "pgisced97", "log_income", "has_children", "sqm_per_head",
    "work_hrs_100", "plb0193_h", "plh0173",
]

# Subset of CONTROLS that vary within individuals — used for Mundlak means
TIME_VARYING = [
    "age", "pgisced97", "log_income", "has_children",
    "sqm_per_head", "work_hrs_100", "plb0193_h", "plh0173",
]

#  Data prep ─
_raw = load_master()
_raw = _raw[_raw["plb0097"].isin([0, 1])].copy()

_raw["work_hrs_100"]     = _raw["e11101"] / 100
_raw["has_children"]     = (_raw["d11107"] > 0).where(_raw["d11107"].notna()).astype("float")
_raw["log_income"]       = np.log(_raw["i11102"].clip(lower=1))
_raw["migback_direct"]   = (_raw["migback"] == 2).where(_raw["migback"].notna()).astype("float")
_raw["migback_indirect"] = (_raw["migback"] == 3).where(_raw["migback"].notna()).astype("float")
_raw["sector"]           = _raw["sector"].astype("float")

# Base sample excludes sector from dropna so its all-NaN state doesn't wipe rows.
# A separate sector sample is used for sector FE models when sector is populated.
_cols      = ["pid", "syear", "plb0097"] + CONTROLS
df         = _raw[_cols].dropna()
df_sector  = _raw[_cols + ["sector"]].dropna()

print(f"Base sample:   N = {len(df):,}  ({df['pid'].nunique():,} individuals)")
print(f"Sector sample: N = {len(df_sector):,}  ({df_sector['pid'].nunique():,} individuals)")

#  Mundlak augmentation 
def _augment(data: pd.DataFrame) -> pd.DataFrame:
    means = (
        data.groupby("pid")[TIME_VARYING]
        .transform("mean")
        .rename(columns={v: f"{v}_mean" for v in TIME_VARYING})  # pyright: ignore
    )
    return pd.concat([data, means], axis=1)

df_cre        = _augment(df)
df_cre_sector = _augment(df_sector)

#  Formulas 
_ctrl  = " + ".join(CONTROLS)
_means = " + ".join(f"{v}_mean" for v in TIME_VARYING)

_base     = f"plb0097 ~ {_ctrl}"
_base_s   = _base   + " + C(sector)"
_cre      = f"plb0097 ~ {_ctrl} + {_means}"
_cre_s    = _cre    + " + C(sector)"

#  Fit models 
def _fit_pooled(estimator, formula, data):
    return estimator(formula, data=data).fit(disp=False)

def _fit_cre(formula, data):
    return smf.probit(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["pid"]},
        maxiter=200,
        disp=False,
    )

MODELS: list = [
    ("(1)", "Logit",      _fit_pooled(smf.logit,  _base,  df),      {"Sector FE": "No"}),
    ("(3)", "Probit",     _fit_pooled(smf.probit, _base,  df),      {"Sector FE": "No"}),
    ("(5)", "CRE Probit", _fit_cre(_cre, df_cre),                   {"Sector FE": "No"}),
]
if len(df_sector) > 0:
    MODELS.insert(1, ("(2)", "Logit",      _fit_pooled(smf.logit,  _base_s, df_sector),      {"Sector FE": "Yes"}))
    MODELS.insert(3, ("(4)", "Probit",     _fit_pooled(smf.probit, _base_s, df_sector),      {"Sector FE": "Yes"}))
    MODELS.append(   ("(6)", "CRE Probit", _fit_cre(_cre_s, df_cre_sector),                  {"Sector FE": "Yes"}))
else:
    print("NOTE: sector not yet populated — sector FE models (2), (4), (6) skipped")

for num, label, res, _ in MODELS:
    print(f"\n{'='*78}\n{num} {label}\n{'='*78}")
    print(res.summary())

#  LaTeX table ─
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
    ("Demographics",   ["age", "sex", "migback_direct", "migback_indirect"]),
    ("Socioeconomic",  ["pgisced97", "log_income", "has_children", "sqm_per_head"]),
    ("Labour",         ["work_hrs_100", "plb0193_h", "plh0173"]),
    ("Mundlak means",  [f"{v}_mean" for v in TIME_VARYING]),
]


def _stars(p: float) -> str:
    if p < 0.01: return "{***}"
    if p < 0.05: return "{**}"
    if p < 0.10: return "{*}"
    return ""


def _fmt_coef(c: float, p: float) -> str:
    stars = _stars(p)
    return f"${c:.4f}^{stars}$" if stars else f"${c:.4f}$"


def _var_rows(var: str, models: list, label: str) -> list[str]:
    coef_cells, se_cells = [], []
    for m in models:
        if var in m[2].params.index:
            coef_cells.append(_fmt_coef(m[2].params[var], m[2].pvalues[var]))
            se_cells.append(f"$({m[2].bse[var]:.4f})$")
        else:
            coef_cells.append("---")
            se_cells.append("")
    return [
        f"    {label} & {' & '.join(coef_cells)} \\\\",
        f"    \\multicolumn{{1}}{{l}}{{}} & {' & '.join(se_cells)} \\\\[2pt]",
    ]


def save_latex_table(models: list, path: str) -> None:
    n_models  = len(models)
    n_cols    = n_models + 1
    col_spec  = "l" + "c" * n_models
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
        r"\renewcommand{\arraystretch}{1.1}",
        r"\footnotesize",
        r"\centering",
        r"\caption{Willingness to Work from Home: Pooled and Correlated Random Effects Models}",
        r"\label{tab:binary_choice}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "  & " + " & ".join(type_row_parts) + r" \\",
        "  " + " ".join(rule_row_parts),
        "  & " + " & ".join(col_nums) + r" \\",
        r"\midrule",
    ]

    for group_label, vars_in_group in GROUPS:
        present = [v for v in vars_in_group if any(
            v in m[2].params.index or v.endswith("_mean") for m in models
        )]
        # For Mundlak means: only include rows where at least one model has the var
        present = [v for v in vars_in_group if any(v in m[2].params.index for m in models)]
        if not present:
            continue
        rows += [
            r"  \addlinespace",
            rf"  \multicolumn{{{n_cols}}}{{l}}{{\textbf{{{group_label}}}}} \\",
        ]
        for var in present:
            rows += _var_rows(var, models, LABELS.get(var, var))

    rows.append(r"  \midrule")
    for fe_key in fe_keys:
        rows.append(f"  {fe_key} & {' & '.join(m[3][fe_key] for m in models)} \\\\")

    rows.append(r"  \midrule")
    rows += [
        r"  $N$ & "              + " & ".join(f"{int(m[2].nobs):,}"  for m in models) + r" \\",
        r"  McFadden $R^2$ & "   + " & ".join(f"{m[2].prsquared:.4f}" for m in models) + r" \\",
        r"  Log-Likelihood & "   + " & ".join(f"{m[2].llf:.1f}"       for m in models) + r" \\",
        r"  \bottomrule",
        r"\end{tabular}",
        r"\smallskip",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"Pooled models: heteroskedasticity-robust standard errors in parentheses. CRE Probit: cluster-robust SEs by individual. *** $p<0.01$, ** $p<0.05$, * $p<0.1$. Constant included but not reported.\par",
        r"$^\dagger$ Variable entered as natural logarithm.\par",
        r"CRE Probit includes individual time-means of time-varying regressors (Mundlak correction).\par",
        r"Sector fixed effects: NACE sectors 1--15 (reference: Agriculture).",
        r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
    ]

    Path(path).write_text("\n".join(rows) + "\n")
    print(f"Saved LaTeX table → {path}")


save_latex_table(MODELS, "output/tables/binary_choice.tex")
