"""Pooled binary choice models (LOGIT, PROBIT) to answer the
first out of my two research questions."""

from itertools import groupby
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from src.data.utils import load_master

# Eligible respondents: exclude plb0097=3 ("Not possible")
df: pd.DataFrame = load_master()
df = df[df["plb0097"].isin([0, 1])]

# Verify d11107 has no negative SOEP codes before creating the dummy
_d11107_neg = (df["d11107"] < 0).sum()
if _d11107_neg > 0:
    print(f"WARNING: {_d11107_neg} negative values in d11107 — treating as NaN")
    df["d11107"] = df["d11107"].where(df["d11107"] >= 0)
print("d11107 value counts (top 10):", df["d11107"].value_counts().head(10).to_dict())

# Derived variables — include source columns in dropna so NaNs propagate correctly
df["migback_direct"]   = (df["migback"] == 2).where(df["migback"].notna()).astype("float")
df["migback_indirect"] = (df["migback"] == 3).where(df["migback"].notna()).astype("float")
df["has_children"]     = (df["d11107"]  >  0).where(df["d11107"].notna()).astype("float")
df["log_income"]       = np.log(df["i11102"].clip(lower=1))
df["work_hrs_100"]     = df["e11101"] / 100

X_vars = [
    "age", "sex", "migback_direct", "migback_indirect",
    "pgisced97", "log_income", "has_children", "sqm_per_head",
    "work_hrs_100", "plb0193_h", "plh0173",
    "sector",
]
df = df[["plb0097", "migback", "d11107", "i11102", "e11101"] + X_vars].dropna().astype(float)
print(f"N = {df.shape[0]:,}")
print(f"has_children distribution: {df['has_children'].value_counts().to_dict()}")

# ── Formulas ──────────────────────────────────────────────────────────────────
_controls = (
    "plb0097 ~ age + sex + migback_direct + migback_indirect"
    " + pgisced97 + log_income + has_children + sqm_per_head"
    " + work_hrs_100 + plb0193_h + plh0173"
)
_sector = _controls + " + C(sector)"

# ── Fit models ────────────────────────────────────────────────────────────────
# Each entry: (number, type, fitted result, fe indicator dict)
MODELS = [
    ("(1)", "Logit",  smf.logit( _controls, data=df).fit(), {"Sector FE": "No"}),
    ("(2)", "Logit",  smf.logit( _sector,   data=df).fit(), {"Sector FE": "Yes"}),
    ("(3)", "Probit", smf.probit(_controls, data=df).fit(), {"Sector FE": "No"}),
    ("(4)", "Probit", smf.probit(_sector,   data=df).fit(), {"Sector FE": "Yes"}),
]

for num, t, res, _ in MODELS:
    print(f"\n{'='*78}\n{num} {t}\n{'='*78}")
    print(res.summary())

# ── LaTeX table (stacked SE format) ───────────────────────────────────────────

LABELS = {
    "Intercept":        "Constant",
    "age":              "Age",
    "sex":              "Female",
    "migback_direct":   "Direct Migr. Background",
    "migback_indirect": "Indirect Migr. Background",
    "pgisced97":        "Education (ISCED-97)",
    "log_income":       r"HH Post-Gov. Income$^\dagger$",
    "has_children":     "Has Children in HH",
    "sqm_per_head":     "Living Space (sqm/person)",
    "work_hrs_100":     "Annual Work Hours (per 100h)",
    "plb0193_h":        "Works Overtime",
    "plh0173":          "Work Satisfaction",
}

GROUPS = [
    ("Demographics",  ["age", "sex", "migback_direct", "migback_indirect"]),
    ("Socioeconomic", ["pgisced97", "log_income", "has_children", "sqm_per_head"]),
    ("Labour",        ["work_hrs_100", "plb0193_h", "plh0173"]),
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
    """Return two LaTeX rows: coefficient row + SE row underneath."""
    coefs = " & ".join(_fmt_coef(m[2].params[var], m[2].pvalues[var]) for m in models)
    ses   = " & ".join(f"$({m[2].bse[var]:.4f})$"                    for m in models)
    return [
        f"    {label} & {coefs} \\\\",
        f"    \\multicolumn{{1}}{{l}}{{}} & {ses} \\\\[2pt]",
    ]


def save_latex_table(models: list, path: str) -> None:
    n_models = len(models)
    n_cols   = n_models + 1  # label column + one per model

    # Use standard tabular; 'c' for models. Remove \resizebox to prevent stretching.
    col_spec = "l" + "c" * n_models

    # Extract info from models
    col_nums  = [m[0] for m in models]
    col_types = [m[1] for m in models]

    # Build type row and cmidrule row
    type_row_parts = []
    rule_row_parts = []
    current_col = 2
    for t, group in groupby(col_types):
        count = len(list(group))
        type_row_parts.append(rf"\multicolumn{{{count}}}{{c}}{{\textbf{{{t}}}}}")
        end_col = current_col + count - 1
        # Trimming (lr) ensures gaps between lines and keeps them from hitting margins
        rule_row_parts.append(rf"\cmidrule(lr){{{current_col}-{end_col}}}")
        current_col += count
    
    type_row = "  & " + " & ".join(type_row_parts) + r" \\"
    rule_row = "  " + " ".join(rule_row_parts)
    num_row  = "  & " + " & ".join(col_nums) + r" \\"

    fe_keys = list(models[0][3].keys())

    rows = [
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\footnotesize",
        r"\centering",
        r"\caption{Willingness to Work from Home: Pooled Logit and Probit}",
        r"\label{tab:pooled_bcm}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        type_row,
        rule_row,
        num_row,
        r"\midrule",
    ]

    for group_label, vars_in_group in GROUPS:
        rows += [
            r"  \addlinespace",
            rf"  \multicolumn{{{n_cols}}}{{l}}{{\textbf{{{group_label}}}}} \\",
        ]
        for var in vars_in_group:
            rows += _var_rows(var, models, LABELS.get(var, var))

    # FE indicators
    rows.append(r"  \midrule")
    for fe_key in fe_keys:
        fe_vals = " & ".join(m[3][fe_key] for m in models)
        rows.append(f"  {fe_key} & {fe_vals} \\\\")

    # Model stats
    rows.append(r"  \midrule")
    n_obs = " & ".join(f"{int(m[2].nobs):,}" for m in models)
    pr2   = " & ".join(f"{m[2].prsquared:.4f}" for m in models)
    ll    = " & ".join(f"{m[2].llf:.1f}"       for m in models)
    rows += [
        rf"  $N$ & {n_obs} \\",
        rf"  McFadden $R^2$ & {pr2} \\",
        rf"  Log-Likelihood & {ll} \\",
        r"  \bottomrule",
        r"\end{tabular}",
        r"\smallskip",
        r"\parbox{\linewidth}{",
        r"\footnotesize Standard errors in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.1$. Constant included but not reported. \\",
        r"\footnotesize $^\dagger$ Variable entered as natural logarithm. \\",
        r"\footnotesize Sector fixed effects: NACE sectors 1--15 (reference: Agriculture).",
        r"}",
        r"\end{table}",
    ]

    Path(path).write_text("\n".join(rows) + "\n")
    print(f"Saved LaTeX table → {path}")


save_latex_table(MODELS, "output/tables/pooled_bcm.tex")
