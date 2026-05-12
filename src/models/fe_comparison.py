"""FE Linear Probability Model — within-estimator robustness for time-varying regressors.

Individual fixed effects absorb all time-constant regressors (sex, migration
background), so only TIME_VARYING controls are identified. The Mundlak (1978)
result implies β_CRE ≈ β_FE for time-varying regressors; agreement here
motivates the CRE probit as the main specification, since CRE jointly identifies
time-varying and time-constant regressors.

One column per stage, combined into a single appendix table.
Standard errors are clustered at the individual level.

Outputs:
    output/tables/fe_comparison.tex
"""

from pathlib import Path

import pandas as pd

from src.models.binary_choice_common import TIME_VARYING, LABELS, prep_raw, fit_fe

# Groups shown in the table (time-constant variables are excluded — absorbed by FE)
_FE_GROUPS = [
    ("Demographic Characteristics", ["age", "has_partner"]),
    ("Socioeconomic Determinants",   ["pgisced97", "log_income", "has_children", "sqm_per_head"]),
    ("Labour Determinants",          ["work_hrs_100", "plb0193_h", "plh0173"]),
]


def _stars(p: float) -> str:
    if p < 0.01: return "{***}"
    if p < 0.05: return "{**}"
    if p < 0.10: return "{*}"
    return ""


def _var_rows(var: str, results: list, label: str) -> list[str]:
    coef_cells, se_cells = [], []
    for res in results:
        if var in res.params.index:
            c  = float(res.params[var])
            p  = float(res.pvalues[var])
            se = float(res.bse[var])
            stars = _stars(p)
            coef_cells.append(f"${c:.4f}^{stars}$" if stars else f"${c:.4f}$")
            se_cells.append(f"$({se:.4f})$")
        else:
            coef_cells.append("---")
            se_cells.append("")
    return [
        f"    {label} & {' & '.join(coef_cells)} \\\\",
        f"    \\multicolumn{{1}}{{l}}{{}} & {' & '.join(se_cells)} \\\\",
    ]


def _save_table(models: list, path: str, caption: str, label: str) -> None:
    """models: list of (col_num, col_label, result)"""
    n        = len(models)
    col_spec = "@{}l" + "c" * n + "@{}"
    results  = [m[2] for m in models]

    rows = [
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\small",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular*}}{{\linewidth}}{{ @{{\extracolsep{{\fill}}}} {col_spec} @{{}} }}",
        r"\toprule",
        "  & " + " & ".join(rf"\textbf{{{m[1]}}}" for m in models) + r" \\",
        "  & " + " & ".join(m[0] for m in models) + r" \\",
        r"\midrule",
    ]

    for group_label, vars_in_group in _FE_GROUPS:
        present = [v for v in vars_in_group if any(v in r.params.index for r in results)]
        if not present:
            continue
        rows.append(rf"  \multicolumn{{{1 + n}}}{{@{{}}l}}{{\textbf{{{group_label}}}}} \\")
        for var in present:
            rows += _var_rows(var, results, LABELS.get(var, var))
        rows.append(r"  \addlinespace[2pt]")

    rows += [
        r"  \midrule",
        r"  Individual FE & " + " & ".join("Yes" for _ in models) + r" \\",
        r"  Sector FE     & " + " & ".join("No"  for _ in models) + r" \\",
        r"  \midrule",
        r"  $N$ & " + " & ".join(f"{int(m[2].nobs):,}" for m in models) + r" \\",
        r"  $R^2$ (within) & " + " & ".join(f"{m[2].rsquared:.4f}" for m in models) + r" \\",
        r"  \bottomrule",
        r"\end{tabular*}",
        r"\smallskip",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Note:} OLS coefficients from within-transformed regression (FE linear probability "
        r"model). Individual fixed effects absorb all time-constant regressors; only time-varying "
        r"controls are identified. Coefficients are directly interpretable as average probability "
        r"changes. Cluster-robust standard errors in parentheses, clustered at the individual level. "
        r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$. $^\dagger$ Variable entered as natural logarithm.",
        r"\end{minipage}",
        r"\end{table}",
    ]

    Path(path).write_text("\n".join(rows) + "\n")
    print(f"Saved LaTeX table → {path}")


# ── RQ1: Preference stage ─────────────────────────────────────────────────────
_raw_rq1 = prep_raw()
_raw_rq1 = _raw_rq1.loc[_raw_rq1["plb0097"].isin([0, 1])].copy()
_cols_rq1 = ["pid", "syear", "plb0097"] + TIME_VARYING
_df_rq1   = _raw_rq1[_cols_rq1].dropna()
print(f"FE RQ1 sample: N = {len(_df_rq1):,}  ({_df_rq1['pid'].nunique():,} individuals)")

_res_rq1 = fit_fe(_df_rq1, "plb0097")

# ── RQ2: Realization stage ────────────────────────────────────────────────────
_raw_rq2 = prep_raw()
_mask_rq2 = _raw_rq2["plb0097"].isin([1]) & _raw_rq2["plb0095_v1"].isin([0, 1])
_raw_rq2  = _raw_rq2.loc[_mask_rq2].copy()
_cols_rq2 = ["pid", "syear", "plb0095_v1"] + TIME_VARYING
_df_rq2   = _raw_rq2[_cols_rq2].dropna()
print(f"FE RQ2 sample: N = {len(_df_rq2):,}  ({_df_rq2['pid'].nunique():,} individuals)")

_res_rq2 = fit_fe(_df_rq2, "plb0095_v1")

# ── Output ────────────────────────────────────────────────────────────────────
_save_table(
    [
        ("(1)", "Preference (RQ1)", _res_rq1),
        ("(2)", "Realization (RQ2)", _res_rq2),
    ],
    path="output/tables/fe_comparison.tex",
    caption=(
        "Fixed Effects Linear Probability Model: Time-Varying Regressors Only "
        r"(Robustness Check for CRE Probit)"
    ),
    label="tab:fe_comparison",
)
