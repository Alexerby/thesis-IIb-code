"""RQ2 robustness — Realization stage: full eligible sample (PTWH unrestricted).

Replicates realization_stage.py on the full eligible sample, including workers
who stated no preference for WFH (plb0097 == 0), with plb0097 entered as a
regressor rather than used as a sample restriction. Serves as a robustness
check: do the RQ2 results change when the sample is not restricted to
preference-positive workers?

Sample: full-time employees with plb0097 ∈ {0,1} and plb0095_v1 ∈ {0,1},
        with sector non-missing. (Same eligibility criterion as RQ1.)
Outcome: plb0095_v1 (1 = actually works from home, 0 = does not).

Mundlak time-means are computed for the same time-varying controls as in the
main specification; plb0097 is included as a level control only.

Outputs:
    output/tables/realization_stage_full_sample.tex
"""

import statsmodels.formula.api as smf

from src.models.binary_choice_common import (
    CONTROLS, TIME_VARYING, LABELS, GROUPS,
    prep_raw, augment, fit_pooled, fit_cre,
    save_latex_table,
)

CONTROLS_FS = ["plb0097"] + CONTROLS

LABELS_FS = {
    **LABELS,
    "plb0097": "WFH Preference (stated)",
}

GROUPS_FS = [
    ("WFH Preference", ["plb0097"]),
    *GROUPS,
]

# ── Sample ────────────────────────────────────────────────────────────────────
_raw = prep_raw()
_mask = _raw["plb0097"].isin([0, 1]) & _raw["plb0095_v1"].isin([0, 1])
_raw  = _raw.loc[_mask].copy()

_cols = ["pid", "syear", "plb0095_v1"] + CONTROLS_FS + ["sector"]
df    = _raw[_cols].dropna()
print(f"RQ2 full-sample robustness: N = {len(df):,}  ({df['pid'].nunique():,} individuals)")

df_cre = augment(df)

# ── Formulas ──────────────────────────────────────────────────────────────────
_ctrl  = " + ".join(CONTROLS_FS)
_means = " + ".join(f"{v}_mean" for v in TIME_VARYING)

_base   = f"plb0095_v1 ~ {_ctrl}"
_base_s = _base + " + C(sector)"
_cre    = f"plb0095_v1 ~ {_ctrl} + {_means}"
_cre_s  = _cre  + " + C(sector)"

# ── Fit ───────────────────────────────────────────────────────────────────────
MODELS: list = [
    ("(1)", "Logit",      fit_pooled(smf.logit,  _base,   df),    {"Sector FE": "No"}),
    ("(2)", "Logit",      fit_pooled(smf.logit,  _base_s, df),    {"Sector FE": "Yes"}),
    ("(3)", "Probit",     fit_pooled(smf.probit, _base,   df),    {"Sector FE": "No"}),
    ("(4)", "Probit",     fit_pooled(smf.probit, _base_s, df),    {"Sector FE": "Yes"}),
    ("(5)", "CRE Probit", fit_cre(_cre,   df_cre),                {"Sector FE": "No"}),
    ("(6)", "CRE Probit", fit_cre(_cre_s, df_cre),                {"Sector FE": "Yes"}),
]

# ── Outputs ───────────────────────────────────────────────────────────────────
save_latex_table(
    MODELS,
    path="output/tables/realization_stage_full_sample.tex",
    caption=(
        r"WFH Realization: Full Eligible Sample Robustness "
        r"(Preference Unrestricted, \texttt{plb0097} Controlled)"
    ),
    label="tab:realization_stage_full_sample",
    groups=GROUPS_FS,
    labels=LABELS_FS,
    controls=CONTROLS_FS,
)
