"""RQ1 — Preference stage: determinants of willingness to WFH.

Sample: full-time employees who did not report WFH as structurally impossible
        (plb0097 ∈ {0, 1}), with sector non-missing.
Outcome: plb0097 (1 = would accept WFH if offered, 0 = would not).

Outputs:
    output/tables/preference_stage.tex
    output/figures/ame_heatmap_rq1.png
"""

import statsmodels.formula.api as smf

from src.models.binary_choice_common import (
    CONTROLS, TIME_VARYING,
    prep_raw, augment, fit_pooled, fit_cre,
    save_latex_table,
)

# ── Sample ────────────────────────────────────────────────────────────────────
_raw = prep_raw()
_raw = _raw[_raw["plb0097"].isin([0, 1])]

_cols = ["pid", "syear", "plb0097"] + CONTROLS + ["sector"]
df    = _raw[_cols].dropna()
print(f"RQ1 sample: N = {len(df):,}  ({df['pid'].nunique():,} individuals)")

df_cre = augment(df)

# ── Formulas ──────────────────────────────────────────────────────────────────
_ctrl  = " + ".join(CONTROLS)
_means = " + ".join(f"{v}_mean" for v in TIME_VARYING)

_base   = f"plb0097 ~ {_ctrl}"
_base_s = _base + " + C(sector)"
_cre    = f"plb0097 ~ {_ctrl} + {_means}"
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
    path="output/tables/preference_stage.tex",
    caption="Willingness to Work from Home: Pooled and Correlated Random Effects Models",
    label="tab:preference_stage",
)

