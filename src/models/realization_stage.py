"""RQ2 — Realization stage: among willing workers, who actually WFH?

Sample: full-time employees who stated a positive WFH preference (plb0097 == 1)
        and have a valid actual-WFH response (plb0095_v1 ∈ {0, 1}),
        with sector non-missing.
Outcome: plb0095_v1 (1 = actually works from home, 0 = does not).

Mundlak time-means are recomputed on this restricted sample, as required by
the methodology (means must reflect the estimation sample, not the full panel).

Outputs:
    output/tables/realization_stage.tex
    output/figures/ame_heatmap_rq2.png
"""

import statsmodels.formula.api as smf

from src.models.binary_choice_common import (
    CONTROLS, TIME_VARYING,
    prep_raw, augment, fit_pooled, fit_cre,
    save_latex_table,
)

# ── Sample ────────────────────────────────────────────────────────────────────
_raw = prep_raw()
_raw = _raw[(_raw["plb0097"] == 1) & _raw["plb0095_v1"].isin([0, 1])]

_cols = ["pid", "syear", "plb0095_v1"] + CONTROLS + ["sector"]
df    = _raw[_cols].dropna()
print(f"RQ2 sample: N = {len(df):,}  ({df['pid'].nunique():,} individuals)")

df_cre = augment(df)

# ── Formulas ──────────────────────────────────────────────────────────────────
_ctrl  = " + ".join(CONTROLS)
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
    path="output/tables/realization_stage.tex",
    caption="Realization of WFH Preferences: Access and Utilization among Willing Workers",
    label="tab:realization_stage",
)

