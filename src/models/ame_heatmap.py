"""Combined AME heatmap: preference stage (RQ1) and realization stage (RQ2).

Fits the No-FE models for both stages and plots a single side-by-side
heatmap saved to output/figures/ame_heatmap.png.

Run: python -m src.models.ame_heatmap
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from src.models.binary_choice_common import (
    CONTROLS, TIME_VARYING, PLOT_LABELS,
    prep_raw, augment, fit_pooled, fit_cre,
)

# ── Build samples ─────────────────────────────────────────────────────────────
_raw = prep_raw()
_ctrl  = " + ".join(CONTROLS)
_means = " + ".join(f"{v}_mean" for v in TIME_VARYING)

_cols_rq1 = ["pid", "syear", "plb0097"]    + CONTROLS + ["sector"]
_cols_rq2 = ["pid", "syear", "plb0095_v1"] + CONTROLS + ["sector"]

df_rq1 = _raw[_raw["plb0097"].isin([0, 1])][_cols_rq1].dropna()
df_rq2 = _raw[(_raw["plb0097"] == 1) & _raw["plb0095_v1"].isin([0, 1])][_cols_rq2].dropna()

df_rq1_cre = augment(df_rq1)
df_rq2_cre = augment(df_rq2)

print(f"RQ1: N={len(df_rq1):,}   RQ2: N={len(df_rq2):,}")

# ── Fit No-FE models only (for heatmap) ──────────────────────────────────────
models_rq1 = [
    ("Logit",      fit_pooled(smf.logit,  f"plb0097 ~ {_ctrl}",          df_rq1)),
    ("Probit",     fit_pooled(smf.probit, f"plb0097 ~ {_ctrl}",          df_rq1)),
    ("CRE Probit", fit_cre(              f"plb0097 ~ {_ctrl} + {_means}", df_rq1_cre)),
]
models_rq2 = [
    ("Logit",      fit_pooled(smf.logit,  f"plb0095_v1 ~ {_ctrl}",          df_rq2)),
    ("Probit",     fit_pooled(smf.probit, f"plb0095_v1 ~ {_ctrl}",          df_rq2)),
    ("CRE Probit", fit_cre(              f"plb0095_v1 ~ {_ctrl} + {_means}", df_rq2_cre)),
]

# ── Collect AMEs ──────────────────────────────────────────────────────────────
def _ames(models):
    ame_cols, p_cols = {}, {}
    for label, res in models:
        frame = res.get_margeff(at="overall").summary_frame()
        frame = frame[frame.index.isin(CONTROLS)].reindex(CONTROLS).dropna()
        ame_cols[label] = frame["dy/dx"]
        p_cols[label]   = frame["Pr(>|z|)"]
    return pd.DataFrame(ame_cols), pd.DataFrame(p_cols)

ame_rq1, p_rq1 = _ames(models_rq1)
ame_rq2, p_rq2 = _ames(models_rq2)

row_labels = [PLOT_LABELS.get(str(v), str(v)) for v in ame_rq1.index]
ame_rq1.index = row_labels
ame_rq2.index = row_labels
p_rq1.index   = row_labels
p_rq2.index   = row_labels

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def _annot(ame, p):
    return pd.DataFrame(
        [[f"{ame.iloc[i,j]:.3f}{_stars(p.iloc[i,j])}" for j in range(len(ame.columns))]
         for i in range(len(ame))],
        index=ame.index, columns=ame.columns,
    )

annot_rq1 = _annot(ame_rq1, p_rq1)
annot_rq2 = _annot(ame_rq2, p_rq2)

# ── Plot ──────────────────────────────────────────────────────────────────────
vmax = max(ame_rq1.abs().max().max(), ame_rq2.abs().max().max())

rc = {
    "font.family": "serif", "font.size": 9,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.spines.right": False, "axes.spines.top": False,
}

with plt.rc_context(rc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    heatmap_kws = dict(
        fmt="", cmap="Greys", vmin=0, vmax=vmax,
        linewidths=0.5, linecolor="#e0e0e0",
        cbar=False,
    )

    sns.heatmap(ame_rq1.abs(), annot=annot_rq1, ax=ax1, **heatmap_kws)
    sns.heatmap(ame_rq2.abs(), annot=annot_rq2, ax=ax2, **heatmap_kws)

    for ax, title in [(ax1, "Preference Stage (RQ1)"), (ax2, "Realization Stage (RQ2)")]:
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    threshold = vmax * 0.55
    for ax, ame in [(ax1, ame_rq1), (ax2, ame_rq2)]:
        for text, val in zip(ax.texts, ame.abs().values.flat):
            text.set_color("white" if val > threshold else "black")
            text.set_fontsize(8.5)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=[ax1, ax2], label="|AME|", shrink=0.7, aspect=20)

fig.tight_layout()
out = Path("output/figures/ame_heatmap.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
