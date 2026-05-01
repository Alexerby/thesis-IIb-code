"""
Generates APA-style descriptive plots for remote work possibility (plb0097).
Covers all three outcome categories: Yes / No / Not possible in my line of work.
Print-safe: greyscale shading + hatch patterns distinguish categories without color.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from src.data.io import load_config


_CATEGORY_MAP = {1: "Yes", 2: "No", 3: "Not possible"}

_STYLE = {
    "Yes":          {"color": "#2b2b2b", "hatch": ""},
    "No":           {"color": "#888888", "hatch": "///"},
    "Not possible": {"color": "#d4d4d4", "hatch": ""},
}


def _apply_apa_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.dpi":        300,
        "hatch.linewidth":   0.6,
    })


def plot_wwfh_stacked(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["plb0097"].isin([1, 2, 3])].copy()
    valid["response"] = valid["plb0097"].map(_CATEGORY_MAP)  # type: ignore[arg-type]

    counts = (
        valid.groupby(["syear", "response"])
        .size()
        .unstack("response")  # type: ignore[call-overload]
    )
    counts = counts.reindex(  # type: ignore[call-overload]
        ["Yes", "No", "Not possible"], axis="columns", fill_value=0
    )
    pct: pd.DataFrame = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 5))

    x = list(range(len(pct)))
    bottom = [0.0] * len(pct)

    for cat in ["Yes", "No", "Not possible"]:
        vals = pct[cat].tolist()
        style = _STYLE[cat]
        bars = ax.bar(
            x, vals, bottom=bottom,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor="white",
            linewidth=0.5,
            label=cat,
            width=0.72,
        )
        # Percentage labels inside bars
        for bar, val, bot in zip(bars, vals, bottom):
            if val >= 6:
                mid = bot + val / 2
                text_color = "white" if style["color"] == "#2b2b2b" else "black"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color,
                )
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in pct.index], rotation=45, ha="right")
    ax.set_xlabel("Survey Year", labelpad=8)
    ax.set_ylabel("Share (%)", labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_ylim(0, 106)

    handles = [
        mpatches.Patch(
            facecolor=_STYLE[c]["color"],
            hatch=_STYLE[c]["hatch"],
            edgecolor="grey",
            label=c,
        )
        for c in ["Yes", "No", "Not possible"]
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")

    ax.set_title(
        "Willingness to Work from Home by Survey Year",
        pad=12, fontsize=11, fontweight="normal",
    )

    fig.tight_layout()
    out_path = out_dir / "wwfh_stacked.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found. Run build_dataframe.py first.")
        return

    config = load_config()
    _apply_apa_style()

    df = pd.read_parquet(master_path)
    out_dir = Path(config["output"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_wwfh_stacked(df, out_dir)


if __name__ == "__main__":
    main()
