"""
Generates APA-style descriptive plots for remote work possibility (plb0097).
Print-safe: greyscale shading + hatch patterns distinguish categories without color.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.axes
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


def _apply_apa_style() -> None:
    """Configure matplotlib rcParams to match APA publication style."""
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


def _stacked_bar(
    df: pd.DataFrame,
    categories: list[str],
    title: str,
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Draw a stacked bar chart of ``plb0097`` response shares onto an axes.

    Each bar represents one survey year; segments show the percentage share
    of each response category. Labels are printed inside segments that are
    wide enough (≥ 6 pp). ``plb0097`` values outside [1, 2, 3] are excluded
    before computing shares.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``plb0097`` and ``syear`` columns.
    categories : list[str]
        Ordered list of response-category labels to display, e.g.
        ``["Yes", "No", "Not possible"]``. Only these categories are shown.
    title : str
        Chart title displayed above the axes.
    ax : matplotlib.axes.Axes
        Axes object to draw onto.
    """
    valid = df[df["plb0097"].isin([1, 2, 3])].copy()  # type: ignore[union-attr]
    valid["response"] = valid["plb0097"].map(_CATEGORY_MAP)  # type: ignore[arg-type]
    valid = valid[valid["response"].isin(categories)]  # type: ignore[union-attr]

    counts = (
        valid.groupby(["syear", "response"])  # type: ignore[union-attr]
        .size()
        .unstack("response")  # type: ignore[call-overload]
    )
    counts = counts.reindex(  # type: ignore[call-overload]
        categories, axis="columns", fill_value=0
    )
    pct: pd.DataFrame = counts.div(counts.sum(axis=1), axis=0) * 100

    x = list(range(len(pct)))
    bottom = [0.0] * len(pct)

    for cat in categories:
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
        for c in categories
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    ax.set_title(title, pad=12, fontsize=11, fontweight="normal")


def plot_wwfh_all(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate a stacked bar chart for all three ``plb0097`` response categories.

    Displays the yearly share of "Yes", "No", and "Not possible" responses
    side-by-side as a stacked bar chart and saves it as a PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``plb0097`` and ``syear`` columns.
    out_dir : Path
        Directory where the output PNG (``wwfh_all.png``) is written.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    _stacked_bar(
        df,
        categories=["Yes", "No", "Not possible"],
        title="Willingness to Work from Home by Survey Year (All Categories)",
        ax=ax,
    )
    fig.tight_layout()
    out_path = out_dir / "wwfh_all.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_outcome_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate a grouped bar chart comparing actual vs. willingness to WFH by year.

    For each survey year, two adjacent bars show the share of "Yes" [1]
    among valid Yes/No respondents:

    - ``plb0095_v1`` — actually works from home.
    - ``plb0097`` — willing to work from home (value 3 "Not possible" excluded).

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``plb0095_v1``, ``plb0097``, and ``syear``
        columns.
    out_dir : Path
        Directory where the output PNG (``outcome_comparison.png``) is
        written.
    """
    years = sorted(df["syear"].dropna().unique().astype(int))

    def yes_share(series: pd.Series) -> list[float]:
        return [
            (series[df["syear"] == y].isin([1])).sum() /
            max((series[df["syear"] == y].isin([1, 2])).sum(), 1)
            for y in years
        ]

    share_act  = yes_share(df["plb0095_v1"])
    share_will = yes_share(df["plb0097"])

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - width / 2, share_act,  width,
           label="Actually works from home (plb0095_v1)",
           color="#2b2b2b", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, share_will, width,
           label="Willing to work from home (plb0097, Yes/No only)",
           color="#888888", hatch="///", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Survey Year", labelpad=8)
    ax.set_ylabel("Share answering Yes [1] / ([1] + [2])", labelpad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1)

    handles = [
        mpatches.Patch(facecolor="#2b2b2b", edgecolor="grey",
                       label="Actually works from home (plb0095_v1)"),
        mpatches.Patch(facecolor="#888888", hatch="///", edgecolor="grey",
                       label="Willing to work from home (plb0097, Yes/No only)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    ax.set_title(
        "Remote Work: Actual vs. Willingness by Year\n"
        "(share of Yes among Yes/No respondents)",
        pad=12, fontsize=11, fontweight="normal",
    )

    fig.tight_layout()
    out_path = out_dir / "outcome_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    """Load the master dataframe and generate all descriptive plots."""
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found. Run build_dataframe.py first.")
        return

    config = load_config()
    _apply_apa_style()

    df = pd.read_parquet(master_path)
    out_dir = Path(config["output"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_wwfh_all(df, out_dir)
    plot_outcome_comparison(df, out_dir)


if __name__ == "__main__":
    main()
