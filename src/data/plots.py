"""Descriptive plot: actual vs. willingness to WFH by survey year."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from src.data.io import load_config

# plb0097 recoded to {0=No, 1=Yes, 3=Not possible}; plb0095_v1 to {0=No, 1=Yes}.


def _apply_apa_style() -> None:
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


def plot_outcome_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: share actually WFH vs. willing to WFH, by year."""
    years = sorted(df["syear"].dropna().unique().astype(int))

    def yes_share(col: pd.Series, valid_vals: list) -> list[float]:
        return [
            float((col[df["syear"] == y] == 1).sum()) /
            max(float(col[df["syear"] == y].isin(valid_vals).sum()), 1)
            for y in years
        ]

    # plb0095_v1: {0,1}; plb0097: exclude 3 ("not possible") from denominator
    share_act  = yes_share(df["plb0095_v1"], [0, 1])
    share_will = yes_share(df["plb0097"],    [0, 1])

    x     = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, share_act,  width, label="Actually WFH",
           color="#2b2b2b", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, share_will, width, label="Willing to WFH",
           color="#888888", hatch="///", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Survey Year", labelpad=8)
    ax.set_ylabel("Share answering Yes", labelpad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1)

    handles = [
        mpatches.Patch(facecolor="#2b2b2b", edgecolor="grey",
                       label=r"Actually WFH (\texttt{plb0095\_v1})"),
        mpatches.Patch(facecolor="#888888", hatch="///", edgecolor="grey",
                       label=r"Willing to WFH (\texttt{plb0097}, Yes/No only)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Remote Work: Actual vs. Willingness by Year", pad=12, fontsize=11)

    fig.tight_layout()
    out_path = out_dir / "wfh_gap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found. Run build_dataframe.py first.")
        return

    config = load_config()
    _apply_apa_style()
    df = pd.read_parquet(master_path)
    out_dir = Path(config["output"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_outcome_comparison(df, out_dir)


if __name__ == "__main__":
    main()
