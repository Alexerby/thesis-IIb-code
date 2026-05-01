"""
Generates an APA-style map plot comparing remote work density
between Pre-COVID and Post-COVID periods.
Calculation: 1 / (1+2) - Share of 'Yes' among remote-capable jobs.
"""

import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# SOEP l11101 (Federal State) mapping
SOEP_STATES = {
    1: "Baden-Württemberg",
    2: "Bayern",
    3: "Berlin",
    4: "Brandenburg",
    5: "Bremen",
    6: "Hamburg",
    7: "Hessen",
    8: "Mecklenburg-Vorpommern",
    9: "Niedersachsen",
    10: "Nordrhein-Westfalen",
    11: "Rheinland-Pfalz",
    12: "Saarland",
    13: "Sachsen",
    14: "Sachsen-Anhalt",
    15: "Schleswig-Holstein",
    16: "Thüringen",
}

# Define offsets for small/enclave states to prevent overlap (longitude, latitude)
OFFSETS = {
    "Berlin": (0.7, 0.0),
    "Brandenburg": (-0.3, -0.3),
    "Bremen": (-0.4, 0.1),
    "Hamburg": (0.4, 0.2),
    "Saarland": (0.0, -0.2),
}

SHORT_NAMES = {
    "Baden-Württemberg": "BW",
    "Bayern": "BY",
    "Berlin": "BE",
    "Brandenburg": "BB",
    "Bremen": "HB",
    "Hamburg": "HH",
    "Hessen": "HE",
    "Mecklenburg-Vorpommern": "MV",
    "Niedersachsen": "NI",
    "Nordrhein-Westfalen": "NRW",
    "Rheinland-Pfalz": "RP",
    "Saarland": "SL",
    "Sachsen": "SN",
    "Sachsen-Anhalt": "ST",
    "Schleswig-Holstein": "SH",
    "Thüringen": "TH",
}


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def get_period_stats(df, years, treatment_var, state_var):
    period_df = df[df["syear"].isin(years)].copy()
    capable_df = period_df[period_df[treatment_var].isin([1, 2])]
    stats = (
        capable_df.groupby(state_var)[treatment_var]
        .apply(lambda x: (x == 1).mean() * 100)
        .reset_index()
    )
    stats["state_name"] = stats[state_var].map(SOEP_STATES)
    return stats


def plot_on_ax(ax, merged, treatment_var, title, vmin=40, vmax=90):
    merged.plot(
        column=treatment_var,
        ax=ax,
        legend=False,
        cmap="Greys",
        edgecolor="0.5",
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
    )
    merged.boundary.plot(ax=ax, color="0.3", linewidth=0.5)

    for idx, row in merged.iterrows():
        centroid = row.geometry.centroid
        x, y = centroid.x, centroid.y
        name = row["name"]
        val = row[treatment_var]
        dx, dy = OFFSETS.get(name, (0, 0))
        short_label = SHORT_NAMES.get(name, name)
        text_color = (
            "white" if val > 65 else "black"
        )  # Adjusted threshold for visual clarity
        ax.text(
            x + dx,
            y + dy,
            short_label,
            fontsize=7,
            ha="center",
            va="center",
            color=text_color,
            alpha=0.9,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_axis_off()


def main():
    config = load_config()
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found.")
        return

    df = pd.read_parquet(master_path)
    treatment_var = config["study"]["soep_outcome_var"].lower()
    state_var = "l11101"

    # 1. Prepare Data for both periods
    pre_years = config["study"]["pre_covid_years"]
    post_years = config["study"]["post_covid_years"]

    pre_stats = get_period_stats(df, pre_years, treatment_var, state_var)
    post_stats = get_period_stats(df, post_years, treatment_var, state_var)

    # 2. Load Geometry
    geojson_url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json"
    print(f"Fetching boundaries ...")
    gdf = gpd.read_file(geojson_url)

    # 3. Merge
    pre_merged = gdf.merge(pre_stats, left_on="name", right_on="state_name")
    post_merged = gdf.merge(post_stats, left_on="name", right_on="state_name")

    # 4. APA Style Plotting (Side-by-Side)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    vmin, vmax = 40, 90
    plot_on_ax(
        ax1,
        pre_merged,
        treatment_var,
        f"Regime 1 Average\n({min(pre_years)}–{max(pre_years)})",
        vmin,
        vmax,
    )
    plot_on_ax(
        ax2,
        post_merged,
        treatment_var,
        f"Regime 2 Average\n({min(post_years)}–{max(post_years)})",
        vmin,
        vmax,
    )

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(
        sm, ax=[ax1, ax2], orientation="horizontal", pad=0.1, shrink=0.6
    )
    cbar.set_label(
        "Willingness to Work Remotely (Share of 'Yes' among Remote-Capable Jobs, %)",
        fontsize=10,
    )

    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "map_comparison_pre_post.png"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Comparison map saved to {out_path}")


if __name__ == "__main__":
    main()
