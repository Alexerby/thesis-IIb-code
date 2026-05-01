"""
Generates an APA-style map plot comparing remote work density
between Pre-COVID and Post-COVID periods.
Calculation: 1 / (1+2) - Share of 'Yes' among remote-capable jobs.
"""

import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.axes
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


def load_config() -> dict:
    """
    Load the project configuration from ``config.json``.

    Returns
    -------
    dict
        Parsed JSON configuration located three directories above this
        file (project root).
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def get_period_stats(
    df: pd.DataFrame,
    years: list[int],
    treatment_var: str,
    state_var: str,
) -> pd.DataFrame:
    """
    Compute the share of "Yes" respondents per federal state for given years.

    Only respondents who answered 1 ("Yes") or 2 ("No") are included in
    the denominator — value 3 ("Not possible") is excluded, matching the
    binary willingness measure used in analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with ``syear``, ``treatment_var``, and
        ``state_var`` columns.
    years : list[int]
        Survey years to include in the aggregation.
    treatment_var : str
        Column name of the willingness-to-WFH variable (typically
        ``"plb0097"``).
    state_var : str
        Column name of the federal state identifier (typically
        ``"l11101"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[state_var, treatment_var, "state_name"]``
        where ``treatment_var`` holds the percentage of "Yes" answers among
        remote-capable respondents in each state.
    """
    period_df = df[df["syear"].isin(years)].copy()
    capable_df = period_df[period_df[treatment_var].isin([1, 2])]
    stats = (
        capable_df.groupby(state_var)[treatment_var]
        .apply(lambda x: (x == 1).mean() * 100)
        .reset_index()
    )
    stats["state_name"] = stats[state_var].map(SOEP_STATES)
    return stats


def plot_on_ax(
    ax: matplotlib.axes.Axes,
    merged: gpd.GeoDataFrame,
    treatment_var: str,
    title: str,
    vmin: float = 40,
    vmax: float = 90,
) -> None:
    """
    Draw a greyscale choropleth map of state-level willingness-to-WFH shares.

    State labels use short abbreviations (e.g. "NRW") positioned at each
    state's centroid with configurable offsets for small or enclave states.
    Label colour switches to white for darker-shaded states.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw onto.
    merged : gpd.GeoDataFrame
        GeoDataFrame with geometry and a ``treatment_var`` column
        containing the percentage share per state, plus a ``"name"``
        column matching ``SHORT_NAMES`` keys.
    treatment_var : str
        Column name in ``merged`` that holds the percentage values to map.
    title : str
        Map title displayed above the axes.
    vmin : float, optional
        Minimum value of the colour scale. Defaults to 40.
    vmax : float, optional
        Maximum value of the colour scale. Defaults to 90.
    """
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


def main() -> None:
    """Load master data, fetch state boundaries, and save the willingness map."""
    config = load_config()
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found.")
        return

    df = pd.read_parquet(master_path)
    treatment_var = config["study"]["outcome_willingness"].lower()
    state_var = "l11101"
    study_years = config["study"]["study_years"]

    period_stats = get_period_stats(df, study_years, treatment_var, state_var)

    # 2. Load Geometry
    geojson_url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json"
    print(f"Fetching boundaries ...")
    gdf = gpd.read_file(geojson_url)

    # 3. Merge
    merged = gdf.merge(period_stats, left_on="name", right_on="state_name")

    # 4. APA Style Plotting
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    vmin, vmax = 40, 90
    plot_on_ax(
        ax1,
        merged,
        treatment_var,
        f"Average Willingness to WFH\n({min(study_years)}–{max(study_years)})",
        vmin,
        vmax,
    )

    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(
        "Willingness to Work Remotely (Share of 'Yes' among Remote-Capable Jobs, %)",
        fontsize=10,
    )

    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "map_willingness.png"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Map saved to {out_path}")


if __name__ == "__main__":
    main()
