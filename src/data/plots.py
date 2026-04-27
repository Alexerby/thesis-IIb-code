"""
Generates APA-style trend plots for remote work possibility.
Compares Pre-COVID (2009-2014) vs Post-COVID (2021-2022).
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)

def main():
    config = load_config()
    
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found.")
        return

    df = pd.read_parquet(master_path)
    
    # APA Style Settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    sns.set_style("white") # No grid lines for APA
    
    # --- Yearly Trend of Remote Work Possibility ---
    plt.figure(figsize=(8, 5))
    
    treatment_var = config["variables"]["outcome_primary"][0]["name"].lower()
    
    # Calculate stats
    trend_df = df.copy()
    trend_df = trend_df[trend_df[treatment_var].isin([1, 2, 3])]
    
    # Line 1: Share of total workforce (1 / 1+2+3)
    total_share = trend_df.groupby('syear').apply(lambda x: (x[treatment_var] == 1).mean() * 100)
    
    # Line 2: Share of capable jobs (1 / 1+2)
    capable_df = trend_df[trend_df[treatment_var].isin([1, 2])]
    capable_share = capable_df.groupby('syear').apply(lambda x: (x[treatment_var] == 1).mean() * 100)

    # Helper to plot segments (to avoid connecting 2014 to 2021)
    def plot_segments(data, label, marker, color, linestyle):
        pre = data[data.index <= 2014]
        post = data[data.index >= 2021]
        sns.lineplot(x=pre.index, y=pre.values, marker=marker, label=label, color=color, linewidth=2, linestyle=linestyle)
        sns.lineplot(x=post.index, y=post.values, marker=marker, color=color, linewidth=2, linestyle=linestyle)

    plot_segments(total_share, "Total Workforce", 'o', '#4a4a4a', '-')
    plot_segments(capable_share, "Remote-Capable Jobs Only", 's', '#9a9a9a', '--')

    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Possibility to Work from Home (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.xticks(total_share.index)
    
    plt.legend(title="Sample Definition", frameon=False)
    sns.despine()
    
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path_trend = figures_dir / "remote_possibility_trend.png"
    plt.savefig(out_path_trend, dpi=300, bbox_inches='tight')
    print(f"Saved dual-line trend plot to {out_path_trend}")

if __name__ == "__main__":
    main()
