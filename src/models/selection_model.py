"""
Selection Model: Predicts 'Possibility to work from home' (1/0).
Excludes jobs where remote work is physically impossible (category 3).
Analyzes shifting determinants Pre- vs Post-COVID.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- Configuration & Model Specification ---

# Variables that should be log-transformed
LOG_TRANSFORMS = {
    "i11102": "log_income"
}

# Continuous variables to be centered around their mean
CONTINUOUS_VARS = [
    "log_income", 
    "pgisced97", 
    "d11107", 
    "age", 
    "plh0173"
]

# Base control variables for the regression
BASE_CONTROLS = [
    "age_c",
    "sex",
    "migback",
    "log_income_c",
    "pgisced97_c",
    "d11107_c",
    "plh0173_c",
    "pglfs",
    "e11101",
    "plb0193_h",
]

# Variables to interact with the 'post_covid' dummy
INTERACTION_VARS = [
    "sex",
    "migback",
    "log_income_c",
    "pgisced97_c",
    "d11107_c",
]

# Mapping for pretty-printing and LaTeX output
VAR_LABELS = {
    "post_covid": "Post-COVID (2021-22)",
    "log_income_c": r"Log Household Income$^\dagger$",
    "pgisced97_c": r"Education$^\dagger$",
    "d11107_c": r"Children$^\dagger$",
    "age_c": r"Age$^\dagger$",
    "plh0173_c": r"Satisfaction with work$^\dagger$",
    "sex": "Gender",
    "migback": "Migration Background",
    "pglfs": "Labor Force Status",
    "e11101": "Annual work hours",
    "plb0193_h": "Works overtime",
    "inter_sex": "Gender x Post-COVID",
    "inter_migback": "Mig. Background x Post-COVID",
    "inter_log_income_c": r"Log Income$^\dagger$ x Post-COVID",
    "inter_pgisced97_c": r"Education$^\dagger$ x Post-COVID",
    "inter_d11107_c": r"Children$^\dagger$ x Post-COVID",
}

# Grouping for the final table
TABLE_GROUPS = {
    "Main Trend": ["post_covid"],
    "Socioeconomic Determinants": [
        "age_c",
        "sex",
        "migback",
        "pgisced97_c",
        "d11107_c",
        "log_income_c",
    ],
    "Work \& Health Determinants": [
        "plh0173_c",
        "pglfs",
        "e11101",
        "plb0193_h",
    ],
    "Shifting Determinants (Interactions)": [
        f"inter_{v}" for v in INTERACTION_VARS
    ],
}

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)

def prepare_data(df, outcome_name):
    """Handles filtering, binarization, transformations, centering, and interactions."""
    
    # 1. Filter for remote-capable occupations (1=Yes, 2=No)
    # Excludes 3 (Impossible) and missing values
    df = df[df[outcome_name].isin([1, 2])].copy()
    
    # Binarize outcome (1=Yes, 0=No).
    df["selection_outcome"] = (df[outcome_name] == 1).astype(int)

    # 2. Transformations
    for raw_var, new_var in LOG_TRANSFORMS.items():
        df[new_var] = np.log(df[raw_var] + 1)

    # 3. Time dummy
    df["post_covid"] = (df["syear"] >= 2021).astype(int)

    # 4. Centering continuous predictors
    for var in CONTINUOUS_VARS:
        df[f"{var}_c"] = df[var] - df[var].mean()
        print(f"  Centered {var} (mean: {df[var].mean():.2f})")

    # 5. Interaction Terms
    for var in INTERACTION_VARS:
        df[f"inter_{var}"] = df[var] * df["post_covid"]

    return df

def run_selection_model(df):
    """Defines and fits the LPM model."""
    
    interaction_names = [f"inter_{v}" for v in INTERACTION_VARS]
    reg_vars = ["selection_outcome", "post_covid", "hid"] + BASE_CONTROLS + interaction_names
    
    df_reg = df[reg_vars].dropna().copy()
    print(f"Final sample size for selection model: {len(df_reg):,}")

    y = df_reg["selection_outcome"]
    X = df_reg[["post_covid"] + BASE_CONTROLS + interaction_names]
    X = sm.add_constant(X)

    print("Running Selection Model (LPM) with clustered standard errors...")
    model = sm.OLS(y, X)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": df_reg["hid"]})
    
    return results

def write_latex(results, out_path):
    """Writes results to a grouped LaTeX table using centralized configuration."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Selection into Remote Work (Linear Probability Model)}",
        r"\label{tab:selection_results}",
        r"\resizebox{0.8\textwidth}{!}{%",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Dependent variable:}} \\ \cline{2-3}",
        r" & \multicolumn{2}{c}{\textit{Possibility to work from home (1/0)}} \\",
        r" \textbf{Predictor} & \textbf{Coeff. (Beta)} & \textbf{(Std. Err.)} \\",
        r"\midrule",
    ]

    for group_name, vars in TABLE_GROUPS.items():
        # Only include variables that actually exist in the model results
        existing_vars = [v for v in vars if v in results.params.index]
        if not existing_vars:
            continue

        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{group_name}}}}} \\")
        lines.append(r"\midrule")

        for var in existing_vars:
            label = VAR_LABELS.get(var, var.replace("_", r"\_"))
            b = results.params[var]
            se = results.bse[var]
            p = results.pvalues[var]
            
            stars = ""
            if p < 0.01: stars = "***"
            elif p < 0.05: stars = "**"
            elif p < 0.1: stars = "*"

            lines.append(rf"{label} & {b:.3f}{stars} & ({se:.3f}) \\")

    lines += [
        r"\midrule",
        rf"Observations & \multicolumn{{2}}{{c}}{{{int(results.nobs):,}}} \\",
        rf"R-squared & \multicolumn{{2}}{{c}}{{{results.rsquared:.3f}}} \\",
        r"\bottomrule",
        r"\multicolumn{3}{l}{\small Standard errors clustered by household in parentheses} \\",
        r"\multicolumn{3}{l}{\small Constant term included in model but not shown.} \\",
        r"\multicolumn{3}{l}{\small $^\dagger$ Continuous variables are centered around their sample means.} \\",
        r"\multicolumn{3}{l}{\small *** $p<0.01$, ** $p<0.05$, * $p<0.1$} \\",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def main():
    config = load_config()
    outcome_name = config["variables"]["outcome_primary"][0]["name"].lower()

    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found.")
        return

    df = pd.read_parquet(master_path)
    
    print("Preparing data...")
    df = prepare_data(df, outcome_name)
    
    results = run_selection_model(df)

    print("\n" + "=" * 80)
    print("  Selection Model Summary")
    print("=" * 80)
    print(results.summary())

    out_dir = Path(config["output"]["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_path = out_dir / "selection_results.tex"

    write_latex(results, latex_path)
    print(f"\nWrote LaTeX table to {latex_path}")

if __name__ == "__main__":
    main()
