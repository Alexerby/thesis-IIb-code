"""
Simple OLS regression: Primary Outcome ~ Treatment.
No controls or fixed effects.
"""

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def main():
    config = load_config()
    
    # Get variable definitions
    outcome_def = config["variables"]["outcome_primary"][0]
    treatment_def = config["variables"]["treatment"][0]
    control_defs = config["variables"]["controls"]
    
    outcome_name = outcome_def["name"].lower()
    treatment_name = treatment_def["name"].lower()
    control_names = [c["name"].lower() for c in control_defs]
    
    outcome_label = outcome_def.get("label", outcome_name)
    treatment_label = treatment_def.get("label", treatment_name)
    
    master_path = Path("output/data/master.parquet")
    if not master_path.exists():
        print(f"Error: {master_path} not found. Run src/data/build_dataframe.py first.")
        return

    print(f"Loading master dataframe ...")
    df = pd.read_parquet(master_path)
    
    # 1. Filter and Recode Treatment
    # plb0097 values: 1=Yes, 2=No, 3=Not possible. 
    # We want 1=Yes, 0=No, and drop 3.
    df = df[df[treatment_name].isin([1, 2])].copy()
    df[treatment_name] = df[treatment_name].map({1: 1, 2: 0})
    print(f"Filtered to 'Yes' and 'No' groups. Recoded {treatment_name}: 1=Yes, 0=No.")

    # 2. Prepare Variables (Drop rows with any missing controls)
    # Note: 'hid' is kept for clustering but not used as a regressor
    all_vars = [outcome_name, treatment_name] + control_names
    df_reg = df[all_vars].dropna().copy()
    print(f"Final sample size after dropping missing values: {len(df_reg):,}")

    # 3. Build Regression Data
    y = df_reg[outcome_name].rename(outcome_label)
    
    # Exclude 'hid' from X (regressors)
    regressor_names = [c for c in control_names if c != "hid"]
    X = df_reg[[treatment_name] + regressor_names]
    
    # Rename for summary readability
    rename_dict = {treatment_name: f"{treatment_label} (1=Yes)"}
    for c in control_defs:
        if c["name"].lower() != "hid":
            rename_dict[c["name"].lower()] = c.get("label", c["name"])
    X = X.rename(columns=rename_dict)
    
    X = sm.add_constant(X)
    
    print(f"Running OLS with {len(regressor_names)} controls (Clustered by Household) ...")
    model = sm.OLS(y, X)
    # Use hid for clustering standard errors
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg['hid']})
    
    print("\n" + "="*80)
    print(f"  OLS Results: {outcome_label}")
    print("="*80)
    print(results.summary())

    # Write LaTeX table
    out_dir = Path(config["output"]["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_path = out_dir / "ols_results.tex"
    
    write_latex(results, outcome_label, f"{treatment_label} (1=Yes)", latex_path)
    print(f"\nWrote LaTeX table to {latex_path}")


def write_latex(results, outcome_label, treatment_label, out_path):
    """Writes OLS results to a LaTeX table matching descriptives.py style."""
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{OLS Regression Results: {outcome_label}}}",
        r"\label{tab:ols_results}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        rf" & Dependent variable: \\ \cline{{2-2}}",
        rf" & {outcome_label} \\",
        r"\midrule",
    ]

    # Add each variable's coefficient and SE
    for var in results.params.index:
        b = results.params[var]
        se = results.bse[var]
        p = results.pvalues[var]
        
        # Determine significance stars
        stars = ""
        if p < 0.01: stars = "***"
        elif p < 0.05: stars = "**"
        elif p < 0.1: stars = "*"
        
        clean_var = var.replace('_', r'\_')
        lines.append(rf"{clean_var} & {b:.3f}{stars} \\")
        lines.append(rf" & ({se:.3f}) \\")
        lines.append(r"\addlinespace")
    
    lines += [
        r"\midrule",
        rf"Observations & {int(results.nobs):,} \\",
        rf"R-squared & {results.rsquared:.3f} \\",
        r"\bottomrule",
        r"\multicolumn{2}{l}{\small Standard errors clustered by household in parentheses} \\",
        r"\multicolumn{2}{l}{\small *** p<0.01, ** p<0.05, * p<0.1}",
        r"\end{tabular}",
        r"\end{table}"
    ]
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
