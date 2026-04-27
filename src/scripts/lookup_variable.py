"""
Variable lookup tool for SOEP-Core data.

Usage:
    python src/lookup.py <search_term>              # AND: all words must match
    python src/lookup.py <search_term> --any        # OR: any word matches
    python src/lookup.py <search_term> --dataset pl # restrict to one dataset
    python src/lookup.py plh0151 --values           # also show value labels
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def search_variables(soep_dir: str, query: str, dataset: str | None, match_any: bool = False) -> list[dict]:
    results = []
    terms = query.lower().split()
    pattern = f"*_variables.csv" if not dataset else f"{dataset}_variables.csv"
    for path in sorted(Path(soep_dir).glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                # columns: study, dataset, version, variable, label, label_de
                variable = row[3]
                label = row[4]
                label_de = row[5] if len(row) > 5 else ""
                ds = row[1]
                haystack = (variable + " " + label).lower()
                matched = any(t in haystack for t in terms) if match_any else all(t in haystack for t in terms)
                if matched:
                    results.append({"dataset": ds, "variable": variable, "label": label, "label_de": label_de})
    return results


def get_values(soep_dir: str, variable: str, dataset: str | None) -> list[dict]:
    results = []
    pattern = f"*_values.csv" if not dataset else f"{dataset}_values.csv"
    for path in sorted(Path(soep_dir).glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                # columns: study, dataset, version, variable, value, label, label_de
                if row[3].lower() == variable.lower():
                    results.append({"dataset": row[1], "value": row[4], "label": row[5], "label_de": row[6] if len(row) > 6 else ""})
    return results


def main():
    parser = argparse.ArgumentParser(description="Look up SOEP-Core variables")
    parser.add_argument("query", help="Variable name or keyword to search for")
    parser.add_argument("--dataset", "-d", help="Restrict to a specific dataset (e.g. pl, pgen, pequiv)")
    parser.add_argument("--values", "-v", action="store_true", help="Also show value labels (treats query as exact variable name)")
    parser.add_argument("--any", dest="match_any", action="store_true", help="Match any word (OR); default is all words (AND)")
    args = parser.parse_args()

    config = load_config()
    soep_dir = config["data"]["soep_dir"]

    results = search_variables(soep_dir, args.query, args.dataset, match_any=args.match_any)

    if not results:
        print(f"No variables found matching '{args.query}'")
        sys.exit(1)

    print(f"\nFound {len(results)} variable(s) matching '{args.query}':\n")
    print(f"{'Dataset':<12} {'Variable':<20} {'Label (EN)'}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<12} {r['variable']:<20} {r['label']}")

    if args.values:
        print(f"\nValue labels for '{args.query}':\n")
        values = get_values(soep_dir, args.query, args.dataset)
        if not values:
            print("  (no value labels found)")
        else:
            ds = None
            for v in values:
                if v["dataset"] != ds:
                    ds = v["dataset"]
                    print(f"  [{ds}]")
                label = v["label"] or v["label_de"]
                print(f"    {v['value']:>5}  {label}")


if __name__ == "__main__":
    main()
