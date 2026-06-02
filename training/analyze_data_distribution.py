"""
Analyze motion grade data distribution across anatomical sites.
Helps inform training strategy (single model vs per-bone vs per-site).

Usage:
    python training/analyze_data_distribution.py --csv "path/to/labeled.csv"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _extract_anatomy(participant_id: str) -> str | None:
    if not participant_id or pd.isna(participant_id):
        return None
    text = str(participant_id).upper()
    patterns = [
        (r"UD.*TIBIA",          "UD_Tibia"),
        (r"D.*TIBIA(?!.*UD)",   "D_Tibia"),
        (r"UD.*RADIUS",         "UD_Radius"),
        (r"D.*RADIUS(?!.*UD)",  "D_Radius"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text):
            return label
    return None


def analyze_csv(csv_path: str | Path) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    pid_col = next((c for c in df.columns if 'participant' in c.lower()), None)
    grade_col = next((c for c in df.columns if 'grade' in c.lower() or 'score' in c.lower()), None)
    sample_col = next((c for c in df.columns if 'sample' in c.lower()), None)
    meas_col = next((c for c in df.columns if 'measurement' in c.lower()), None)

    if pid_col:
        df['anatomy'] = df[pid_col].apply(_extract_anatomy)

    print(f"\nTotal rows: {len(df)}")

    # Drop blank rows
    if sample_col and meas_col:
        df = df.dropna(subset=[sample_col, meas_col])
    print(f"Non-blank rows: {len(df)}")

    results: dict = {}

    if grade_col and df[grade_col].notna().any():
        grade_dist = df[grade_col].value_counts().sort_index()
        print(f"\nGrade distribution ({grade_col}):")
        for g, cnt in grade_dist.items():
            print(f"  Grade {g}: {cnt:4d} ({100*cnt/len(df):.1f}%)")
        results['grade_distribution'] = grade_dist.to_dict()

    if pid_col and 'anatomy' in df.columns:
        anatomy_dist = df['anatomy'].value_counts()
        print(f"\nAnatomy distribution:")
        for a, cnt in anatomy_dist.items():
            print(f"  {a}: {cnt}")

        if grade_col:
            print(f"\nGrade distribution by anatomy:")
            crosstab = pd.crosstab(df['anatomy'], df[grade_col])
            print(crosstab.to_string())
            results['crosstab'] = crosstab.to_dict()

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze motion grade data distribution.")
    parser.add_argument("--csv", required=True, help="Input CSV with participant IDs, sample/measurement numbers, and grades")
    args = parser.parse_args()
    analyze_csv(args.csv)


if __name__ == "__main__":
    main()
