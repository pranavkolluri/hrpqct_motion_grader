"""
Build or update a .npy training dataset.

Two modes:
  1. Initial build:  --input_csv <labeled_csv> → resolves ISQ paths → builds .npy dataset
  2. Merge corrections: --corrections <grading_log.csv> → adds operator-corrected scans to an existing dataset

Both modes update scores.xlsx in-place with one row per scan.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, 'data')
for _p in [_ROOT, _DATA]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from csv_path_resolver import resolve_csv_to_isq, first_isq
from isq_data_builder import _process_isq, build_isq_dataset


def merge_corrections(
    dataset_dir: str | Path,
    corrections_csv: str | Path,
    isq_root: str | Path,
    crop_size: int = 1024,
) -> int:
    """
    Read operator corrections from grading_log.csv and add new scans to dataset_dir.

    Only processes scans that are NOT already in scores.xlsx.
    Returns the number of new scans added.
    """
    dataset_dir = Path(dataset_dir)
    corrections_csv = Path(corrections_csv)
    isq_root = Path(isq_root)

    if not corrections_csv.exists():
        print(f"Corrections file not found: {corrections_csv}")
        return 0

    corrections = pd.read_csv(corrections_csv)
    if corrections.empty:
        print("Corrections log is empty.")
        return 0

    # Only include rows with an operator-provided grade
    grade_col = 'operator_grade' if 'operator_grade' in corrections.columns else 'motion_grade'
    corrections = corrections.dropna(subset=[grade_col]).copy()
    print(f"Found {len(corrections)} correction rows with operator grades")

    # Load existing scores.xlsx
    scores_path = dataset_dir / 'scores.xlsx'
    if scores_path.exists():
        existing_df = pd.read_excel(scores_path)
        existing_df.columns = [c.strip() for c in existing_df.columns]
        existing_keys = set(existing_df['Slice'].astype(str).str.strip().str.lower())
    else:
        existing_df = pd.DataFrame(columns=['Slice', 'Score', 'BoneType'])
        existing_keys = set()

    new_rows = []
    added = 0

    for _, row in corrections.iterrows():
        try:
            sample_num = int(row['sample_number'])
            meas_num = int(row['measurement_number'])
        except (KeyError, ValueError):
            continue

        scan_folder_name = f"s{sample_num:08d}_m{meas_num:08d}"
        if scan_folder_name.lower() in existing_keys:
            continue  # already in dataset

        grade = int(row[grade_col])
        bone_type = row.get('bone_type', None)
        if pd.isna(bone_type):
            bone_type = None

        # Find ISQ
        scan_dir = isq_root / f"{sample_num:08d}" / f"{meas_num:08d}"
        isq_path = first_isq(scan_dir)
        if isq_path is None:
            print(f"  Warning: ISQ not found for sample={sample_num} meas={meas_num} — skipping")
            continue

        # Process ISQ → .npy slices
        print(f"  Processing {scan_folder_name} (grade {grade}) ...")
        result = _process_isq(str(isq_path), sample_num, meas_num, grade, bone_type, dataset_dir, crop_size)
        if result is not None:
            sf_name, score, bt = result
            new_rows.append({'Slice': sf_name, 'Score': score, 'BoneType': bt})
            existing_keys.add(sf_name.lower())
            added += 1

    if new_rows:
        merged = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_excel(scores_path, index=False)
        print(f"\nAdded {added} new scans to dataset. Total: {len(merged)} scans in scores.xlsx")
    else:
        print("No new scans to add from corrections.")

    return added


def main():
    parser = argparse.ArgumentParser(
        description="Build or update a .npy training dataset from ISQ files + operator corrections."
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Output .npy dataset directory (may already exist for incremental updates)")
    parser.add_argument("--isq_root", default=r"Y:\data",
                        help="Mounted XCT root directory (default: Y:\\data)")

    build_group = parser.add_argument_group("Initial build from labeled CSV")
    build_group.add_argument("--input_csv",
                             help="Labeled CSV (Participant ID, Sample Number, Measurement Number, grade)")
    build_group.add_argument("--num_workers", type=int, default=4)
    build_group.add_argument("--crop_size", type=int, default=1024)

    corr_group = parser.add_argument_group("Merge operator corrections")
    corr_group.add_argument("--corrections",
                            help="Path to grading_log.csv from the grading app")

    args = parser.parse_args()

    if not args.input_csv and not args.corrections:
        parser.error("Provide --input_csv (initial build) or --corrections (merge corrections), or both.")

    dataset_dir = Path(args.dataset_dir)
    scores_path = dataset_dir / 'scores.xlsx'

    if args.input_csv:
        print("=" * 60)
        print("Building dataset from labeled CSV")
        print("=" * 60)
        resolved_csv = dataset_dir / "_resolved_paths.csv"
        resolved_csv.parent.mkdir(parents=True, exist_ok=True)

        from csv_path_resolver import resolve_csv_to_isq as _resolve
        _resolve(args.input_csv, str(resolved_csv), base_root=args.isq_root, require_isq=True)

        build_isq_dataset(str(resolved_csv), str(dataset_dir), args.num_workers, args.crop_size)

    if args.corrections:
        print("\n" + "=" * 60)
        print("Merging operator corrections")
        print("=" * 60)
        if not scores_path.exists():
            print("Warning: scores.xlsx not found in dataset_dir — will create new dataset from corrections only.")
        merge_corrections(dataset_dir, args.corrections, args.isq_root, args.crop_size)


if __name__ == "__main__":
    main()
