"""
Build a .npy training dataset from ISQ files.

Input:  CSV with columns: sample_number, measurement_number, isq_path, status,
        and optionally motion_grade, bone_type.

Output: output_dir/
          scores.xlsx          — one row per scan: (Slice=scan_folder_name, Score, BoneType)
          s<sample>_m<meas>/   — one folder per scan
            slice_0000.npy, slice_0001.npy, ...
"""

from __future__ import annotations

import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.isq_reader import read_isq
from data.bone_cropper import crop_slice_per_bone


def _process_isq(
    isq_path: str,
    sample_num: int,
    measurement_num: int,
    grade: Optional[int],
    bone_type: Optional[str],
    out_root: Path,
    crop_size: int = 1024,
) -> Optional[tuple]:
    """
    Read an ISQ, crop each slice, save as .npy files.

    Returns (scan_folder_name, grade, bone_type) on success, None on error.
    The scan_folder_name is used as the key in scores.xlsx so HRPQCTDatasetV2 can look it up.
    """
    try:
        isq_path = Path(isq_path)
        if not isq_path.exists():
            print(f"ISQ not found: {isq_path}")
            return None

        volume = read_isq(isq_path, verbose=False)  # (z, y, x)
        scan_folder_name = f"s{sample_num:08d}_m{measurement_num:08d}"
        scan_folder = out_root / scan_folder_name
        scan_folder.mkdir(parents=True, exist_ok=True)

        for slice_idx, slice_arr in enumerate(volume):
            cropped = crop_slice_per_bone(slice_arr, crop_size=crop_size, threshold_method="percentile", percentile=85.0)
            slc = cropped.astype(np.float32)
            vmin, vmax = slc.min(), slc.max()
            if vmax > vmin:
                slc = (slc - vmin) / (vmax - vmin)
            np.save(str(scan_folder / f"slice_{slice_idx:04d}.npy"), slc)

        return (scan_folder_name, grade if grade is not None else -1, bone_type)

    except Exception as e:
        print(f"Error processing ISQ {isq_path}: {e}")
        return None


def build_isq_dataset(
    resolved_csv: str | Path,
    output_dir: str | Path,
    num_workers: int = 4,
    crop_size: int = 1024,
) -> pd.DataFrame:
    """
    Convert ISQ files listed in a resolved CSV into a .npy training dataset.

    scores.xlsx is keyed by scan folder name (e.g. 's00000362_m00001201') — one row per scan.
    This matches what HRPQCTDatasetV2 expects when it does p.parent.name lookup.
    """
    resolved_csv = Path(resolved_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolved_csv)
    required = {"sample_number", "measurement_number", "isq_path", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    valid_rows = df[df["status"] == "ok"].copy()
    print(f"Found {len(valid_rows)} valid ISQ files out of {len(df)} rows")
    if len(valid_rows) == 0:
        raise ValueError("No valid ISQ files found in resolved CSV")

    rows_to_process = []
    for _, row in valid_rows.iterrows():
        rows_to_process.append({
            "isq_path": row["isq_path"],
            "sample_num": int(row["sample_number"]),
            "measurement_num": int(row["measurement_number"]),
            "bone_type": row.get("bone_type") if "bone_type" in row else None,
            "grade": int(row["motion_grade"]) if "motion_grade" in row and not pd.isna(row.get("motion_grade")) else None,
        })

    worker = partial(_process_isq, out_root=output_dir, crop_size=crop_size)
    print(f"Processing {len(rows_to_process)} ISQ files with {num_workers} workers...")

    all_results = []
    seen_scans: set = set()

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(worker,
                      isq_path=r["isq_path"],
                      sample_num=r["sample_num"],
                      measurement_num=r["measurement_num"],
                      grade=r["grade"],
                      bone_type=r["bone_type"]): r
            for r in rows_to_process
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting ISQ"):
            result = future.result()
            if result is not None:
                scan_folder_name, score, bone_type = result
                if scan_folder_name not in seen_scans:
                    all_results.append({"Slice": scan_folder_name, "Score": score, "BoneType": bone_type})
                    seen_scans.add(scan_folder_name)

    out_df = pd.DataFrame(all_results)

    # Merge with existing scores.xlsx if present (for incremental builds)
    scores_path = output_dir / "scores.xlsx"
    if scores_path.exists():
        existing = pd.read_excel(scores_path)
        existing.columns = [c.strip() for c in existing.columns]
        existing_keys = set(existing["Slice"].astype(str).str.strip().str.lower())
        new_rows = out_df[~out_df["Slice"].str.lower().isin(existing_keys)]
        merged = pd.concat([existing, new_rows], ignore_index=True)
        merged.to_excel(scores_path, index=False)
        print(f"Added {len(new_rows)} new scans to existing scores.xlsx ({len(merged)} total)")
    else:
        out_df.to_excel(scores_path, index=False)
        print(f"Wrote {len(out_df)} scans to scores.xlsx")

    print(f".npy files saved in {output_dir}/s<sample>_m<measurement>/")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Convert ISQ files to .npy training dataset.")
    parser.add_argument("--resolved_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=1024)
    args = parser.parse_args()
    build_isq_dataset(args.resolved_csv, args.output_dir, args.num_workers, args.crop_size)


if __name__ == "__main__":
    main()
