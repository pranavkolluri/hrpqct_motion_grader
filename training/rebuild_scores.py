"""
Rebuild scores.xlsx from a resolved CSV.

Useful if scores.xlsx was generated with an old format (slice names instead of
scan folder names). Generates one row per scan keyed by scan folder name
(s{sample:08d}_m{measurement:08d}) as required by HRPQCTDatasetV2.

Usage:
    python training/rebuild_scores.py --csv F:/data/resolved_all_ok.csv --dataset_dir F:/data/training_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rebuild(csv_path: str | Path, dataset_dir: str | Path) -> None:
    csv_path = Path(csv_path)
    dataset_dir = Path(dataset_dir)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"sample_number", "measurement_number", "motion_grade", "bone_type", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ok = df[df["status"].str.strip().str.lower() == "ok"].copy()
    print(f"Using {len(ok)} rows with status=ok")

    ok["Slice"] = ok.apply(
        lambda r: f"s{int(r['sample_number']):08d}_m{int(r['measurement_number']):08d}", axis=1
    )
    ok["Score"] = ok["motion_grade"].astype(int)
    ok["BoneType"] = ok["bone_type"].str.strip()

    existing_folders = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
    unlabeled = existing_folders - set(ok["Slice"])
    if unlabeled:
        print(f"  {len(unlabeled)} dataset folders have no label (will be ignored during training)")

    out = dataset_dir / "scores.xlsx"
    ok[["Slice", "Score", "BoneType"]].to_excel(out, index=False)
    print(f"Wrote {len(ok)} rows to {out}")
    print(f"  Tibia: {(ok['BoneType'] == 'Tibia').sum()}  Radius: {(ok['BoneType'] == 'Radius').sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild scores.xlsx from resolved CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()
    rebuild(args.csv, args.dataset_dir)


if __name__ == "__main__":
    main()
