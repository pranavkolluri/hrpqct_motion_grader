"""Resolve XCT scan CSV rows to mounted-drive ISQ paths (Y:\\ workflow)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    by_norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in by_norm:
            return by_norm[key]
    return None


def _to_int_or_none(val) -> Optional[int]:
    if pd.isna(val):
        return None
    txt = str(val).strip()
    if not txt:
        return None
    m = re.search(r"\d+", txt)
    return int(m.group()) if m else None


def _parse_grade(primary, secondary) -> Optional[int]:
    p = _to_int_or_none(primary)
    if p is not None:
        return p
    return _to_int_or_none(secondary)


def _extract_bone_type(participant_id: str) -> Optional[str]:
    """Extract 'Tibia' or 'Radius' from a participant ID string."""
    if not participant_id or pd.isna(participant_id):
        return None
    text = str(participant_id).upper()
    if "TIBIA" in text:
        return "Tibia"
    elif "RADIUS" in text:
        return "Radius"
    return None


def first_isq(scan_dir: Path) -> Optional[Path]:
    """Return the first .ISQ or .ISQ;* file found in scan_dir, or None."""
    if not scan_dir.exists() or not scan_dir.is_dir():
        return None
    candidates = [p for p in scan_dir.iterdir() if p.is_file() and
                  (p.name.lower().endswith(".isq") or ".isq;" in p.name.lower())]
    return sorted(candidates, key=lambda x: x.name.lower())[0] if candidates else None


def resolve_csv_to_isq(
    input_csv: str | Path,
    output_csv: str | Path,
    base_root: str | Path = r"Y:\data",
    require_isq: bool = False,
) -> pd.DataFrame:
    """
    Resolve sample/measurement numbers in a CSV to ISQ file paths.

    Reads columns: Participant ID, Sample Number, Measurement Number, motion grade (optional).
    Writes resolved CSV with: participant_id, bone_type, sample_number, measurement_number,
    scan_dir, isq_path, motion_grade, status, notes.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    base_root = Path(base_root)

    df = pd.read_csv(input_csv)

    sample_col = _find_col(df, ["Sample Number", "SampleNumber"])
    meas_col = _find_col(df, ["Measurement Number", "MeasurementNumber"])
    participant_col = _find_col(df, ["Participant ID", "ParticipantID"])
    grade_col = _find_col(df, ["Motion grades for Pranav", "Motion Grade", "Score"])
    notes_col = _find_col(df, ["Notes from Gabby", "Notes"])

    if sample_col is None or meas_col is None:
        raise ValueError("Could not find sample/measurement columns in input CSV.")

    rows = []
    for _, row in df.iterrows():
        sample_num = _to_int_or_none(row.get(sample_col))
        meas_num = _to_int_or_none(row.get(meas_col))
        if sample_num is None or meas_num is None:
            continue

        participant_id = ""
        if participant_col is not None and not pd.isna(row.get(participant_col)):
            participant_id = str(row.get(participant_col)).strip()

        grade = None
        if grade_col is not None or notes_col is not None:
            grade = _parse_grade(
                row.get(grade_col) if grade_col else None,
                row.get(notes_col) if notes_col else None,
            )

        bone_type = _extract_bone_type(participant_id)

        scan_dir = base_root / f"{sample_num:08d}" / f"{meas_num:08d}"
        isq_path = first_isq(scan_dir)

        if not scan_dir.exists():
            status, note = "missing_scan_dir", "scan directory not found"
        elif isq_path is None:
            status, note = "missing_isq", "no .isq file found"
        else:
            status, note = "ok", ""

        if require_isq and status != "ok":
            continue

        rows.append({
            "participant_id": participant_id,
            "bone_type": bone_type,
            "sample_number": sample_num,
            "measurement_number": meas_num,
            "scan_dir": str(scan_dir),
            "isq_path": str(isq_path) if isq_path is not None else "",
            "motion_grade": grade,
            "status": status,
            "notes": note,
        })

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve XCT scan CSV to ISQ paths.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--base_root", default=r"Y:\data")
    parser.add_argument("--require_isq", action="store_true")
    args = parser.parse_args()

    out_df = resolve_csv_to_isq(args.input_csv, args.output_csv, args.base_root, args.require_isq)
    ok = int((out_df["status"] == "ok").sum()) if len(out_df) else 0
    print(f"Wrote {len(out_df)} rows to {args.output_csv} ({ok} with status=ok)")


if __name__ == "__main__":
    main()
