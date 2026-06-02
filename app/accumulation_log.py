"""
Append-only operator corrections log.

Each grading session appends rows to logs/grading_log.csv.
These accumulate over time and can be fed into prepare_dataset.py for retraining.
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

COLUMNS = [
    'participant_id',
    'sample_number',
    'measurement_number',
    'bone_type',
    'machine_grade_bin',
    'machine_grade_label',
    'operator_grade',
    'operator_approved',
    'operator_notes',
    'confidence',
    'graded_at',
]


class AccumulationLog:
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file or config.LOG_FILE)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def append(self, rows: list[dict]) -> None:
        """Append a list of result dicts to the log. Missing fields filled with empty string."""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
            for row in rows:
                row.setdefault('graded_at', datetime.now().isoformat(timespec='seconds'))
                writer.writerow({col: row.get(col, '') for col in COLUMNS})

    def corrections_only(self) -> list[dict]:
        """Return rows where the operator overrode the machine grade."""
        import csv as _csv
        if not self.log_file.exists():
            return []
        rows = []
        with open(self.log_file, newline='', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                if row.get('operator_approved', '').lower() == 'false':
                    rows.append(row)
        return rows

    def summary(self) -> dict:
        import csv as _csv
        if not self.log_file.exists():
            return {'total': 0, 'approved': 0, 'corrected': 0}
        total = approved = corrected = 0
        with open(self.log_file, newline='', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                total += 1
                if row.get('operator_approved', '').lower() == 'true':
                    approved += 1
                else:
                    corrected += 1
        return {'total': total, 'approved': approved, 'corrected': corrected}
