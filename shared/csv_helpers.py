"""Append-only CSV helpers used across all pipeline scripts.

Two flavours are provided so callers can work with either ``pathlib.Path``
or plain ``str`` paths.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ---------- Path-based helpers (used by most pipeline scripts) ----------


def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    """Create a CSV file with *columns* as header if it does not already exist."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    """Append a single *row* (dict) to a CSV file preserving column order."""
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


# ---------- str-based helpers (used by validation pipelines) ----------


def ensure_parent_dir(file_path: str) -> None:
    """Create the parent directory of *file_path* if it does not exist."""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_row_csv(csv_path: str, row: Dict, fieldnames: List[str]) -> None:
    """Append *row* to *csv_path*, writing the header first if the file is new."""
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
