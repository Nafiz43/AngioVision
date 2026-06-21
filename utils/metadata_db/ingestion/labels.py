"""Loading and filtering of the labeled DSA sequence CSV."""

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_labeled_csv(csv_path: Path) -> list[dict]:
    """
    Load labeled_DSA_2023_10_24.csv, returning one dict per unique file_path.

    Filtering rules:
      • Rows where angio_run contains 'other' (case-insensitive) → skipped
      • Rows with an empty file_path                              → skipped
      • Duplicate file_path values                                → first kept
    """
    if not csv_path.exists():
        log.error(f"Labeled CSV not found: {csv_path}")
        return []

    rows: list[dict]      = []
    seen_paths: set[str]  = set()
    skipped_other = skipped_dup = skipped_nopath = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            log.error("Labeled CSV appears to be empty.")
            return []

        norm = {h.strip().lower(): h for h in reader.fieldnames}

        angio_key = norm.get("angio_run")
        path_key  = norm.get("file_path")
        acc_key   = norm.get("accession")
        ser_key   = norm.get("seriesuid")
        run_key   = norm.get("run_type")

        if path_key is None:
            log.error(f"'file_path' column not found. Available: {list(reader.fieldnames)}")
            return []

        for row in reader:
            fp        = (row.get(path_key) or "").strip()
            angio_val = (row.get(angio_key) or "").strip().lower() if angio_key else ""

            if not fp:
                skipped_nopath += 1
                continue
            if "other" in angio_val:
                skipped_other += 1
                continue
            if fp in seen_paths:
                skipped_dup += 1
                continue
            seen_paths.add(fp)

            rows.append({
                "accession":  (row.get(acc_key)  or "").strip() if acc_key  else "",
                "series_uid": (row.get(ser_key)  or "").strip() if ser_key  else "",
                "run_type":   (row.get(run_key)  or "").strip() if run_key  else "",
                "angio_run":  angio_val,
                "file_path":  fp,
            })

    log.info(
        f"Labeled CSV — kept: {len(rows):,} | "
        f"skipped (other): {skipped_other:,} | "
        f"skipped (dup file_path): {skipped_dup:,} | "
        f"skipped (no path): {skipped_nopath:,}"
    )
    return rows
