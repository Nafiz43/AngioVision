"""Labeled-CSV loading, DICOM file index and path resolution.

Stdlib-only (csv / sqlite3 / os) — safe to import without torch/pydicom.
"""

import csv
import logging
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Labeled CSV  (mode-aware filtering)
# ═══════════════════════════════════════════════════════════════════════════════

def load_labeled_csv_grouped(csv_path: Path, frame_mode: str) -> dict[str, list[dict]]:
    """Load the labeled sequence CSV grouped by angio_run category.

    Rows labeled 'other', duplicates and rows without a file path are skipped.
    Rows missing the frame annotation required by *frame_mode* ('best' needs
    Best_Image, 'fl' needs First/Last_Diag_Image) are excluded and written to
    a sidecar CSV next to the input for later annotation.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    missing_rows: list[dict] = []
    fieldnames: list[str] = []
    skip_other = skip_path = skip_dup = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {csv_path}")
        fieldnames = list(reader.fieldnames)
        norm = {h.strip().lower(): h for h in fieldnames}

        def col(k):
            return norm.get(k.lower())

        path_key  = col("file_path");  angio_key = col("angio_run")
        acc_key   = col("accession");  ser_key   = col("seriesuid")
        run_key   = col("run_type");   best_key  = col("best_image")
        first_key = col("first_diag_image"); last_key = col("last_diag_image")

        if not path_key:
            raise ValueError("'file_path' column not found")

        def _p1(raw):
            """1-based CSV annotation → 0-based frame index; -1 if absent/invalid."""
            try:
                v = float((raw or "").strip())
                return int(v) - 1 if v >= 1 else -1
            except (ValueError, TypeError):
                return -1

        for row in reader:
            fp    = (row.get(path_key)  or "").strip()
            angio = (row.get(angio_key) or "").strip() if angio_key else ""
            if not fp:
                skip_path += 1; continue
            if "other" in angio.lower():
                skip_other += 1; continue
            if fp in seen:
                skip_dup += 1; continue
            seen.add(fp)

            best_image_idx = _p1(row.get(best_key,  "") if best_key  else "")
            first_diag_idx = _p1(row.get(first_key, "") if first_key else "")
            last_diag_idx  = _p1(row.get(last_key,  "") if last_key  else "")
            if first_diag_idx >= 0 and last_diag_idx >= 0 and last_diag_idx < first_diag_idx:
                first_diag_idx = last_diag_idx = -1

            exclude = False
            if frame_mode == "best" and best_image_idx < 0:
                exclude = True
            elif frame_mode == "fl" and (first_diag_idx < 0 or last_diag_idx < 0):
                exclude = True
            if exclude:
                missing_rows.append(dict(row)); continue

            groups[angio].append({
                "accession": (row.get(acc_key) or "").strip() if acc_key else "",
                "series_uid": (row.get(ser_key) or "").strip() if ser_key else "",
                "run_type":  (row.get(run_key) or "").strip() if run_key else "",
                "angio_run": angio, "file_path": fp,
                "best_image_idx": best_image_idx,
                "first_diag_idx": first_diag_idx,
                "last_diag_idx":  last_diag_idx,
            })

    if missing_rows:
        sidecar = csv_path.parent / {
            "best": "missing_best_image.csv",
            "fl":   "missing_diag_window.csv",
        }.get(frame_mode, "missing_frames.csv")
        with open(sidecar, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader(); w.writerows(missing_rows)
        log.warning(f"  {len(missing_rows):,} sequences excluded (mode={frame_mode}) → {sidecar}")

    total = sum(len(v) for v in groups.values())
    log.info(f"CSV loaded — {len(groups)} categories, {total:,} sequences retained "
             f"(skipped other={skip_other:,}, dup={skip_dup:,}, no-path={skip_path:,}, "
             f"no-annotation={len(missing_rows):,})")
    return dict(groups)


# ═══════════════════════════════════════════════════════════════════════════════
# DICOM index  (SQLite fast path, filesystem-walk fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def stem_of(fp: str) -> Optional[str]:
    """File-name stem of a CSV file_path — the join key against the DICOM index."""
    p = (fp or "").strip()
    if not p:
        return None
    fname = p.rsplit("/", 1)[-1] if "/" in p else p
    return fname[:-4] if fname.lower().endswith(".dcm") else (fname or None)


def build_dicom_index(dicom_root: Path, sqlite_db: Optional[Path] = None) -> dict[str, str]:
    """Map file-name stem → absolute DICOM path.

    Prefers the metadata_db SQLite `dicom_files` table (fast); falls back to
    walking *dicom_root* for .dcm files.
    """
    if sqlite_db and sqlite_db.exists():
        try:
            con = sqlite3.connect(str(sqlite_db))
            n = con.execute(
                "SELECT COUNT(*) FROM dicom_files WHERE source_file IS NOT NULL"
            ).fetchone()[0]
            if n > 0:
                log.info(f"Building DICOM index from SQLite ({n:,} rows) …")
                idx: dict[str, str] = {}
                for sf, sp in con.execute(
                    "SELECT source_file, source_path FROM dicom_files "
                    "WHERE source_file IS NOT NULL AND source_path IS NOT NULL"
                ):
                    s = Path(sf).stem
                    if s and s not in idx:
                        idx[s] = sp
                con.close()
                log.info(f"Index ready: {len(idx):,} stems (SQLite)")
                return idx
            con.close()
        except Exception as e:
            log.warning(f"SQLite index failed ({e}) — filesystem walk …")

    log.info(f"Walking {dicom_root} …")
    idx: dict[str, str] = {}
    dups = 0
    for dp, _, fnames in os.walk(str(dicom_root)):
        for fn in fnames:
            if fn.lower().endswith(".dcm"):
                s = Path(fn).stem
                if s in idx:
                    dups += 1
                else:
                    idx[s] = str(Path(dp) / fn)
    log.info(f"Index ready: {len(idx):,} stems ({dups} dups ignored)")
    return idx


def resolve_paths(groups: dict[str, list[dict]], idx: dict[str, str]) -> dict[str, list[dict]]:
    """Attach 'dicom_path'/'stem' to every sequence resolvable via the index; drop the rest."""
    resolved: dict[str, list[dict]] = {}
    n_missing = 0
    for label, seqs in groups.items():
        kept = []
        for seq in seqs:
            s = stem_of(seq["file_path"])
            if s and s in idx:
                kept.append({**seq, "dicom_path": idx[s], "stem": s})
            else:
                n_missing += 1
        if kept:
            resolved[label] = kept
    log.info(f"Path resolution — found: {sum(len(v) for v in resolved.values()):,}, "
             f"missing (dropped): {n_missing:,}")
    return resolved
