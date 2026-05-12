#!/usr/bin/env python3
"""
DICOM Consistency Check
=======================

Parallel header-only scan of an input DICOM tree.
No image extraction — reads metadata only.

Checks performed per file:
    1. Unreadable files       — header cannot be parsed at all
    2. Missing AccessionNumber — tag absent or empty
    3. Duplicate SOPInstanceUID — same UID shared by more than one file path
    4. AccessionNumber / StudyInstanceUID mismatch — same accession maps to
       more than one StudyInstanceUID (indicates data integrity issue)

Architecture
------------
- Walk input tree once (main process) → collect leaf directories
- Divide leaf dirs across N workers
- Each worker reads every DICOM header, returns lightweight records
- Main process merges all records and runs all four checks
- Writes a detailed Markdown report + four CSVs into LOG_DIR/consistency_<ts>/

Output
------
    LOG_DIR/consistency_<YYYYMMDD_HHMMSS>/
        summary.md                  ← human-readable full report
        unreadable.csv              ← check 1
        no_accession.csv            ← check 2
        duplicate_sop.csv           ← check 3  (grouped by SOPInstanceUID)
        accession_study_mismatch.csv ← check 4  (grouped by AccessionNumber)

Tips
----
- --workers      number of scan workers  (default: cpu_count - 1)
- --input_root   override INPUT_ROOT from config.py
"""

import csv
import datetime
import gc
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pydicom
from pydicom.multival import MultiValue
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Load input path from config.py ────────────────────────────────────────────
try:
    from config import INPUT_ROOT
except ImportError as e:
    raise SystemExit(
        f"Could not import INPUT_ROOT from config.py: {e}\n"
        "Make sure config.py is in the same directory as this script."
    )

# =========================================================
# Configuration
# =========================================================
LOG_DIR = Path("/data/Deep_Angiography/DICOM-metadata-stats")

GC_NUDGE_INTERVAL = 10_000


# =========================================================
# Lightweight per-file record (returned by each worker)
# =========================================================
@dataclass
class FileRecord:
    """
    One record per DICOM file found.
    Carries only the fields needed for all four checks.
    """
    file_path:         str
    readable:          bool            # False  → check 1
    accession_number:  str             # ""     → check 2
    sop_instance_uid:  str             # ""     → cannot deduplicate
    study_instance_uid: str            # ""     → cannot check 4
    error_msg:         str             # populated only when readable=False


# =========================================================
# Worker-level accumulator
# =========================================================
@dataclass
class WorkerResult:
    records:    List[FileRecord] = field(default_factory=list)
    total_seen: int = 0          # raw files touched (including non-DICOM)
    total_dicom: int = 0         # files that passed is_probably_dicom()


# =========================================================
# Utilities
# =========================================================
def is_probably_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except Exception:
        return False
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def get_tag_str(ds, tag: str) -> str:
    val = getattr(ds, tag, None)
    if val is None:
        return ""
    if isinstance(val, MultiValue):
        val = val[0] if val else ""
    return str(val).strip()


# =========================================================
# Directory discovery
# =========================================================
def collect_leaf_dirs(root_dir: Path) -> List[Path]:
    """
    Return directories that contain at least one file directly.
    No rglob overlap — leaf dirs never contain each other.
    """
    leaf_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if filenames:
            leaf_dirs.append(Path(dirpath))
    return leaf_dirs


# =========================================================
# Scan worker  (header-only, no pixel access)
# =========================================================
def _scan_leaf_dirs(leaf_dir_strs: List[str]) -> WorkerResult:
    """
    For each DICOM file found in the given leaf directories:
      - Attempt to read the header (stop_before_pixels=True)
      - Extract SOPInstanceUID, AccessionNumber, StudyInstanceUID
      - Return one FileRecord per file

    GC hygiene: explicit del ds + periodic gc.collect()
    """
    result     = WorkerResult()
    file_count = 0

    for leaf_str in leaf_dir_strs:
        for f in Path(leaf_str).iterdir():
            result.total_seen += 1

            if not f.is_file() or not is_probably_dicom(f):
                continue

            result.total_dicom += 1
            file_count         += 1
            if file_count % GC_NUDGE_INTERVAL == 0:
                gc.collect()

            # ── Attempt header read ────────────────────────────────────────
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception as e:
                result.records.append(FileRecord(
                    file_path          = str(f),
                    readable           = False,
                    accession_number   = "",
                    sop_instance_uid   = "",
                    study_instance_uid = "",
                    error_msg          = f"{type(e).__name__}: {e}",
                ))
                continue

            acc   = get_tag_str(ds, "AccessionNumber")
            uid   = get_tag_str(ds, "SOPInstanceUID")
            study = get_tag_str(ds, "StudyInstanceUID")

            del ds

            result.records.append(FileRecord(
                file_path          = str(f),
                readable           = True,
                accession_number   = acc,
                sop_instance_uid   = uid,
                study_instance_uid = study,
                error_msg          = "",
            ))

    return result


# =========================================================
# Merge worker results
# =========================================================
def _merge_results(worker_results: List[WorkerResult]) -> Tuple[List[FileRecord], int, int]:
    all_records  = []
    total_seen   = 0
    total_dicom  = 0
    for wr in worker_results:
        all_records  += wr.records
        total_seen   += wr.total_seen
        total_dicom  += wr.total_dicom
    return all_records, total_seen, total_dicom


# =========================================================
# Four consistency checks
# =========================================================
def run_checks(records: List[FileRecord]) -> Dict:
    """
    Run all four checks over the merged record list.
    Returns a dict with structured results for each check.
    """

    # ── Check 1: Unreadable files ─────────────────────────────────────────────
    unreadable = [
        {"file": r.file_path, "error": r.error_msg}
        for r in records if not r.readable
    ]

    # ── Check 2: Missing AccessionNumber ──────────────────────────────────────
    no_accession = [
        {"file": r.file_path, "sop_instance_uid": r.sop_instance_uid}
        for r in records
        if r.readable and not r.accession_number.strip()
    ]

    # ── Check 3: Duplicate SOPInstanceUID ─────────────────────────────────────
    # Group all file paths by SOPInstanceUID
    uid_to_paths: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r.readable and r.sop_instance_uid:
            uid_to_paths[r.sop_instance_uid].append(r.file_path)

    # Keep only groups with more than one path
    duplicate_sop_groups = {
        uid: sorted(paths)
        for uid, paths in uid_to_paths.items()
        if len(paths) > 1
    }
    # Flatten to CSV rows — one row per file, grouped by uid
    duplicate_sop_rows = []
    for uid, paths in sorted(duplicate_sop_groups.items()):
        for i, path in enumerate(paths):
            duplicate_sop_rows.append({
                "sop_instance_uid": uid,
                "copy_index":       i + 1,
                "total_copies":     len(paths),
                "file":             path,
            })

    # ── Check 4: AccessionNumber → multiple StudyInstanceUIDs ─────────────────
    # For each AccessionNumber, collect the set of StudyInstanceUIDs seen
    acc_to_studies: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.readable and r.accession_number.strip() and r.study_instance_uid:
            acc_to_studies[r.accession_number][r.study_instance_uid].append(r.file_path)

    # Keep only accessions that map to MORE than one distinct StudyInstanceUID
    mismatch_groups = {
        acc: dict(study_map)
        for acc, study_map in acc_to_studies.items()
        if len(study_map) > 1
    }
    # Flatten to CSV rows — one row per (accession, study, file)
    mismatch_rows = []
    for acc, study_map in sorted(mismatch_groups.items()):
        for study_uid, paths in sorted(study_map.items()):
            for path in sorted(paths):
                mismatch_rows.append({
                    "accession_number":   acc,
                    "study_instance_uid": study_uid,
                    "n_studies_for_acc":  len(study_map),
                    "file":               path,
                })

    return {
        "unreadable":         unreadable,
        "no_accession":       no_accession,
        "duplicate_sop":      duplicate_sop_rows,
        "duplicate_sop_groups": duplicate_sop_groups,   # for MD report
        "mismatch":           mismatch_rows,
        "mismatch_groups":    mismatch_groups,           # for MD report
    }


# =========================================================
# Report writing
# =========================================================
def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    run_dir:       Path,
    checks:        Dict,
    total_seen:    int,
    total_dicom:   int,
    total_records: int,
    n_workers:     int,
    input_root:    Path,
    elapsed_sec:   float,
    run_ts:        str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dt = (f"{run_ts[:4]}-{run_ts[4:6]}-{run_ts[6:8]} "
          f"{run_ts[9:11]}:{run_ts[11:13]}:{run_ts[13:15]}")

    # ── CSVs ──────────────────────────────────────────────────────────────────
    if checks["unreadable"]:
        _write_csv(
            run_dir / "unreadable.csv",
            ["file", "error"],
            checks["unreadable"],
        )

    if checks["no_accession"]:
        _write_csv(
            run_dir / "no_accession.csv",
            ["file", "sop_instance_uid"],
            checks["no_accession"],
        )

    if checks["duplicate_sop"]:
        _write_csv(
            run_dir / "duplicate_sop.csv",
            ["sop_instance_uid", "copy_index", "total_copies", "file"],
            checks["duplicate_sop"],
        )

    if checks["mismatch"]:
        _write_csv(
            run_dir / "accession_study_mismatch.csv",
            ["accession_number", "study_instance_uid", "n_studies_for_acc", "file"],
            checks["mismatch"],
        )

    # ── summary.md ────────────────────────────────────────────────────────────
    n_unreadable  = len(checks["unreadable"])
    n_no_acc      = len(checks["no_accession"])
    n_dup_uids    = len(checks["duplicate_sop_groups"])
    n_dup_files   = len(checks["duplicate_sop"])
    n_mismatch_acc = len(checks["mismatch_groups"])
    n_mismatch_files = len(checks["mismatch"])

    overall_ok = all([
        n_unreadable   == 0,
        n_no_acc       == 0,
        n_dup_uids     == 0,
        n_mismatch_acc == 0,
    ])

    md = run_dir / "summary.md"
    with md.open("w", encoding="utf-8") as f:

        f.write("# DICOM Consistency Check Report\n\n")
        f.write(f"**Run timestamp:** {dt}  \n")
        f.write(f"**Elapsed:** {elapsed_sec:.1f} s  \n")
        f.write(f"**Input root:** `{input_root}`  \n")
        f.write(f"**Workers:** {n_workers}  \n\n")
        f.write("---\n\n")

        # Overall health
        health_icon = "✅" if overall_ok else "⚠️"
        f.write(f"## {health_icon} Overall Health\n\n")
        f.write("| Metric | Count |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total files seen (all types)   | **{total_seen}** |\n")
        f.write(f"| DICOM files identified         | **{total_dicom}** |\n")
        f.write(f"| Successfully read (headers)    | **{total_records}** |\n\n")

        # Check summary table
        f.write("## Check Summary\n\n")
        f.write("| # | Check | Status | Count |\n")
        f.write("|---|-------|--------|-------|\n")

        def _row(num, label, count, unit="file(s)"):
            icon = "✅" if count == 0 else "❌"
            return f"| {num} | {label} | {icon} | **{count}** {unit} |\n"

        f.write(_row(1, "Unreadable files",                        n_unreadable))
        f.write(_row(2, "Missing AccessionNumber",                 n_no_acc))
        f.write(_row(3, "Duplicate SOPInstanceUID",                n_dup_uids,  "unique UID(s)"))
        f.write(_row(4, "AccessionNumber→StudyInstanceUID mismatch", n_mismatch_acc, "accession(s)"))
        f.write("\n")

        f.write("---\n\n")

        # ── Check 1 detail ────────────────────────────────────────────────────
        f.write("## Check 1 — Unreadable Files\n\n")
        if n_unreadable == 0:
            f.write("✅ All DICOM files could be read successfully.\n\n")
        else:
            f.write(f"❌ **{n_unreadable}** file(s) could not be read.  \n")
            f.write("Full list: `unreadable.csv`\n\n")
            f.write("| File | Error |\n")
            f.write("|------|-------|\n")
            for row in checks["unreadable"][:20]:   # first 20 inline
                f.write(f"| `{row['file']}` | `{row['error']}` |\n")
            if n_unreadable > 20:
                f.write(f"\n*... and {n_unreadable - 20} more — see `unreadable.csv`*\n")
            f.write("\n")

        # ── Check 2 detail ────────────────────────────────────────────────────
        f.write("## Check 2 — Missing AccessionNumber\n\n")
        if n_no_acc == 0:
            f.write("✅ Every readable DICOM file has an AccessionNumber.\n\n")
        else:
            f.write(f"❌ **{n_no_acc}** file(s) have no AccessionNumber.  \n")
            f.write("Full list: `no_accession.csv`\n\n")
            f.write("| File | SOPInstanceUID |\n")
            f.write("|------|----------------|\n")
            for row in checks["no_accession"][:20]:
                f.write(f"| `{row['file']}` | `{row['sop_instance_uid']}` |\n")
            if n_no_acc > 20:
                f.write(f"\n*... and {n_no_acc - 20} more — see `no_accession.csv`*\n")
            f.write("\n")

        # ── Check 3 detail ────────────────────────────────────────────────────
        f.write("## Check 3 — Duplicate SOPInstanceUID\n\n")
        if n_dup_uids == 0:
            f.write("✅ Every DICOM file has a unique SOPInstanceUID.\n\n")
        else:
            f.write(f"❌ **{n_dup_uids}** SOPInstanceUID(s) are shared by more than one file  \n")
            f.write(f"(**{n_dup_files}** files total involved).  \n")
            f.write("Full list: `duplicate_sop.csv`\n\n")
            f.write("> Files are grouped by SOPInstanceUID. "
                    "Groups are sorted by copy count (most copies first).\n\n")

            sorted_groups = sorted(
                checks["duplicate_sop_groups"].items(),
                key=lambda kv: len(kv[1]),
                reverse=True,
            )
            for idx, (uid, paths) in enumerate(sorted_groups[:10], start=1):
                f.write(f"### Group {idx} — {len(paths)} copies\n\n")
                f.write(f"**SOPInstanceUID:** `{uid}`  \n\n")
                f.write("| # | File |\n")
                f.write("|---|------|\n")
                for i, path in enumerate(paths):
                    f.write(f"| {i+1} | `{path}` |\n")
                f.write("\n")
            if len(sorted_groups) > 10:
                f.write(f"*... and {len(sorted_groups) - 10} more groups — "
                        f"see `duplicate_sop.csv`*\n\n")

        # ── Check 4 detail ────────────────────────────────────────────────────
        f.write("## Check 4 — AccessionNumber → StudyInstanceUID Mismatch\n\n")
        f.write("> A healthy dataset maps each AccessionNumber to exactly one "
                "StudyInstanceUID. Multiple StudyInstanceUIDs for the same "
                "AccessionNumber indicates a data integrity issue — e.g. "
                "two different studies were assigned the same accession number, "
                "or a study was re-exported with a new StudyInstanceUID.\n\n")
        if n_mismatch_acc == 0:
            f.write("✅ Every AccessionNumber maps to exactly one StudyInstanceUID.\n\n")
        else:
            f.write(f"❌ **{n_mismatch_acc}** AccessionNumber(s) map to more than one "
                    f"StudyInstanceUID  \n")
            f.write(f"(**{n_mismatch_files}** files total involved).  \n")
            f.write("Full list: `accession_study_mismatch.csv`\n\n")

            sorted_mismatches = sorted(
                checks["mismatch_groups"].items(),
                key=lambda kv: len(kv[1]),
                reverse=True,
            )
            for idx, (acc, study_map) in enumerate(sorted_mismatches[:10], start=1):
                f.write(f"### Accession {idx}: `{acc}`  "
                        f"({len(study_map)} distinct StudyInstanceUIDs)\n\n")
                for study_uid, paths in sorted(study_map.items()):
                    f.write(f"**StudyInstanceUID:** `{study_uid}`  "
                            f"({len(paths)} file(s))\n\n")
                    f.write("| File |\n")
                    f.write("|------|\n")
                    for path in paths[:5]:
                        f.write(f"| `{path}` |\n")
                    if len(paths) > 5:
                        f.write(f"| *... and {len(paths)-5} more* |\n")
                    f.write("\n")
            if len(sorted_mismatches) > 10:
                f.write(f"*... and {len(sorted_mismatches) - 10} more accessions — "
                        f"see `accession_study_mismatch.csv`*\n\n")

        f.write("---\n\n")
        f.write("*Report generated by dicom_consistency_check.py*\n")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Files seen (all types)  : {total_seen}")
    print(f"  DICOM files identified  : {total_dicom}")
    print(f"  Successfully read       : {total_records}")
    print(f"{'='*60}")
    print(f"  [1] Unreadable          : {n_unreadable}")
    print(f"  [2] No AccessionNumber  : {n_no_acc}")
    print(f"  [3] Duplicate SOP UID   : {n_dup_uids} UID(s)  ({n_dup_files} files)")
    print(f"  [4] Acc/Study mismatch  : {n_mismatch_acc} accession(s)  ({n_mismatch_files} files)")
    print(f"{'='*60}")
    print(f"  Elapsed                 : {elapsed_sec:.1f} s")
    print(f"  Report written to       : {run_dir}")
    print(f"{'='*60}\n")


# =========================================================
# Orchestrator
# =========================================================
def run_consistency_check(
    input_root: Path,
    workers:    int,
) -> None:
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"consistency_{run_ts}"
    t_start = datetime.datetime.now()

    print(f"Input  : {input_root}")
    print(f"Logs   : {run_dir}")
    print(f"Workers: {workers}\n")

    # ── Discover leaf directories ──────────────────────────────────────────────
    print("Discovering leaf directories...")
    leaf_dirs = collect_leaf_dirs(input_root)
    if not leaf_dirs:
        print("No directories with files found. Exiting.")
        return

    n_workers = max(1, min(workers, len(leaf_dirs)))
    print(f"Found {len(leaf_dirs)} leaf directories — "
          f"scanning across {n_workers} worker(s).\n")

    # ── Submit one future per leaf directory ───────────────────────────────────
    worker_results: List[WorkerResult] = []

    agg_seen  = 0
    agg_dicom = 0

    with tqdm(
        total         = len(leaf_dirs),
        unit          = "dir",
        desc          = "Scanning",
        dynamic_ncols = True,
    ) as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_dir = {
                ex.submit(_scan_leaf_dirs, [str(d)]): str(d)
                for d in leaf_dirs
            }

            for fut in as_completed(future_to_dir):
                dir_str = future_to_dir[fut]
                try:
                    result = fut.result()
                    worker_results.append(result)
                    agg_seen  += result.total_seen
                    agg_dicom += result.total_dicom
                except Exception as e:
                    tqdm.write(
                        f"[SCAN ERROR] {Path(dir_str).name}: "
                        f"{type(e).__name__}: {e}"
                    )

                pbar.set_postfix(
                    seen  = agg_seen,
                    dicom = agg_dicom,
                )
                pbar.set_description(f"Scanning  [{Path(dir_str).name[:30]}]")
                pbar.update(1)

    # ── Merge ──────────────────────────────────────────────────────────────────
    print("\nMerging records and running consistency checks...")
    all_records, total_seen, total_dicom = _merge_results(worker_results)
    total_records = sum(1 for r in all_records if r.readable)

    # ── Run checks ────────────────────────────────────────────────────────────
    checks = run_checks(all_records)

    elapsed_sec = (datetime.datetime.now() - t_start).total_seconds()

    # ── Write reports ─────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_reports(
        run_dir        = run_dir,
        checks         = checks,
        total_seen     = total_seen,
        total_dicom    = total_dicom,
        total_records  = total_records,
        n_workers      = n_workers,
        input_root     = input_root,
        elapsed_sec    = elapsed_sec,
        run_ts         = run_ts,
    )


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "DICOM consistency check — header-only scan, no image extraction. "
            "Checks: unreadable files, missing AccessionNumber, "
            "duplicate SOPInstanceUID, AccessionNumber/StudyInstanceUID mismatch."
        )
    )
    parser.add_argument(
        "--input_root", type=Path, default=Path(INPUT_ROOT),
        help="Root directory to scan (default: INPUT_ROOT from config.py)",
    )
    parser.add_argument(
        "--workers", type=int,
        default=max(1, (os.cpu_count() or 8) - 1),
        help="Number of scan workers (default: cpu_count - 1)",
    )

    args = parser.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    run_consistency_check(
        input_root = args.input_root,
        workers    = args.workers,
    )