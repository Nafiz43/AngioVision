"""
Step 00 — DICOM consistency check (read-only, header-only).

Pre-flight checks over the raw input tree (run() — pipeline position 00):
    1. Unreadable files
    2. Missing AccessionNumber
    3. Duplicate SOPInstanceUID
    4. AccessionNumber mapping to >1 StudyInstanceUID

Post-QA checks (run_post() — invoked after the other steps finish, because
their inputs are produced later in the run; outputs still land under
00_consistency_check/):
    5. Frame-count outliers: copies mosaic.png + metadata.csv of sequences
       with frame counts outside [outlier_low_thresh, outlier_high_thresh]
       into outliers/ for reviewer eyeballing
       (from 06_outlier_identification.py; needs steps 02 + 03)
    6. No-visual accessions: for accessions the reports list expected but
       the pipeline never produced (step 05's missing list), rescans the
       raw tree to tell "DICOM exists but has no pixel data" apart from
       "no DICOM at all" (from 06_finding_accessions_with_no_visual.py)
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pydicom
from tqdm import tqdm

from vdp.common import collect_leaf_dirs, get_tag_str, is_probably_dicom, write_csv


def _scan_leaf_dir(leaf_str: str) -> List[Dict]:
    records = []
    for f in Path(leaf_str).iterdir():
        if not f.is_file() or not is_probably_dicom(f):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception as e:
            records.append({"file": str(f), "readable": False,
                            "accession": "", "sop": "", "study": "",
                            "error": f"{type(e).__name__}: {e}"})
            continue
        nof = get_tag_str(ds, "NumberOfFrames")
        try:
            n_frames = int(nof) if nof else 1  # absent => single-frame instance
        except (ValueError, TypeError):
            n_frames = 1
        records.append({
            "file": str(f), "readable": True,
            "accession": get_tag_str(ds, "AccessionNumber"),
            "sop": get_tag_str(ds, "SOPInstanceUID"),
            "study": get_tag_str(ds, "StudyInstanceUID"),
            "num_frames": n_frames,
            "error": "",
        })
        del ds
    return records


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / "00_consistency_check"
    input_root = Path(cfg.input_root)

    leaf_dirs = collect_leaf_dirs(input_root)
    records: List[Dict] = []
    with tqdm(total=len(leaf_dirs), unit="dir", desc="[00] Scanning") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [ex.submit(_scan_leaf_dir, str(d)) for d in leaf_dirs]
            for fut in as_completed(futures):
                records.extend(fut.result())
                pbar.update(1)

    unreadable = [r for r in records if not r["readable"]]
    no_accession = [r for r in records if r["readable"] and not r["accession"].strip()]

    uid_to_paths: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r["readable"] and r["sop"]:
            uid_to_paths[r["sop"]].append(r["file"])
    duplicate_rows = [
        {"sop_instance_uid": uid, "copy_index": i + 1,
         "total_copies": len(paths), "file": p}
        for uid, paths in sorted(uid_to_paths.items()) if len(paths) > 1
        for i, p in enumerate(sorted(paths))
    ]

    acc_to_studies: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["readable"] and r["accession"].strip() and r["study"]:
            acc_to_studies[r["accession"]][r["study"]].append(r["file"])
    mismatch_rows = [
        {"accession_number": acc, "study_instance_uid": study,
         "n_studies_for_acc": len(study_map), "file": p}
        for acc, study_map in sorted(acc_to_studies.items()) if len(study_map) > 1
        for study, paths in sorted(study_map.items())
        for p in sorted(paths)
    ]

    write_csv(step_dir / "unreadable.csv", ["file", "error"], unreadable)
    write_csv(step_dir / "no_accession.csv", ["file", "sop"], no_accession)
    write_csv(step_dir / "duplicate_sop.csv",
              ["sop_instance_uid", "copy_index", "total_copies", "file"], duplicate_rows)
    write_csv(step_dir / "accession_study_mismatch.csv",
              ["accession_number", "study_instance_uid", "n_studies_for_acc", "file"],
              mismatch_rows)

    n_dup_uids = len({r["sop_instance_uid"] for r in duplicate_rows})
    n_mismatch = len({r["accession_number"] for r in mismatch_rows})
    total_frames = sum(r.get("num_frames", 0) for r in records if r["readable"])
    summary = {
        "dicom_files": len(records),
        "total_frames": total_frames,
        "unreadable": len(unreadable),
        "missing_accession": len(no_accession),
        "duplicate_sop_uids": n_dup_uids,
        "accession_study_mismatches": n_mismatch,
        "healthy": not (unreadable or no_accession or n_dup_uids or n_mismatch),
    }
    print(f"[00] {summary}")
    return summary


# =========================================================
# Post-QA check 5 — frame-count outliers (06_outlier_identification.py)
# =========================================================
def _check_outliers(cfg, run_dir: Path, step_dir: Path) -> Dict:
    frame_stats_csv = run_dir / "02_stats_gen" / "frame_statistics.csv"
    if not frame_stats_csv.exists():
        return {"skipped": "frame_statistics.csv not found (run step 02 first)"}

    out_dir = step_dir / "outliers"
    out_dir.mkdir(parents=True, exist_ok=True)
    low, high = cfg.outlier_low_thresh, cfg.outlier_high_thresh

    copied = in_range = missing_mosaic = 0
    with frame_stats_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                n_frames = int(row["number_of_frames"])
            except (KeyError, ValueError):
                continue
            if low <= n_frames <= high:
                in_range += 1
                continue

            seq_dir = Path(row["frames_dir_path"].strip()).parent
            mosaic_src = seq_dir / "mosaic.png"
            if not mosaic_src.exists():
                missing_mosaic += 1
                continue

            dest = out_dir / f"{n_frames}.png"
            if dest.exists():  # frame-count collision → disambiguate
                dest = out_dir / (f"{n_frames}_{row.get('outer_dir_name', 'x')}"
                                  f"_{row.get('inner_dir_name', 'x')[:8]}.png")
            shutil.copy2(mosaic_src, dest)

            metadata_src = seq_dir / "metadata.csv"
            if metadata_src.exists():
                shutil.copy2(metadata_src, dest.with_suffix(".csv"))
            copied += 1

    return {"outliers_copied": copied, "in_range": in_range,
            "missing_mosaic": missing_mosaic,
            "thresholds": f"<{low} or >{high}"}


# =========================================================
# Post-QA check 6 — no-visual accessions
# (06_finding_accessions_with_no_visual.py)
# =========================================================
def _scan_leaf_for_accessions(leaf_str: str, targets: frozenset) -> List[Dict]:
    hits = []
    for f in Path(leaf_str).iterdir():
        if not f.is_file() or not is_probably_dicom(f):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        acc = get_tag_str(ds, "AccessionNumber")
        if not acc or acc not in targets:
            del ds
            continue
        rows = int(getattr(ds, "Rows", 0) or 0)
        cols = int(getattr(ds, "Columns", 0) or 0)
        has_visual = hasattr(ds, "PixelData") or (rows > 0 and cols > 0)
        hits.append({"accession": acc, "file": str(f), "visual": has_visual})
        del ds
    return hits


def _check_no_visual(cfg, run_dir: Path, step_dir: Path) -> Dict:
    missing_list_csv = (run_dir / "05_accession_check"
                        / "missing_accession_list_only.csv")
    if not missing_list_csv.exists():
        return {"skipped": "missing-accession list not found (run step 05 first)"}

    with missing_list_csv.open(newline="", encoding="utf-8") as f:
        targets = frozenset(
            row["AccessionNumber_missing"].strip()
            for row in csv.DictReader(f)
            if row.get("AccessionNumber_missing", "").strip()
        )
    if not targets:
        return {"no_visual": 0, "has_visual": 0,
                "note": "no missing accessions to check"}

    leaf_dirs = collect_leaf_dirs(Path(cfg.input_root))
    acc_files: Dict[str, List[str]] = defaultdict(list)
    acc_has_visual: Dict[str, bool] = defaultdict(bool)

    with tqdm(total=len(leaf_dirs), unit="dir", desc="[00-post] No-visual scan") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [ex.submit(_scan_leaf_for_accessions, str(d), targets)
                       for d in leaf_dirs]
            for fut in as_completed(futures):
                for hit in fut.result():
                    acc_files[hit["accession"]].append(hit["file"])
                    if hit["visual"]:
                        acc_has_visual[hit["accession"]] = True
                pbar.update(1)

    found = set(acc_files)
    not_found = targets - found
    found_no_visual = {a for a in found if not acc_has_visual[a]}
    truly_no_visual = not_found | found_no_visual
    has_visual = found - found_no_visual

    def _rows(accessions):
        return [{"AccessionNumber": a, "found_in_dicom_dir": a in found,
                 "has_visual_data": acc_has_visual.get(a, False)}
                for a in sorted(accessions)]

    write_csv(step_dir / "missing_accessions_no_visual.csv",
              ["AccessionNumber", "found_in_dicom_dir", "has_visual_data"],
              _rows(truly_no_visual))
    write_csv(step_dir / "missing_accessions_has_visual.csv",
              ["AccessionNumber", "found_in_dicom_dir", "has_visual_data"],
              _rows(has_visual))

    return {"target_accessions": len(targets),
            "not_found_in_raw": len(not_found),
            "found_but_no_visual": len(found_no_visual),
            "has_visual": len(has_visual)}


def run_post(cfg, run_dir: Path) -> Dict:
    """Post-QA phase — call after steps 02/03/05 have produced their outputs."""
    step_dir = run_dir / "00_consistency_check"
    summary = {
        "outliers": _check_outliers(cfg, run_dir, step_dir),
        "no_visual": _check_no_visual(cfg, run_dir, step_dir),
    }
    print(f"[00-post] {summary}")
    return summary
