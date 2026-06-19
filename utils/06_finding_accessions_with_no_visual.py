#!/usr/bin/env python3
"""
AngioVision — Find Missing Accessions with No Visual DICOM Data
================================================================
Reads accession numbers from:
    /data/Deep_Angiography/DICOM-metadata-stats/all_accession_with_no_sequence.csv
Recursively scans:
    /data/Deep_Angiography/DICOM/
For each DICOM file, reads AccessionNumber from metadata.
Cross-references with the missing list, then checks whether any matched
file has actual pixel data (Rows/Columns > 0 and PixelData present).

Output:
    missing_accessions_no_visual.csv  — accessions with NO visual data at all
    missing_accessions_has_visual.csv — accessions that DO have at least one frame
"""

import os
import csv
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CSV_INPUT   = "/data/Deep_Angiography/DICOM-metadata-stats/all_accession_with_no_sequence.csv"
DICOM_ROOT  = "/data/Deep_Angiography/DICOM"
OUT_DIR     = "/data/Deep_Angiography/DICOM-metadata-stats"
OUT_NO_VIS  = "missing_accessions_no_visual.csv"
OUT_HAS_VIS = "missing_accessions_has_visual.csv"
MAX_WORKERS = max(1, os.cpu_count() - 2)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_missing_accessions(csv_path: str) -> set:
    """Load the set of accession numbers to look for."""
    accessions = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("AccessionNumber_missing", "").strip()
            if val:
                accessions.add(val)
    log.info(f"Loaded {len(accessions):,} missing accession numbers from CSV.")
    return accessions


def collect_dicom_files(root: str) -> list:
    """Walk the DICOM root and collect all file paths."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            paths.append(os.path.join(dirpath, fname))
    log.info(f"Found {len(paths):,} files under {root}")
    return paths


def has_visual_data(ds) -> bool:
    """
    Return True if the DICOM dataset contains meaningful pixel/image data.
    Checks:
      - PixelData tag (7FE0,0010) is present
      - Rows and Columns are both > 0
    """
    if not hasattr(ds, "PixelData"):
        return False
    rows = getattr(ds, "Rows", 0) or 0
    cols = getattr(ds, "Columns", 0) or 0
    return int(rows) > 0 and int(cols) > 0


# Module-level global set in each worker via initializer
_target_accessions: set = set()


def _worker_init(accessions: set):
    """Called once per worker process — avoids pickling the set for every task."""
    global _target_accessions
    _target_accessions = accessions


def inspect_file(fpath: str) -> tuple | None:
    """
    Worker function. Returns:
        (accession_number: str, file_path: str, visual: bool)
    or None if the file is not DICOM or accession not in target set.
    """
    try:
        ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=False)
        acc = str(getattr(ds, "AccessionNumber", "") or "").strip()
        if not acc or acc not in _target_accessions:
            return None

        rows = int(getattr(ds, "Rows", 0) or 0)
        cols = int(getattr(ds, "Columns", 0) or 0)
        pixel_tag_present = hasattr(ds, "PixelData") or (rows > 0 and cols > 0)

        return (acc, fpath, pixel_tag_present)

    except InvalidDicomError:
        return None
    except (OSError, ValueError, RuntimeError) as exc:
        log.debug(f"Could not read DICOM {fpath}: {exc}")
        return None


def main(args):
    # 1. Load targets
    target_accessions = load_missing_accessions(args.csv_input)

    # 2. Collect files
    all_files = collect_dicom_files(args.dicom_root)

    # 3. Parallel scan
    log.info(f"Scanning with {args.workers} workers …")
    # acc → True if ANY file has visual data
    acc_has_visual: dict[str, bool] = defaultdict(lambda: False)
    # acc → list of matched file paths (for debugging)
    acc_files: dict[str, list] = defaultdict(list)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(target_accessions,),
    ) as pool:
        futures = {pool.submit(inspect_file, f): f for f in all_files}
        with tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scanning DICOM files",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for fut in pbar:
                result = fut.result()
                if result is None:
                    continue
                acc, fpath, visual = result
                acc_files[acc].append(fpath)
                if visual:
                    acc_has_visual[acc] = True
                pbar.set_postfix(matched=len(acc_files), refresh=False)

    # 4. Partition
    found_accessions   = set(acc_files.keys())
    not_found          = target_accessions - found_accessions   # no DICOM file at all
    found_no_visual    = {a for a in found_accessions if not acc_has_visual[a]}
    found_has_visual   = {a for a in found_accessions if acc_has_visual[a]}

    # "truly no visual" = not found anywhere + found but zero pixel data
    truly_no_visual    = not_found | found_no_visual

    log.info(f"\n{'='*55}")
    log.info(f"Target accessions          : {len(target_accessions):>8,}")
    log.info(f"Found in DICOM dir         : {len(found_accessions):>8,}")
    log.info(f"  → has visual pixel data  : {len(found_has_visual):>8,}")
    log.info(f"  → no visual pixel data   : {len(found_no_visual):>8,}")
    log.info(f"Not found in DICOM at all  : {len(not_found):>8,}")
    log.info(f"TOTAL with NO visual data  : {len(truly_no_visual):>8,}")
    log.info(f"{'='*55}\n")

    # 5. Write outputs
    os.makedirs(args.out_dir, exist_ok=True)

    no_vis_path  = os.path.join(args.out_dir, OUT_NO_VIS)
    has_vis_path = os.path.join(args.out_dir, OUT_HAS_VIS)

    def write_csv(path, accessions, label):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["AccessionNumber", "found_in_dicom_dir", "has_visual_data"])
            for acc in sorted(accessions):
                found = acc in found_accessions
                visual = acc_has_visual.get(acc, False)
                writer.writerow([acc, found, visual])
        log.info(f"Wrote {len(accessions):,} {label} → {path}")

    write_csv(no_vis_path,  truly_no_visual,  "no-visual accessions")
    write_csv(has_vis_path, found_has_visual, "has-visual accessions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find missing accessions with no visual DICOM data.")
    parser.add_argument("--csv-input",  default=CSV_INPUT,  help="Path to input CSV")
    parser.add_argument("--dicom-root", default=DICOM_ROOT, help="Root DICOM directory")
    parser.add_argument("--out-dir",    default=OUT_DIR,    help="Output directory")
    parser.add_argument("--workers",    type=int, default=MAX_WORKERS, help=f"Number of parallel workers (default: {MAX_WORKERS})")
    args = parser.parse_args()
    main(args)