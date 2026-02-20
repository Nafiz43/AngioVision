#!/usr/bin/env python3
"""
consolidate_metadata.py

Recursively scans:
  /data/Deep_Angiography/DICOM_Sequence_Processed
for files named: metadata.csv

Supports TWO metadata.csv formats:
1) Wide format (columns):
     StudyInstanceUID, SOPInstanceUID, AccessionNumber
2) Row-wise key/value format (rows):
     Information, Value
     where Information ∈ {StudyInstanceUID, SOPInstanceUID, AccessionNumber}

Outputs:
  /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv

Columns:
  StudyInstanceUID, AccessionNumber, Number of Instances, SOPInstanceUIDs

Also outputs:
  <same-dir-as-out>/unmatched_SOPInstanceUIDs.csv

Definition of "unmatched":
  SOPInstanceUID is present but StudyInstanceUID is missing/blank.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
OUT_CSV = Path("/data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv")
TARGET_NAME = "metadata.csv"

STUDY_KEY = "StudyInstanceUID"
SOP_KEY = "SOPInstanceUID"
ACCESSION_KEY = "AccessionNumber"


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def _guess_kv_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = [c.strip() for c in df.columns.astype(str)]
    lower = {c.lower(): c for c in cols}

    if "information" in lower and "value" in lower:
        return lower["information"], lower["value"]

    if len(cols) == 2:
        return cols[0], cols[1]

    return None


def _extract_entries_from_wide(df: pd.DataFrame, src_path: str) -> List[Dict[str, str]]:
    """
    Extract entries from wide-format metadata.

    Returns list of dict entries with:
      - StudyInstanceUID (may be "")
      - SOPInstanceUID (must be non-empty to be included)
      - AccessionNumber (may be "")
      - SourceMetadataPath
    """
    if SOP_KEY not in df.columns:
        return []

    cols_to_take = [c for c in [STUDY_KEY, SOP_KEY, ACCESSION_KEY] if c in df.columns]
    sub = df[cols_to_take].copy()

    for c in sub.columns:
        sub[c] = sub[c].map(_norm_str)

    # Keep anything that has SOP (even if Study is missing) so we can detect unmatched SOPs
    sub = sub[sub[SOP_KEY] != ""]
    sub = sub.drop_duplicates()

    out: List[Dict[str, str]] = []
    for _, r in sub.iterrows():
        out.append(
            {
                STUDY_KEY: r.get(STUDY_KEY, ""),
                SOP_KEY: r.get(SOP_KEY, ""),
                ACCESSION_KEY: r.get(ACCESSION_KEY, ""),
                "SourceMetadataPath": src_path,
            }
        )
    return out


def _extract_entries_from_rowwise_kv(df: pd.DataFrame, src_path: str) -> List[Dict[str, str]]:
    """
    Extract entries from row-wise key/value metadata.

    Returns list with one dict entry (if SOP is present), allowing missing Study to surface.
    """
    kv = _guess_kv_columns(df)
    if not kv:
        return []

    key_col, val_col = kv
    tmp = df[[key_col, val_col]].copy()

    tmp[key_col] = tmp[key_col].map(_norm_str)
    tmp[val_col] = tmp[val_col].map(_norm_str)

    info: Dict[str, str] = {}
    for k, v in tmp.itertuples(index=False, name=None):
        if k and v:
            info[k] = v

    study_uid = info.get(STUDY_KEY, "")
    sop_uid = info.get(SOP_KEY, "")
    accession = info.get(ACCESSION_KEY, "")

    # If SOP isn't present, nothing to validate for "unmatched SOP"
    if not sop_uid:
        return []

    return [
        {
            STUDY_KEY: study_uid,
            SOP_KEY: sop_uid,
            ACCESSION_KEY: accession,
            "SourceMetadataPath": src_path,
        }
    ]


def parse_one_metadata_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Parse a single metadata.csv and return a list of entries.
    Never raises; returns [] on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}", file=sys.stderr)
        return []

    src_path = str(csv_path)

    # Try wide first (common case)
    wide_entries = _extract_entries_from_wide(df, src_path)
    if wide_entries:
        return wide_entries

    # Fallback to KV format
    kv_entries = _extract_entries_from_rowwise_kv(df, src_path)
    if kv_entries:
        return kv_entries

    print(
        f"[WARN] Could not parse usable fields from {csv_path} (columns={list(df.columns)})",
        file=sys.stderr,
    )
    return []


def _worker_parse_one(csv_path_str: str) -> List[Dict[str, str]]:
    # Keep only picklable args/returns for ProcessPool
    return parse_one_metadata_csv(Path(csv_path_str))


def main() -> int:
    if not BASE_DIR.exists():
        print(f"[ERROR] Base directory does not exist: {BASE_DIR}", file=sys.stderr)
        return 2

    metadata_files = sorted(BASE_DIR.rglob(TARGET_NAME))
    if not metadata_files:
        print(f"[WARN] No {TARGET_NAME} files found under: {BASE_DIR}", file=sys.stderr)

    # Aggregation in main process
    study_to_sop: dict[str, set[str]] = defaultdict(set)
    study_to_accession: dict[str, set[str]] = defaultdict(set)

    # Unmatched SOPs (SOP present but Study missing)
    unmatched_rows: List[Dict[str, str]] = []
    unmatched_sop_set: set[str] = set()

    total_sop_seen = 0
    matched_sop = 0
    unmatched_sop = 0
    parsed_files_with_entries = 0

    max_workers = min(32, (os.cpu_count() or 4))
    file_strs = [str(p) for p in metadata_files]
    chunksize = 50 if len(file_strs) >= 1000 else 10

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_worker_parse_one, file_strs, chunksize=chunksize)
        for entries in tqdm(
            it,
            total=len(file_strs),
            desc="Processing metadata.csv files",
            unit="file",
        ):
            if not entries:
                continue
            parsed_files_with_entries += 1

            for e in entries:
                study_uid = _norm_str(e.get(STUDY_KEY, ""))
                sop_uid = _norm_str(e.get(SOP_KEY, ""))
                accession = _norm_str(e.get(ACCESSION_KEY, ""))
                src_path = e.get("SourceMetadataPath", "")

                if sop_uid:
                    total_sop_seen += 1

                if study_uid and sop_uid:
                    matched_sop += 1
                    study_to_sop[study_uid].add(sop_uid)
                    if accession:
                        study_to_accession[study_uid].add(accession)
                elif sop_uid and not study_uid:
                    unmatched_sop += 1
                    if sop_uid not in unmatched_sop_set:
                        unmatched_sop_set.add(sop_uid)
                        unmatched_rows.append(
                            {
                                "SOPInstanceUID": sop_uid,
                                "AccessionNumber": accession,
                                "SourceMetadataPath": src_path,
                            }
                        )

    # Build consolidated output
    out_rows = []
    for study_uid in sorted(study_to_sop.keys()):
        sop_list = sorted(study_to_sop[study_uid])
        accession_list = sorted(study_to_accession.get(study_uid, []))
        out_rows.append(
            {
                "StudyInstanceUID": study_uid,
                "AccessionNumber": ",".join(accession_list),
                "Number of Instances": len(sop_list),
                "SOPInstanceUIDs": ",".join(sop_list),
            }
        )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "StudyInstanceUID",
            "AccessionNumber",
            "Number of Instances",
            "SOPInstanceUIDs",
        ],
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # Write unmatched SOP CSV (same directory as OUT_CSV)
    unmatched_csv = OUT_CSV.parent / "unmatched_SOPInstanceUIDs.csv"
    unmatched_df = pd.DataFrame(
        unmatched_rows,
        columns=["SOPInstanceUID", "AccessionNumber", "SourceMetadataPath"],
    )
    unmatched_df.to_csv(unmatched_csv, index=False)

    # Print summary
    print(f"[OK] Found {len(metadata_files)} metadata.csv files")
    print(f"[OK] Parsed {parsed_files_with_entries} metadata.csv files with usable entries")
    print(f"[OK] Aggregated {len(study_to_sop)} studies")
    print(f"[OK] Wrote: {OUT_CSV}")

    print(f"[INFO] Total SOPInstanceUIDs seen: {total_sop_seen}")
    print(f"[INFO] Matched to studies: {matched_sop}")
    print(f"[WARN] Unmatched SOPInstanceUIDs (SOP present but Study missing): {unmatched_sop}")
    print(f"[OK] Unique unmatched SOPInstanceUIDs written: {len(unmatched_rows)}")
    print(f"[OK] Wrote: {unmatched_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())