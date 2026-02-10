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
  /data/Deep_Angiography/DICOM_Sequence_Processed/consolidated_metadata.csv

Columns:
  StudyInstanceUID, AccessionNumber, Number of Instances, SOPInstanceUIDs
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from tqdm import tqdm  # ✅ added

BASE_DIR = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
OUT_CSV = Path("/data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_GT.csv")
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


def _extract_from_wide(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Extract (StudyInstanceUID, SOPInstanceUID, AccessionNumber) from wide format."""
    if STUDY_KEY not in df.columns or SOP_KEY not in df.columns:
        return []

    sub = df[[c for c in [STUDY_KEY, SOP_KEY, ACCESSION_KEY] if c in df.columns]].copy()

    for c in sub.columns:
        sub[c] = sub[c].map(_norm_str)

    sub = sub[(sub[STUDY_KEY] != "") & (sub[SOP_KEY] != "")]
    sub = sub.drop_duplicates()

    rows: List[Tuple[str, str, str]] = []
    for _, r in sub.iterrows():
        rows.append(
            (
                r[STUDY_KEY],
                r[SOP_KEY],
                r.get(ACCESSION_KEY, ""),
            )
        )
    return rows


def _guess_kv_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = [c.strip() for c in df.columns.astype(str)]
    lower = {c.lower(): c for c in cols}

    if "information" in lower and "value" in lower:
        return lower["information"], lower["value"]

    if len(cols) == 2:
        return cols[0], cols[1]

    return None


def _extract_from_rowwise_kv(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Extract (StudyInstanceUID, SOPInstanceUID, AccessionNumber) from row-wise KV."""
    kv = _guess_kv_columns(df)
    if not kv:
        return []

    key_col, val_col = kv
    tmp = df[[key_col, val_col]].copy()

    tmp[key_col] = tmp[key_col].map(_norm_str)
    tmp[val_col] = tmp[val_col].map(_norm_str)

    info = {}
    for k, v in tmp.itertuples(index=False, name=None):
        if k and v:
            info[k] = v

    study_uid = info.get(STUDY_KEY, "")
    sop_uid = info.get(SOP_KEY, "")
    accession = info.get(ACCESSION_KEY, "")

    if not study_uid or not sop_uid:
        return []

    return [(study_uid, sop_uid, accession)]


def load_uids_from_metadata_csv(csv_path: Path) -> List[Tuple[str, str, str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}", file=sys.stderr)
        return []

    rows = _extract_from_wide(df)
    if rows:
        return rows

    rows = _extract_from_rowwise_kv(df)
    if rows:
        return rows

    print(
        f"[WARN] Could not extract UIDs from {csv_path} (columns={list(df.columns)})",
        file=sys.stderr,
    )
    return []


def main() -> int:
    if not BASE_DIR.exists():
        print(f"[ERROR] Base directory does not exist: {BASE_DIR}", file=sys.stderr)
        return 2

    metadata_files = sorted(BASE_DIR.rglob(TARGET_NAME))

    study_to_sop: dict[str, set[str]] = defaultdict(set)
    study_to_accession: dict[str, set[str]] = defaultdict(set)

    used = 0
    for csv_path in tqdm(
        metadata_files,
        desc="Processing metadata.csv files",
        unit="file",
    ):
        rows = load_uids_from_metadata_csv(csv_path)
        if not rows:
            continue
        used += 1
        for study_uid, sop_uid, accession in rows:
            study_to_sop[study_uid].add(sop_uid)
            if accession:
                study_to_accession[study_uid].add(accession)

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

    print(f"[OK] Found {len(metadata_files)} metadata.csv files")
    print(f"[OK] Parsed {used} metadata.csv files with usable UIDs")
    print(f"[OK] Aggregated {len(study_to_sop)} studies")
    print(f"[OK] Wrote: {OUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
