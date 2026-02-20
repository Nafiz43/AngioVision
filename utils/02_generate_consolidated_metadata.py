#!/usr/bin/env python3
"""
consolidate_metadata.py

Recursively scans:
  /data/Deep_Angiography/DICOM_Sequence_Processed
for files named: metadata.csv

Supports TWO metadata.csv formats:
1) Wide format (columns):
     StudyInstanceUID, SOPInstanceUID, AccessionNumber, ContentDate (optional)
2) Row-wise key/value format (rows):
     Information, Value
     where Information ∈ {StudyInstanceUID, SOPInstanceUID, AccessionNumber, ContentDate}

Outputs:
  /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv

Columns:
  StudyInstanceUID, AccessionNumber, Number of Instances, SOPInstanceUIDs

NEW REQUIREMENT:
  SOPInstanceUIDs must be ordered by ascending ContentDate per study.
  If ContentDate is missing/unparseable for a SOP, it is placed at the end (ties broken by SOPInstanceUID).

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
CONTENTDATE_KEY = "ContentDate"


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def _parse_content_date_to_int(x) -> Optional[int]:
    """
    Convert ContentDate to an int YYYYMMDD for sorting.
    Accepts common representations:
      - 'YYYYMMDD'
      - 'YYYY-MM-DD'
      - timestamps parseable by pandas
    Returns None if missing/unparseable.
    """
    s = _norm_str(x)
    if not s:
        return None

    # Fast path: 8-digit date
    if len(s) == 8 and s.isdigit():
        try:
            y = int(s[0:4])
            m = int(s[4:6])
            d = int(s[6:8])
            if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                return int(s)
        except Exception:
            pass

    # General parse
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(dt):
        return None
    return int(dt.strftime("%Y%m%d"))


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
    Wide format: expect SOPInstanceUID column at minimum.
    We also capture optional StudyInstanceUID, AccessionNumber, ContentDate.
    """
    if SOP_KEY not in df.columns:
        return []

    cols_to_take = [c for c in [STUDY_KEY, SOP_KEY, ACCESSION_KEY, CONTENTDATE_KEY] if c in df.columns]
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
                CONTENTDATE_KEY: r.get(CONTENTDATE_KEY, ""),
                "SourceMetadataPath": src_path,
            }
        )
    return out


def _extract_entries_from_rowwise_kv(df: pd.DataFrame, src_path: str) -> List[Dict[str, str]]:
    """
    Row-wise KV format: expect two columns like (Information, Value) or any 2-col file.
    We look for keys StudyInstanceUID, SOPInstanceUID, AccessionNumber, ContentDate.
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
    content_date = info.get(CONTENTDATE_KEY, "")

    if not sop_uid:
        return []

    return [
        {
            STUDY_KEY: study_uid,
            SOP_KEY: sop_uid,
            ACCESSION_KEY: accession,
            CONTENTDATE_KEY: content_date,
            "SourceMetadataPath": src_path,
        }
    ]


def parse_one_metadata_csv(csv_path: Path) -> List[Dict[str, str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}", file=sys.stderr)
        return []

    src_path = str(csv_path)

    wide_entries = _extract_entries_from_wide(df, src_path)
    if wide_entries:
        return wide_entries

    kv_entries = _extract_entries_from_rowwise_kv(df, src_path)
    if kv_entries:
        return kv_entries

    print(
        f"[WARN] Could not parse usable fields from {csv_path} (columns={list(df.columns)})",
        file=sys.stderr,
    )
    return []


def _worker_parse_one(csv_path_str: str) -> List[Dict[str, str]]:
    return parse_one_metadata_csv(Path(csv_path_str))


def main() -> int:
    if not BASE_DIR.exists():
        print(f"[ERROR] Base directory does not exist: {BASE_DIR}", file=sys.stderr)
        return 2

    metadata_files = sorted(BASE_DIR.rglob(TARGET_NAME))

    # For counting + accession aggregation
    study_to_accession: dict[str, set[str]] = defaultdict(set)

    # NEW: per-study SOP ordering by ContentDate
    # study -> sop -> best_date_int (min date when multiple seen)
    study_to_sop_date: dict[str, dict[str, Optional[int]]] = defaultdict(dict)

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
        for entries in tqdm(it, total=len(file_strs), desc="Processing metadata.csv files", unit="file"):
            if not entries:
                continue
            parsed_files_with_entries += 1

            for e in entries:
                study_uid = _norm_str(e.get(STUDY_KEY, ""))
                sop_uid = _norm_str(e.get(SOP_KEY, ""))
                accession = _norm_str(e.get(ACCESSION_KEY, ""))
                content_date_raw = _norm_str(e.get(CONTENTDATE_KEY, ""))
                content_date_int = _parse_content_date_to_int(content_date_raw)
                src_path = e.get("SourceMetadataPath", "")

                if sop_uid:
                    total_sop_seen += 1

                if study_uid and sop_uid:
                    matched_sop += 1

                    # Track best (earliest) content date per SOP within study
                    prev = study_to_sop_date[study_uid].get(sop_uid)
                    if prev is None:
                        # if prev is None, prefer an actual date if we have it
                        study_to_sop_date[study_uid][sop_uid] = content_date_int
                    else:
                        # prev is an int; keep the minimum
                        if content_date_int is not None and content_date_int < prev:
                            study_to_sop_date[study_uid][sop_uid] = content_date_int

                    # If we haven't stored this SOP yet and we have a date, store it
                    if sop_uid not in study_to_sop_date[study_uid]:
                        study_to_sop_date[study_uid][sop_uid] = content_date_int

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
                                "ContentDate": content_date_raw,
                                "SourceMetadataPath": src_path,
                            }
                        )

    # Build consolidated output with SOPs ordered by ContentDate ascending
    out_rows = []
    for study_uid in study_to_sop_date.keys():
        sop_to_date = study_to_sop_date[study_uid]

        # Sort key:
        #   1) content_date_int ascending (None goes last)
        #   2) SOPInstanceUID lexicographically as tie-breaker
        def _sop_sort_key(item: Tuple[str, Optional[int]]) -> Tuple[int, int, str]:
            sop, dt_int = item
            # Put missing dates last: use a big sentinel
            sentinel = 99991231
            dt_val = dt_int if dt_int is not None else sentinel
            missing_flag = 1 if dt_int is None else 0  # ensures real dates come before missing for same sentinel
            return (dt_val, missing_flag, sop)

        ordered_sops = [sop for sop, _ in sorted(sop_to_date.items(), key=_sop_sort_key)]
        accession_list = sorted(study_to_accession.get(study_uid, []))

        out_rows.append(
            {
                "StudyInstanceUID": study_uid,
                "AccessionNumber": ",".join(accession_list),
                "Number of Instances": len(ordered_sops),
                "SOPInstanceUIDs": ",".join(ordered_sops),
            }
        )

    out_df = pd.DataFrame(out_rows)

    # Keep your previous global sorting: most instances first
    if not out_df.empty and "Number of Instances" in out_df.columns:
        out_df = out_df.sort_values(by="Number of Instances", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    unmatched_csv = OUT_CSV.parent / "unmatched_SOPInstanceUIDs.csv"
    unmatched_df = pd.DataFrame(
        unmatched_rows,
        columns=["SOPInstanceUID", "AccessionNumber", "ContentDate", "SourceMetadataPath"],
    )
    unmatched_df.to_csv(unmatched_csv, index=False)

    print(f"[OK] Found {len(metadata_files)} metadata.csv files")
    print(f"[OK] Parsed {parsed_files_with_entries} metadata.csv files with usable entries")
    print(f"[OK] Aggregated {len(study_to_sop_date)} studies")
    print(f"[OK] Wrote: {OUT_CSV}")

    print(f"[INFO] Total SOPInstanceUIDs seen: {total_sop_seen}")
    print(f"[INFO] Matched to studies: {matched_sop}")
    print(f"[WARN] Unmatched SOPInstanceUIDs: {unmatched_sop}")
    print(f"[OK] Wrote: {unmatched_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())