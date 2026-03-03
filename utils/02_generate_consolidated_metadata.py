#!/usr/bin/env python3
"""
consolidate_metadata.py

Recursively scans:
  /data/Deep_Angiography/DICOM_Sequence_Processed
for files named: metadata.csv

Supports TWO metadata.csv formats:
1) Wide format (columns):
     StudyInstanceUID, SOPInstanceUID, AccessionNumber,
     SeriesDate (optional), AcquisitionTime (optional), ContentDate (optional)
2) Row-wise key/value format (rows):
     Information, Value
     where Information ∈ {
        StudyInstanceUID, SOPInstanceUID, AccessionNumber,
        SeriesDate, AcquisitionTime, ContentDate
     }

Outputs:
  /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_ALL_Sequences.csv

Columns:
  StudyInstanceUID, AccessionNumber, Number of Instances, SOPInstanceUIDs

SORTING REQUIREMENT (UPDATED):
  SOPInstanceUIDs must be ordered by ascending (SeriesDate, AcquisitionTime) per study.
  - If SeriesDate is missing/unparseable => that SOP goes to the end.
  - If AcquisitionTime is missing/unparseable => that SOP goes after those with time (same date).
  - If both missing, falls back to ContentDate (if available), else goes to the end.
  - Final tie-breaker: SOPInstanceUID lexicographic.

Also outputs:
  <same-dir-as-out>/unmatched_SOPInstanceUIDs.csv

Definition of "unmatched":
  SOPInstanceUID is present but StudyInstanceUID is missing/blank.
"""

from __future__ import annotations

import os
import re
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

# Existing optional
CONTENTDATE_KEY = "ContentDate"

# NEW optional keys you mentioned
SERIESDATE_KEY = "SeriesDate"
ACQTIME_KEY = "AcquisitionTime"


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def _parse_date_to_int(x) -> Optional[int]:
    """
    Convert date to int YYYYMMDD for sorting.
    Accepts:
      - 'YYYYMMDD'
      - 'YYYY-MM-DD'
      - any pandas-parseable date string
    Returns None if missing/unparseable.
    """
    s = _norm_str(x)
    if not s:
        return None

    # Fast path: 8-digit date (DICOM style)
    if len(s) == 8 and s.isdigit():
        try:
            y = int(s[0:4])
            m = int(s[4:6])
            d = int(s[6:8])
            if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                return int(s)
        except Exception:
            pass

    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(dt):
        return None
    return int(dt.strftime("%Y%m%d"))


_TIME_DIGITS_RE = re.compile(r"^\d{2,6}(\.\d+)?$")


def _parse_time_to_int(x) -> Optional[int]:
    """
    Convert time to an integer for sorting.
    Supports common DICOM-ish times:
      - 'HHMMSS'
      - 'HHMMSS.ffffff'
      - 'HH:MM:SS'
      - 'HH:MM:SS.ffffff'
      - 'HHMM' (treated as HHMM00)
      - 'HH' (treated as HH0000)

    Returns microseconds-from-midnight as int, or None if missing/unparseable.
    """
    s = _norm_str(x)
    if not s:
        return None

    # Normalize separators
    s2 = s.strip()

    # Case 1: Contains ':' -> parse with pandas
    if ":" in s2:
        dt = pd.to_datetime(s2, errors="coerce")
        if pd.isna(dt):
            return None
        # dt is a Timestamp with a date; use time component
        return (
            int(dt.hour) * 3600 * 1_000_000
            + int(dt.minute) * 60 * 1_000_000
            + int(dt.second) * 1_000_000
            + int(dt.microsecond)
        )

    # Case 2: DICOM style digits, possibly fractional
    # Strip anything non-digit/non-dot (just in case)
    s2 = re.sub(r"[^0-9.]", "", s2)
    if not s2:
        return None

    # Allow HH / HHMM / HHMMSS / HHMMSS.ffffff
    if not _TIME_DIGITS_RE.match(s2):
        # last attempt: pandas
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return (
            int(dt.hour) * 3600 * 1_000_000
            + int(dt.minute) * 60 * 1_000_000
            + int(dt.second) * 1_000_000
            + int(dt.microsecond)
        )

    # Split fractional part
    if "." in s2:
        main, frac = s2.split(".", 1)
        frac = (frac[:6]).ljust(6, "0")  # microseconds
        try:
            micros = int(frac)
        except Exception:
            micros = 0
    else:
        main = s2
        micros = 0

    # Pad main to HHMMSS
    if len(main) == 2:      # HH
        main = main + "0000"
    elif len(main) == 4:    # HHMM
        main = main + "00"
    elif len(main) == 6:    # HHMMSS
        pass
    else:
        return None

    try:
        hh = int(main[0:2])
        mm = int(main[2:4])
        ss = int(main[4:6])
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            return None
        return hh * 3600 * 1_000_000 + mm * 60 * 1_000_000 + ss * 1_000_000 + micros
    except Exception:
        return None


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
    We also capture optional StudyInstanceUID, AccessionNumber, SeriesDate, AcquisitionTime, ContentDate.
    """
    if SOP_KEY not in df.columns:
        return []

    cols_to_take = [
        c for c in [
            STUDY_KEY,
            SOP_KEY,
            ACCESSION_KEY,
            SERIESDATE_KEY,
            ACQTIME_KEY,
            CONTENTDATE_KEY,
        ]
        if c in df.columns
    ]

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
                SERIESDATE_KEY: r.get(SERIESDATE_KEY, ""),
                ACQTIME_KEY: r.get(ACQTIME_KEY, ""),
                CONTENTDATE_KEY: r.get(CONTENTDATE_KEY, ""),
                "SourceMetadataPath": src_path,
            }
        )
    return out


def _extract_entries_from_rowwise_kv(df: pd.DataFrame, src_path: str) -> List[Dict[str, str]]:
    """
    Row-wise KV format: expect two columns like (Information, Value) or any 2-col file.
    We look for keys:
      StudyInstanceUID, SOPInstanceUID, AccessionNumber, SeriesDate, AcquisitionTime, ContentDate.
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
    series_date = info.get(SERIESDATE_KEY, "")
    acq_time = info.get(ACQTIME_KEY, "")
    content_date = info.get(CONTENTDATE_KEY, "")

    if not sop_uid:
        return []

    return [
        {
            STUDY_KEY: study_uid,
            SOP_KEY: sop_uid,
            ACCESSION_KEY: accession,
            SERIESDATE_KEY: series_date,
            ACQTIME_KEY: acq_time,
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

    # Per-study ordering info:
    # study -> sop -> (series_date_int, acq_time_int, content_date_int)
    study_to_sop_sortinfo: dict[str, dict[str, Tuple[Optional[int], Optional[int], Optional[int]]]] = defaultdict(dict)

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

                series_date_raw = _norm_str(e.get(SERIESDATE_KEY, ""))
                acq_time_raw = _norm_str(e.get(ACQTIME_KEY, ""))
                content_date_raw = _norm_str(e.get(CONTENTDATE_KEY, ""))

                series_date_int = _parse_date_to_int(series_date_raw)
                acq_time_int = _parse_time_to_int(acq_time_raw)
                content_date_int = _parse_date_to_int(content_date_raw)

                src_path = e.get("SourceMetadataPath", "")

                if sop_uid:
                    total_sop_seen += 1

                if study_uid and sop_uid:
                    matched_sop += 1

                    prev = study_to_sop_sortinfo[study_uid].get(sop_uid)
                    new = (series_date_int, acq_time_int, content_date_int)

                    # If SOP appears multiple times: keep the "earliest" sort tuple.
                    # We compare using the same ordering rules we later use for sorting.
                    if prev is None:
                        study_to_sop_sortinfo[study_uid][sop_uid] = new
                    else:
                        def _cmp_key(t: Tuple[Optional[int], Optional[int], Optional[int]]) -> Tuple[int, int, int, int, int, int]:
                            sd, at, cd = t
                            sd_sentinel = 99991231
                            at_sentinel = 99999999 * 1_000_000  # big microsecond sentinel
                            cd_sentinel = 99991231
                            sd_val = sd if sd is not None else sd_sentinel
                            at_val = at if at is not None else at_sentinel
                            cd_val = cd if cd is not None else cd_sentinel
                            sd_miss = 1 if sd is None else 0
                            at_miss = 1 if at is None else 0
                            cd_miss = 1 if cd is None else 0
                            return (sd_val, sd_miss, at_val, at_miss, cd_val, cd_miss)

                        if _cmp_key(new) < _cmp_key(prev):
                            study_to_sop_sortinfo[study_uid][sop_uid] = new

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
                                "SeriesDate": series_date_raw,
                                "AcquisitionTime": acq_time_raw,
                                "ContentDate": content_date_raw,
                                "SourceMetadataPath": src_path,
                            }
                        )

    # Build consolidated output with SOPs ordered by (SeriesDate, AcquisitionTime)
    out_rows = []
    for study_uid in study_to_sop_sortinfo.keys():
        sop_to_info = study_to_sop_sortinfo[study_uid]

        def _sop_sort_key(item: Tuple[str, Tuple[Optional[int], Optional[int], Optional[int]]]) -> Tuple[int, int, int, int, int, int, str]:
            sop, (sd, at, cd) = item

            sd_sentinel = 99991231
            at_sentinel = 99999999 * 1_000_000
            cd_sentinel = 99991231

            sd_val = sd if sd is not None else sd_sentinel
            at_val = at if at is not None else at_sentinel
            cd_val = cd if cd is not None else cd_sentinel

            sd_miss = 1 if sd is None else 0
            at_miss = 1 if at is None else 0
            cd_miss = 1 if cd is None else 0

            # Order: SeriesDate, AcquisitionTime, ContentDate (fallback), SOP
            return (sd_val, sd_miss, at_val, at_miss, cd_val, cd_miss, sop)

        ordered_sops = [sop for sop, _info in sorted(sop_to_info.items(), key=_sop_sort_key)]
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

    # Global sorting: most instances first
    if not out_df.empty and "Number of Instances" in out_df.columns:
        out_df = out_df.sort_values(by="Number of Instances", ascending=False)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    unmatched_csv = OUT_CSV.parent / "unmatched_SOPInstanceUIDs.csv"
    unmatched_df = pd.DataFrame(
        unmatched_rows,
        columns=["SOPInstanceUID", "AccessionNumber", "SeriesDate", "AcquisitionTime", "ContentDate", "SourceMetadataPath"],
    )
    unmatched_df.to_csv(unmatched_csv, index=False)

    print(f"[OK] Found {len(metadata_files)} metadata.csv files")
    print(f"[OK] Parsed {parsed_files_with_entries} metadata.csv files with usable entries")
    print(f"[OK] Aggregated {len(study_to_sop_sortinfo)} studies")
    print(f"[OK] Wrote: {OUT_CSV}")

    print(f"[INFO] Total SOPInstanceUIDs seen: {total_sop_seen}")
    print(f"[INFO] Matched to studies: {matched_sop}")
    print(f"[WARN] Unmatched SOPInstanceUIDs: {unmatched_sop}")
    print(f"[OK] Wrote: {unmatched_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())