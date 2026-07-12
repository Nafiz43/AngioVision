"""
Step 04 — Consolidate all per-sequence metadata.csv files into one hub CSV.

One row per StudyInstanceUID, with SOPInstanceUIDs ordered chronologically
by (SeriesDate, AcquisitionTime), falling back to ContentDate, then SOP UID.
SOPs with no StudyInstanceUID go to unmatched_SOPInstanceUIDs.csv.

After consolidating, triggers the 'Number of Instances' distribution
analysis in s02_stats_gen (the consolidated CSV only exists from this
point, but the analysis belongs to — and its outputs land under —
02_stats_gen/).
"""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from vdp import s02_stats_gen
from vdp.common import write_csv

KEYS = ("StudyInstanceUID", "SOPInstanceUID", "AccessionNumber",
        "SeriesDate", "AcquisitionTime", "ContentDate")

_TIME_DIGITS_RE = re.compile(r"^\d{2,6}(\.\d+)?$")


def _norm(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def _parse_date(x) -> Optional[int]:
    s = _norm(x)
    if not s:
        return None
    if len(s) == 8 and s.isdigit():
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
            return int(s)
    dt = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(dt) else int(dt.strftime("%Y%m%d"))


def _parse_time(x) -> Optional[int]:
    """Microseconds from midnight, or None."""
    s = _norm(x)
    if not s:
        return None
    if ":" in s:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return ((dt.hour * 3600 + dt.minute * 60 + dt.second) * 1_000_000
                + dt.microsecond)
    s2 = re.sub(r"[^0-9.]", "", s)
    if not s2 or not _TIME_DIGITS_RE.match(s2):
        return None
    main, _, frac = s2.partition(".")
    micros = int(frac[:6].ljust(6, "0")) if frac else 0
    main = {2: main + "0000", 4: main + "00", 6: main}.get(len(main))
    if main is None:
        return None
    hh, mm, ss = int(main[:2]), int(main[2:4]), int(main[4:6])
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return None
    return (hh * 3600 + mm * 60 + ss) * 1_000_000 + micros


def _parse_one_metadata_csv(csv_path_str: str) -> List[Dict[str, str]]:
    """Extract the key fields from one metadata.csv (wide or Information/Value)."""
    csv_path = Path(csv_path_str)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    # Wide format
    if "SOPInstanceUID" in df.columns:
        cols = [c for c in KEYS if c in df.columns]
        sub = df[cols].copy()
        for c in sub.columns:
            sub[c] = sub[c].map(_norm)
        sub = sub[sub["SOPInstanceUID"] != ""].drop_duplicates()
        return [
            {**{k: r.get(k, "") for k in KEYS}, "SourceMetadataPath": csv_path_str}
            for r in sub.to_dict("records")
        ]

    # Row-wise Information/Value format
    cols = [c.strip() for c in df.columns.astype(str)]
    lower = {c.lower(): c for c in cols}
    if "information" in lower and "value" in lower:
        key_col, val_col = lower["information"], lower["value"]
    elif len(cols) == 2:
        key_col, val_col = cols
    else:
        return []

    info = {}
    for k, v in df[[key_col, val_col]].itertuples(index=False, name=None):
        k, v = _norm(k), _norm(v)
        if k and v:
            info[k] = v
    # step 01 writes lowercase convenience keys too — accept either casing
    sop = info.get("SOPInstanceUID") or info.get("sop_instance_uid", "")
    if not sop:
        return []
    return [{
        "StudyInstanceUID": info.get("StudyInstanceUID", ""),
        "SOPInstanceUID": sop,
        "AccessionNumber": info.get("AccessionNumber")
                           or info.get("accession_number", ""),
        "SeriesDate": info.get("SeriesDate", ""),
        "AcquisitionTime": info.get("AcquisitionTime", ""),
        "ContentDate": info.get("ContentDate", ""),
        "SourceMetadataPath": csv_path_str,
    }]


def _sort_key(sd: Optional[int], at: Optional[int], cd: Optional[int]
              ) -> Tuple[int, int, int, int, int, int]:
    """Missing values sort AFTER present ones at each level."""
    return (
        sd if sd is not None else 99991231, 1 if sd is None else 0,
        at if at is not None else 99999999 * 1_000_000, 1 if at is None else 0,
        cd if cd is not None else 99991231, 1 if cd is None else 0,
    )


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / "04_consolidated_metadata"
    output_root = Path(cfg.dsa_sequences_root())  # potential-DSA subset (step 06)

    metadata_files = [str(p) for p in sorted(output_root.rglob("metadata.csv"))]

    study_to_accessions: Dict[str, set] = defaultdict(set)
    study_to_sops: Dict[str, Dict[str, Tuple]] = defaultdict(dict)
    unmatched: List[Dict] = []
    unmatched_seen = set()

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        it = ex.map(_parse_one_metadata_csv, metadata_files, chunksize=10)
        for entries in tqdm(it, total=len(metadata_files),
                            desc="[04] Consolidating", unit="file"):
            for e in entries:
                study, sop = _norm(e["StudyInstanceUID"]), _norm(e["SOPInstanceUID"])
                if not sop:
                    continue
                key = _sort_key(_parse_date(e["SeriesDate"]),
                                _parse_time(e["AcquisitionTime"]),
                                _parse_date(e["ContentDate"]))
                if study:
                    prev = study_to_sops[study].get(sop)
                    if prev is None or key < prev:
                        study_to_sops[study][sop] = key
                    acc = _norm(e["AccessionNumber"])
                    if acc:
                        study_to_accessions[study].add(acc)
                elif sop not in unmatched_seen:
                    unmatched_seen.add(sop)
                    unmatched.append({
                        "SOPInstanceUID": sop,
                        "AccessionNumber": _norm(e["AccessionNumber"]),
                        "SourceMetadataPath": e["SourceMetadataPath"],
                    })

    out_rows = []
    for study, sop_map in study_to_sops.items():
        ordered = [sop for sop, _ in sorted(sop_map.items(),
                                            key=lambda kv: (kv[1], kv[0]))]
        out_rows.append({
            "StudyInstanceUID": study,
            "AccessionNumber": ",".join(sorted(study_to_accessions.get(study, []))),
            "Number of Instances": len(ordered),
            "SOPInstanceUIDs": ",".join(ordered),
        })
    out_rows.sort(key=lambda r: r["Number of Instances"], reverse=True)

    consolidated_csv = step_dir / "consolidated_metadata_ALL_Sequences.csv"
    write_csv(consolidated_csv,
              ["StudyInstanceUID", "AccessionNumber",
               "Number of Instances", "SOPInstanceUIDs"], out_rows)
    write_csv(step_dir / "unmatched_SOPInstanceUIDs.csv",
              ["SOPInstanceUID", "AccessionNumber", "SourceMetadataPath"], unmatched)

    # Instances distribution — code + outputs live under 02_stats_gen
    instances_summary = {}
    if out_rows:
        instances_summary = s02_stats_gen.analyze_instances(
            cfg, run_dir, consolidated_csv
        )

    summary = {
        "metadata_files": len(metadata_files),
        "studies": len(out_rows),
        "unmatched_sops": len(unmatched),
        "consolidated_csv": str(consolidated_csv),
        **instances_summary,
    }
    print(f"[04] {summary}")
    return summary
