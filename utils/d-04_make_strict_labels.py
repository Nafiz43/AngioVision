#!/usr/bin/env python3
"""
make_strict_labels.py

- Drops rows where question == "Which artery is catheterized?"
- Normalizes the `answer` column to strict labels: yes / no / n/a
- Uses ONLY string matching (no AI).

Usage:
  python make_strict_labels.py \
    --in_csv /data/Deep_Angiography/DICOM_Sequence_Processed_Output/mosaics_extracted_labels_clip.csv \
    --out_csv /data/Deep_Angiography/DICOM_Sequence_Processed_Output/mosaics_extracted_labels_clip_strict.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


DROP_QUESTION_EXACT = "Which artery is catheterized?"


def _clean_text(s: str) -> str:
    """Lowercase, strip, remove surrounding quotes, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).strip()
    # Remove surrounding quotes if present
    if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_strict_label(raw_answer: str) -> str:
    """
    Map raw answer text -> one of: "yes", "no", "n/a"
    using deterministic string/regex matching.
    """
    s = _clean_text(raw_answer)

    if s == "" or s in {"na", "n/a", "n\\a", "none", "null"}:
        return "n/a"

    # Strong "n/a" / unknown / unclear / not stated signals (check early)
    na_patterns = [
        r"\bnot stated\b",
        r"\bnot mentioned\b",
        r"\bunknown\b",
        r"\bunclear\b",
        r"\bcan'?t tell\b",
        r"\bcannot tell\b",
        r"\bunable to (determine|assess|tell)\b",
        r"\bindeterminate\b",
        r"\bequivocal\b",
        r"\bnot visible\b",
        r"\bpoor (opacification|visualization|image quality)\b",
        r"\blimited (opacification|visualization|evaluation)\b",
    ]
    if any(re.search(p, s) for p in na_patterns):
        return "n/a"

    # Strong NO signals (check before YES to catch "no evidence of ..." etc.)
    no_patterns = [
        r"^\s*no\s*$",
        r"^\s*n\s*$",
        r"\bno evidence of\b",
        r"\bwithout\b",
        r"\babsent\b",
        r"\bnot present\b",
        r"\bnot seen\b",
        r"\bno (stenosis|dissection|extravasation|hemorrhage|stent)\b",
        r"\bnegative\b",
        r"\bnone\b",
        r"\bdenies\b",
    ]
    if any(re.search(p, s) for p in no_patterns):
        return "no"

    # Strong YES signals
    yes_patterns = [
        r"^\s*yes\s*$",
        r"^\s*y\s*$",
        r"\bpresent\b",
        r"\bseen\b",
        r"\bvisible\b",
        r"\bevidence of\b",
        r"\bconsistent with\b",
        r"\bpositive\b",
        r"\bidentified\b",
        r"\bnoted\b",
    ]
    if any(re.search(p, s) for p in yes_patterns):
        return "yes"

    # Catch common phrases like "no stenosis visible" (NO) or "stent visible" (YES)
    # If "no" appears close to key terms, treat as NO.
    if re.search(r"\bno\b", s) and re.search(r"\b(stenosis|dissection|extravasation|hemorrhage|stent|variant anatomy)\b", s):
        return "no"
    # If key terms appear with "visible/seen/present" and no explicit negation, treat as YES.
    if re.search(r"\b(stenosis|dissection|extravasation|hemorrhage|stent|variant anatomy)\b", s) and re.search(r"\b(visible|seen|present)\b", s) and not re.search(r"\b(no|not)\b", s):
        return "yes"

    # Default: if it doesn't confidently match, mark n/a
    return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="/data/Deep_Angiography/DICOM_Sequence_Processed_Output/mosaics_extracted_labels_clip.csv",
        help="Input CSV path",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Output CSV path (default: input name + _strict.csv)",
    )
    ap.add_argument(
        "--question_col",
        default="question",
        help="Column name for question (case-insensitive match supported)",
    )
    ap.add_argument(
        "--answer_col",
        default="answer",
        help="Column name for answer (case-insensitive match supported)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_path = Path(args.out_csv) if args.out_csv else in_path.with_name(in_path.stem + "_strict.csv")

    df = pd.read_csv(in_path)

    # Resolve column names case-insensitively
    cols_lower = {c.lower(): c for c in df.columns}
    if args.question_col.lower() not in cols_lower:
        raise KeyError(f"Question column '{args.question_col}' not found. Available: {list(df.columns)}")
    if args.answer_col.lower() not in cols_lower:
        raise KeyError(f"Answer column '{args.answer_col}' not found. Available: {list(df.columns)}")

    qcol = cols_lower[args.question_col.lower()]
    acol = cols_lower[args.answer_col.lower()]

    # Drop rows with the exact question
    df[qcol] = df[qcol].astype(str)
    df = df[df[qcol].str.strip() != DROP_QUESTION_EXACT].copy()

    # Strict-label the answer column
    df[acol] = df[acol].apply(to_strict_label)

    # Safety check: enforce only allowed values
    allowed = {"yes", "no", "n/a"}
    bad = set(df[acol].dropna().unique()) - allowed
    if bad:
        raise ValueError(f"Found non-strict labels after processing: {bad}")

    df.to_csv(out_path, index=False)
    print(f"Saved strict-labeled CSV to: {out_path}")
    print("Label counts:")
    print(df[acol].value_counts(dropna=False))


if __name__ == "__main__":
    main()

# Saved strict-labeled CSV to: /data/Deep_Angiography/DICOM_Sequence_Processed_Output/mosaics_extracted_labels_clip_strict.csv
# Label counts:
# answer
# yes    14095
# n/a    10953
# no      1646
# Name: count, dtype: int64