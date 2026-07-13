#!/usr/bin/env python3
"""
Step 00b — Slice cleaned reports down to visually-grounded sections.

Keeps only findings/impression/angiography/variant-anatomy/vessel-catheterized
sections and drops administrative + procedural boilerplate (personnel, consent,
sedation, plan, closure, ...). The cleaned-text column is overwritten in place
so downstream steps and training (--report_text_col) need no change; the
pre-slice text is preserved in <col>_unsliced.

Keep rules (case-insensitive on the section header):
  - header contains FINDINGS or IMPRESSION
  - VARIANT ANATOMY, ANGIOGRAPHIC ENDPOINT, VESSEL CATHETERIZED, PROCEDURE
  - contains ANGIOGRAPHY/AORTOGRAPHY unless it is an INDICATION FOR ... header
Fallback: if kept text < MIN_CHARS the full original is kept (counted in the
summary), so no report ever goes empty.

Also runnable standalone on any CSV:
    python -m tdp.s00b_slice_reports --csv /path/reports.csv --col cleaned_radrpt --apply
(dry run without --apply; --apply backs up the CSV to <csv>.bak_preslice first)
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

MIN_CHARS = 100
UNSLICED_SUFFIX = "_unsliced"

HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /\-()\x27]{1,60}):(.*)$")
# break single-paragraph reports: newline before inline ALL-CAPS headers
INLINE_HEADER_RE = re.compile(r"(?<!\n)(?=\b[A-Z][A-Z /\-()]{2,40}:)")

# boilerplate sentences that ride along inside kept narrative sections
BOILER = [
    re.compile(p, re.I) for p in (
        r"[^.]*maximal sterile barrier[^.]*\.\s*",
        r"[^.]*prepped and draped[^.]*\.\s*",
        r"[^.]*time.?out was performed[^.]*\.\s*",
        r"[^.]*informed consent[^.]*\.\s*",
    )
]


def keep_header(h: str) -> bool:
    h = h.upper().strip()
    if h.startswith("INDICATION"):
        return False
    if "FINDINGS" in h or "IMPRESSION" in h:
        return True
    if h in ("VARIANT ANATOMY", "ANGIOGRAPHIC ENDPOINT",
             "VESSEL CATHETERIZED", "PROCEDURE"):
        return True
    return "ANGIOGRAPHY" in h or "AORTOGRAPHY" in h


def slice_report(text) -> str:
    text = INLINE_HEADER_RE.sub("\n", str(text))
    kept, keeping = [], False
    for line in text.splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            keeping = keep_header(m.group(1))
        if keeping and line.strip():
            kept.append(line.rstrip())
    out = "\n".join(kept)
    for b in BOILER:
        out = b.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def slice_csv(csv_path: str, text_col: str, apply: bool = False) -> Dict:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise KeyError(f"column {text_col!r} not in {csv_path}")
    sliced = df[text_col].fillna("").map(slice_report)
    short = sliced.str.len() < MIN_CHARS
    summary = {
        "reports": len(df),
        "fallback_full_text": int(short.sum()),
        "median_chars_before": int(df[text_col].str.len().median()),
        "median_chars_after": int(sliced[~short].str.len().median()),
        "applied": apply,
    }
    if apply:
        shutil.copy(csv_path, str(csv_path) + ".bak_preslice")
        df[text_col + UNSLICED_SUFFIX] = df[text_col]
        df[text_col] = sliced.where(~short, df[text_col])
        df.to_csv(csv_path, index=False)
    return summary


def run(cfg, run_dir: Path, data_dir: Path) -> Dict:
    cleaned_csv = data_dir / "cleaned_reports.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"{cleaned_csv} not found — run step 00 (cleaning) first.")
    df = pd.read_csv(cleaned_csv, nrows=1)
    text_col = next((c for c in df.columns
                     if c.startswith("cleaned_")
                     and not c.endswith(UNSLICED_SUFFIX)), None)
    if text_col is None:
        raise KeyError(f"no cleaned_* column found in {cleaned_csv}")
    summary = slice_csv(str(cleaned_csv), text_col, apply=True)
    print(f"[00b] sliced {summary[reports]} reports "
          f"(median {summary[median_chars_before]} -> "
          f"{summary[median_chars_after]} chars, "
          f"{summary[fallback_full_text]} kept full text)")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", default="cleaned_radrpt")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run summary only)")
    a = ap.parse_args()
    print(slice_csv(a.csv, a.col, apply=a.apply))
