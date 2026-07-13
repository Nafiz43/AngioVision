#!/usr/bin/env python3
"""Slice cleaned_radrpt down to visually-grounded sections (findings/impression/
angiography), dropping administrative and procedural boilerplate.

Overwrites cleaned_radrpt in place; original text is preserved in
cleaned_radrpt_unsliced. A CSV backup is written before modification.

Keep rules (case-insensitive on the section header):
  - header contains FINDINGS or IMPRESSION
  - VARIANT ANATOMY, ANGIOGRAPHIC ENDPOINT, VESSEL CATHETERIZED, PROCEDURE (title line)
  - contains ANGIOGRAPHY/AORTOGRAPHY unless it is an INDICATION FOR ... header
Fallback: if kept text < MIN_CHARS the full original is kept (flagged), so the
training pipeline never sees an empty report.
"""
import re, shutil, sys
import pandas as pd

CSV = "/data/Deep_Angiography/Reports/Report_List_v01_01_cleaned.csv"
COL = "cleaned_radrpt"
RAW_COL = "cleaned_radrpt_unsliced"
MIN_CHARS = 100
DRY = "--apply" not in sys.argv

HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /\-()\x27]{1,60}):(.*)$")

def keep_header(h):
    h = h.upper().strip()
    if h.startswith("INDICATION"):
        return False
    if any(k in h for k in ("FINDINGS", "IMPRESSION")):
        return True
    if h in ("VARIANT ANATOMY", "ANGIOGRAPHIC ENDPOINT", "VESSEL CATHETERIZED", "PROCEDURE"):
        return True
    if "ANGIOGRAPHY" in h or "AORTOGRAPHY" in h:
        return True
    return False

# boilerplate sentences that ride along inside kept narrative sections
BOILER = [
    re.compile(p, re.I) for p in (
        r"[^.]*maximal sterile barrier[^.]*\.\s*",
        r"[^.]*prepped and draped[^.]*\.\s*",
        r"[^.]*time.?out was performed[^.]*\.\s*",
        r"[^.]*informed consent[^.]*\.\s*",
    )
]

def slice_report(text):
    # break single-paragraph reports: newline before inline ALL-CAPS headers
    text = re.sub(r"(?<!\n)(?=\b[A-Z][A-Z /\-()]{2,40}:)", "\n", str(text))
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
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

df = pd.read_csv(CSV)
sliced = df[COL].fillna("").map(slice_report)
short = sliced.str.len() < MIN_CHARS
n = len(df)
print(f"reports: {n}  fallback (kept<{MIN_CHARS} chars): {short.sum()} ({100*short.mean():.1f}%)")
print(f"chars orig mean/median: {int(df[COL].str.len().mean())}/{int(df[COL].str.len().median())}")
ok = sliced[~short]
print(f"chars sliced mean/median (non-fallback): {int(ok.str.len().mean())}/{int(ok.str.len().median())}")

if DRY:
    for i in list(ok.sample(3, random_state=0).index):
        print("=" * 30, "sample row", i)
        print(sliced[i][:1200])
    print("\nDRY RUN — rerun with --apply to write.")
else:
    shutil.copy(CSV, CSV + ".bak_preslice")
    df[RAW_COL] = df[COL]
    df[COL] = sliced.where(~short, df[COL])
    df.to_csv(CSV, index=False)
    print(f"written: {CSV}  (backup: {CSV}.bak_preslice; original text in {RAW_COL})")
