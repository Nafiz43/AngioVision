#!/usr/bin/env python3
"""
filter_slr.py
-------------
Remove degenerate records from BOTH:
  - stage2_results.jsonl
  - claim2024_results.jsonl

The two files are linked by _md_file (same key in both).
A record is degenerate if it matches the stage2 signal OR the claim2024 signal.

stage2 degenerate: title="untitled" AND year=null AND ≥80% null/empty leaves
claim2024 degenerate: title="untitled" AND year=null AND all C01–C12 adherence="No"

A _md_file flagged degenerate in EITHER file is removed from BOTH files.
"""

import json, sys, shutil
from pathlib import Path
from datetime import datetime

BASE   = Path("/data/Deep_Angiography/AngioVision/slr/results")
STAGE2 = BASE / "stage2_results.jsonl"
CLAIM  = BASE / "claim2024_results.jsonl"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def count_leaves(obj):
    """Returns (total_leaves, null_or_empty_leaves)."""
    if isinstance(obj, dict):
        t, e = 0, 0
        for v in obj.values():
            ct, ce = count_leaves(v)
            t += ct; e += ce
        return t, e
    elif isinstance(obj, list):
        if not obj:
            return 1, 1
        t, e = 0, 0
        for v in obj:
            ct, ce = count_leaves(v)
            t += ct; e += ce
        return t, e
    else:
        is_empty = obj is None or obj in ("other", "unknown")
        return 1, int(is_empty)


def stage2_degenerate(rec: dict) -> tuple[bool, list[str]]:
    """Flag based on stage2 field structure."""
    reasons = []
    title = (rec.get("title") or "").strip().lower()
    if title == "untitled":
        reasons.append("title='untitled'")
    if rec.get("year") is None:
        reasons.append("year=null")
    if safe_get(rec, "study_identity", "publication_type") == "other":
        reasons.append("publication_type='other'")
    if (safe_get(rec, "imaging", "modality") == "other" and
            safe_get(rec, "imaging", "anatomy") == "other"):
        reasons.append("imaging modality+anatomy='other'")
    if safe_get(rec, "task", "primary_task") == "other":
        reasons.append("primary_task='other'")
    if (safe_get(rec, "method", "input_type") == "other" and
            safe_get(rec, "method", "training_supervision") == "other"):
        reasons.append("method input_type+supervision='other'")
    metrics = safe_get(rec, "evaluation", "metrics", default=[])
    if isinstance(metrics, list) and not metrics:
        reasons.append("evaluation.metrics=[]")
    total, empty = count_leaves(rec)
    if total > 0:
        ratio = empty / total
        if ratio >= 0.80:
            reasons.append(f"null/empty ratio={ratio:.0%} (≥80%)")

    has_title = "title='untitled'" in reasons
    has_year  = "year=null"        in reasons
    has_ratio = any("null/empty ratio" in r for r in reasons)
    return (has_title and has_year and has_ratio), reasons


def claim_degenerate(rec: dict) -> tuple[bool, list[str]]:
    """
    Flag if title='untitled', year=null, AND every CLAIM item adherence='No'.
    Also catches the degenerate case via the same null-density check as stage2
    (title + year + ≥80% null/empty).
    """
    reasons = []
    title = (rec.get("title") or "").strip().lower()
    if title == "untitled":
        reasons.append("title='untitled'")
    if rec.get("year") is None:
        reasons.append("year=null")

    adherence_block = rec.get("claim2024_adherence") or {}
    all_no = bool(adherence_block) and all(
        (v.get("adherence") or "").strip().upper() == "NO"
        for v in adherence_block.values()
        if isinstance(v, dict)
    )
    if all_no:
        reasons.append(f"all {len(adherence_block)} CLAIM items adherence='No'")

    # Fallback: density check (handles variants)
    total, empty = count_leaves(rec)
    if total > 0:
        ratio = empty / total
        if ratio >= 0.80:
            reasons.append(f"null/empty ratio={ratio:.0%} (≥80%)")

    has_title = "title='untitled'" in reasons
    has_year  = "year=null"        in reasons
    has_signal = all_no or any("null/empty ratio" in r for r in reasons)
    return (has_title and has_year and has_signal), reasons


# ── load + scan ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path):
    """Returns list of (raw_line, parsed_dict_or_None)."""
    records = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                records.append((lineno, line, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path.name} line {lineno}: JSON error – {e}; keeping")
                records.append((lineno, line, None))
    return records


def scan_file(records, detect_fn) -> tuple[set, list]:
    """Returns (bad_md_files set, removal_log list)."""
    bad, log = set(), []
    for lineno, raw, rec in records:
        if rec is None:
            continue
        flag, reasons = detect_fn(rec)
        if flag:
            md = rec.get("_md_file", "?")
            src = rec.get("_source_file", "?")
            bad.add(md)
            log.append({"lineno": lineno, "src": src, "md": md, "reasons": reasons})
    return bad, log


# ── write filtered output ────────────────────────────────────────────────────

def write_filtered(path: Path, records, bad_mds: set):
    backup = path.with_suffix(f".backup_{TS}.jsonl")
    shutil.copy2(path, backup)
    print(f"  Backup : {backup}")

    out_path = path.with_name(path.stem + ".filtered.jsonl")
    kept, removed_count = [], 0
    for _lineno, raw, rec in records:
        if rec is None:
            kept.append(raw)
            continue
        md = rec.get("_md_file", "?")
        if md in bad_mds:
            removed_count += 1
        else:
            kept.append(json.dumps(rec, ensure_ascii=False))

    with out_path.open("w") as fh:
        fh.write("\n".join(kept))
        if kept:
            fh.write("\n")

    total = len([r for r in records if r[2] is not None])
    print(f"  Input  : {total}  →  Kept: {len(kept)}  Removed: {removed_count}")
    print(f"  Output : {out_path}")
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    for p in (STAGE2, CLAIM):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("Loading files …")
    stage2_recs = load_jsonl(STAGE2)
    claim_recs  = load_jsonl(CLAIM)

    print("\nScanning for degenerate records …")
    bad_stage2, log_stage2 = scan_file(stage2_recs, stage2_degenerate)
    bad_claim,  log_claim  = scan_file(claim_recs,  claim_degenerate)

    # Union: remove any _md_file flagged in either file from both files
    bad_all = bad_stage2 | bad_claim

    print(f"\n  Flagged by stage2  : {len(bad_stage2)} md-files")
    print(f"  Flagged by claim   : {len(bad_claim)}  md-files")
    print(f"  Union (removed)    : {len(bad_all)}  md-files")

    # ── detailed removal log ─────────────────────────────────────────────────
    all_logs = {}
    for entry in log_stage2 + log_claim:
        md = entry["md"]
        all_logs.setdefault(md, {"src": entry["src"], "reasons": []})
        all_logs[md]["reasons"].extend(entry["reasons"])

    print(f"\n{'─'*65}")
    print("Degenerate records to be removed from BOTH files:")
    for md, info in sorted(all_logs.items()):
        print(f"  md={md:<14s} src={info['src']:<12s}")
        for r in dict.fromkeys(info["reasons"]):   # deduplicate, preserve order
            print(f"    → {r}")

    # ── write outputs ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Writing {STAGE2.name} …")
    write_filtered(STAGE2, stage2_recs, bad_all)

    print(f"\nWriting {CLAIM.name} …")
    write_filtered(CLAIM, claim_recs, bad_all)

    print(f"\n{'─'*65}")
    print("Done. Review the .filtered.jsonl files, then rename to replace originals:")
    print(f"  mv {STAGE2.with_name(STAGE2.stem + '.filtered.jsonl')} {STAGE2}")
    print(f"  mv {CLAIM.with_name(CLAIM.stem + '.filtered.jsonl')} {CLAIM}")


if __name__ == "__main__":
    main()