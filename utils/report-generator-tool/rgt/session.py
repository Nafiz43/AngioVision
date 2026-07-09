"""Interactive session flow: study/sequence selection tables and the
per-study report + Q&A loop."""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

import torch

from rgt.inference import answer_question, encode_study, generate_report
from rgt.model import PooledCLIP
from rgt.ui import (
    BOLD, CYAN, DIM, GREEN,
    banner, c, err, info, print_report_box, prompt, section, success, warn,
)


def display_study_table(studies: List[Dict]) -> None:
    hdr = f"  {'#':>3}  {'Accession':<22}  {'Seqs':>5}  {'Frames':>8}"
    print(c(hdr, BOLD))
    print(c("  " + "─" * (len(hdr) - 2), DIM))
    for i, s in enumerate(studies):
        frames = sum(si["n_frames"] for si in s["seq_info"])
        row = f"  {i+1:>3}  {s['acc']:<22}  {len(s['sop_uids']):>5}  {frames:>8}"
        print(c(row, CYAN if i % 2 == 0 else ""))
    print()


def pick_studies(studies: List[Dict]) -> List[Dict]:
    display_study_table(studies)
    while True:
        raw = prompt(
            "Select study number(s)  (e.g. 1  or  1,3  or  all)"
        ).strip().lower()
        if not raw:
            continue
        if raw == "all":
            return studies
        parts = re.split(r"[\s,]+", raw)
        sel, ok = [], True
        for p in parts:
            if not p.isdigit():
                err(f"'{p}' is not a number.")
                ok = False
                break
            idx = int(p) - 1
            if not (0 <= idx < len(studies)):
                err(f"Index {int(p)} out of range (1–{len(studies)}).")
                ok = False
                break
            sel.append(studies[idx])
        if ok and sel:
            return sel


def pick_sequences(study: Dict) -> Dict:
    si = study["seq_info"]
    if len(si) <= 1:
        return study

    print(f"\n  Study {c(study['acc'], BOLD)} has {len(si)} sequences:\n")
    print(c(f"    {'#':>3}  {'SOP (truncated)':<44}  {'Frames':>6}", BOLD))
    print(c("    " + "─" * 56, DIM))
    for i, s in enumerate(si):
        short = s["sop"][:42] + ".." if len(s["sop"]) > 42 else s["sop"]
        print(f"    {i+1:>3}  {short:<44}  {s['n_frames']:>6}")

    raw = prompt(
        "Use which sequences? (e.g. 1,2  or  all — default: all)"
    ).strip().lower()
    if not raw or raw == "all":
        return study
    keep = {
        int(p) - 1
        for p in re.split(r"[\s,]+", raw)
        if p.isdigit() and 1 <= int(p) <= len(si)
    }
    if not keep:
        warn("Invalid selection — using all sequences.")
        return study
    return {
        "acc":      study["acc"],
        "sop_uids": [study["sop_uids"][i] for i in sorted(keep)],
        "seq_info": [si[i] for i in sorted(keep)],
    }


def run_study_session(
    model:     PooledCLIP,
    study:     Dict,
    base_dir:  Path,
    processor,
    gen_tok,
    device:    torch.device,
    args,
) -> None:
    acc = study["acc"]
    section(f"Study: {acc}")

    # ── Encode visual tokens once; cache for the whole session ────────────────
    info("Encoding visual sequences …")
    result = encode_study(model, study, base_dir, processor, device, args)
    if result is None:
        warn(f"No valid frames found for {acc} — skipping.")
        return
    vtok, vmask = result
    n_seqs = sum(1 for s in study["seq_info"] if s["n_frames"] > 0)
    success(
        f"Encoded {n_seqs} sequence(s)  →  "
        f"visual token shape {tuple(vtok.shape)}"
    )

    # ── Generate initial report ───────────────────────────────────────────────
    info("Generating report …")
    report = generate_report(model, vtok, vmask, gen_tok, device, args)
    print_report_box("Generated Report", report)

    # ── Q&A loop ──────────────────────────────────────────────────────────────
    section("Q&A  —  your question → decoder answers using visual + report context")
    print(c("  Commands:", BOLD))
    print(c("    show   — re-print the full report", DIM))
    print(c("    regen  — regenerate the report from scratch", DIM))
    print(c("    back   — return to study selection", DIM))
    print(c("    quit   — exit the tool", DIM))
    print()

    while True:
        try:
            user_q = prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_q:
            continue

        cmd = user_q.lower()

        if cmd in ("back", "b"):
            break

        if cmd in ("quit", "exit", "q"):
            banner("Session ended  ◈  Thank you for using AngioVision")
            sys.exit(0)

        if cmd == "show":
            print_report_box("Generated Report", report)
            continue

        if cmd == "regen":
            info("Regenerating report …")
            report = generate_report(model, vtok, vmask, gen_tok, device, args)
            print_report_box("Regenerated Report", report)
            continue

        # Regular question
        info("Generating answer …")
        answer = answer_question(
            model, vtok, vmask, report, user_q, gen_tok, device, args
        )
        print()
        print(c("  Model:", BOLD + GREEN))
        for line in textwrap.wrap(answer or "(no answer generated)", width=74):
            print("  " + line)
        print()
