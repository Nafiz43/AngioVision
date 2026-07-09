#!/usr/bin/env python3
"""
AngioVision Report Generator Tool
=================================

Interactive CLI for AngioVision — no external API required.
Everything is powered by your fine-tuned checkpoint.

Workflow:
  1. Load the trained PooledCLIP checkpoint.
  2. List all studies in the holdout CSV; user picks one or more.
  3. Encode the selected sequences → visual tokens (kept in memory).
  4. Generate the initial free-form report via the decoder.
  5. Enter a Q&A loop:
       Each question is wrapped as:
           "Report: <generated report>
            Q: <user question>
            A:"
       and fed to the decoder with the SAME visual cross-attention tokens,
       so every answer is grounded in both the images and the prior report.

Usage:
    python run_tool.py \\
        --checkpoint /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/gen/500_16_16_32/last.pt \\
        [--decoder_model_name microsoft/biogpt | gpt2] \\
        [--vit_name google/vit-base-patch16-224-in21k] \\
        [--bert_name dmis-lab/biobert-base-cased-v1.1] \\
        [--embed_dim 256] \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from rgt.data import get_vit_processor, load_holdout_studies  # noqa: E402
from rgt.model import POOL_CHOICES, load_model  # noqa: E402
from rgt.session import pick_sequences, pick_studies, run_study_session  # noqa: E402
from rgt.ui import banner, err, info, prompt, section, success, warn  # noqa: E402


def build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="AngioVision Interactive CLI (fine-tuned model only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Checkpoint ────────────────────────────────────────────────────────────
    ap.add_argument(
        "--checkpoint",
        default=(
            "/data/Deep_Angiography/AngioVision/fine-tuning/"
            "checkpoints/500_16_16_32/last.pt"
        ),
        help="Path to trained .pt checkpoint",
    )
    # ── Architecture (matches training command exactly) ───────────────────────
    ap.add_argument("--vit_name",  default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--bert_name", default="UCSD-VA-health/RadBERT-RoBERTa-4m")
    ap.add_argument("--decoder_model_name", default="gpt2")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pooling",   default="logsumexp", choices=list(POOL_CHOICES))
    # ── Holdout data ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--holdout_meta_csv",
        default=(
            "/data/Deep_Angiography/Validation_Data/"
            "Validation_Data_2026_03_23/consolidated_metadata_ALL_Sequences.csv"
        ),
    )
    ap.add_argument(
        "--holdout_base_frames_dir",
        default=(
            "/data/Deep_Angiography/Validation_Data/"
            "Validation_Data_2026_03_23/DICOM_Sequence_Processed"
        ),
    )
    ap.add_argument("--anon_col", default="Anon Acc #")
    ap.add_argument("--sop_col",  default="SOPInstanceUIDs")
    # ── Inference knobs ───────────────────────────────────────────────────────
    ap.add_argument("--device",                  default=None,
                    help="cuda or cpu (auto-detect if omitted)")
    ap.add_argument("--frame_chunk_size",        type=int,   default=32)
    ap.add_argument("--max_new_tokens",          type=int,   default=256,
                    help="Max tokens for the initial report")
    ap.add_argument("--qa_max_new_tokens",       type=int,   default=128,
                    help="Max tokens for each Q&A answer")
    ap.add_argument("--max_frames_per_sequence", type=int,   default=32)
    ap.add_argument("--vit_image_size",          type=int,   default=None)
    ap.add_argument("--frame_temporal_scale",    type=float, default=0.75)
    ap.add_argument("--seq_temporal_scale",      type=float, default=0.5)
    ap.add_argument("--do_sample",               action="store_true")
    ap.add_argument("--top_p",                   type=float, default=1.0)
    ap.add_argument("--temperature",             type=float, default=1.0)
    ap.add_argument("--repetition_penalty",      type=float, default=1.5,
                    help=">1.0 penalises repeated tokens — fixes EAT/EAT loops")
    ap.add_argument("--no_repeat_ngram_size",    type=int,   default=4,
                    help="Forbid repeating any n-gram of this size")
    return ap


def main() -> None:
    args = build_args().parse_args()
    banner("AngioVision Interactive  ◈  Powered by your fine-tuned model")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args, device)

    # ── Tokenisers & processor ────────────────────────────────────────────────
    section("Loading tokenisers & image processor")
    processor = get_vit_processor(args.vit_name)
    gen_tok = AutoTokenizer.from_pretrained(args.decoder_model_name)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token or "<|pad|>"
    success("Ready")

    # ── Holdout metadata ──────────────────────────────────────────────────────
    section("Loading holdout studies")
    meta_csv = Path(args.holdout_meta_csv)
    base_dir = Path(args.holdout_base_frames_dir)
    if not meta_csv.exists():
        err(f"Holdout CSV not found: {meta_csv}")
        sys.exit(1)
    if not base_dir.exists():
        warn(f"Frames dir not found: {base_dir}  (frame counts will show 0)")

    studies = load_holdout_studies(meta_csv, args.anon_col, args.sop_col, base_dir)
    if not studies:
        err("No studies found in holdout CSV.")
        sys.exit(1)
    success(f"Found {len(studies)} holdout studies")

    # ── Main interaction loop ─────────────────────────────────────────────────
    while True:
        section("Study Selection")
        selected = pick_studies(studies)

        for raw_study in selected:
            refined = pick_sequences(raw_study)
            run_study_session(
                model, refined, base_dir, processor, gen_tok, device, args
            )

        again = prompt("Analyse more studies? (y / n)").strip().lower()
        if again not in ("y", "yes"):
            break

    banner("Session ended  ◈  Thank you for using AngioVision")


if __name__ == "__main__":
    main()
