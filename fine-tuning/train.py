#!/usr/bin/env python3
"""
train.py - unified AngioVision contrastive fine-tuning entrypoint.

Replaces custom_framework_train_2 / _skip_frames / _temporal / _with_generation
and siglip.py.  Select the ablation via flags:

    # CLIP, no temporal
    python3 train.py --arch clip   --temporal_mode none        <data flags>
    # CLIP, with temporal
    python3 train.py --arch clip   --temporal_mode sinusoidal  <data flags>
    # SigLIP, no temporal
    python3 train.py --arch siglip --temporal_mode none        <data flags>
    # SigLIP, with temporal
    python3 train.py --arch siglip --temporal_mode sinusoidal  <data flags>
    # SigLIP2 (sigmoid loss + SigLIP2 vision tower by default)
    python3 train.py --arch siglip2 <data flags>
    # X-CLIP (softmax loss + video tower: frames encoded jointly with
    # cross-frame attention; temporal PE redundant -> none)
    python3 train.py --arch xclip --temporal_mode none <data flags>

Per-epoch validation: add ``--val_fraction 0.1`` (validation loss on a held-out
study split) and ``--epoch_qa_eval`` (full QA metrics incl. ALL_YES / ALL_NO /
RANDOM / FLIPPED baselines) - both are logged per epoch to
``<run_dir>/<run_name>_epoch_metrics.csv``, written incrementally.

One-command per-architecture pipelines: ``./train_clip.sh``,
``./train_siglip.sh``, ``./train_siglip2.sh``, ``./train_xclip.sh``
(see README_pipeline.md §5).

Run ``python3 train.py --help`` for the full hyper-parameter surface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Ensure the package next to this script is importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from angio_ft.cli import build_train_argparser  # noqa: E402

# Flags that exist ONLY in the unified CLI (the archived generation trainer does
# not understand them). They must be stripped before delegating to it.
_UNIFIED_ONLY_VALUED = {"--arch", "--io_threads", "--loss_flush_every", "--cache_clear_interval",
                        "--seed", "--run_name", "--val_fraction", "--validation_csv"}
_UNIFIED_ONLY_FLAGS = {"--run_post_pipeline", "--enable_generation",
                       "--freeze_vision_proj", "--freeze_text_proj",
                       "--resume", "--force", "--contrastive_accum", "--epoch_qa_eval"}


def _resolve_generation_trainer() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "legacy" / "custom_framework_train_with_generation.py",
        here / "custom_framework_train_with_generation.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "[ERROR] Could not locate custom_framework_train_with_generation.py in "
        f"{[str(c) for c in candidates]}."
    )


def _forward_argv_for_generation(argv: List[str]) -> List[str]:
    """Drop unified-only flags so the archived trainer's parser accepts argv."""
    out: List[str] = []
    dropped: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        key = tok.split("=", 1)[0]
        if key in _UNIFIED_ONLY_FLAGS:
            if key != "--enable_generation":
                dropped.append(key)
            i += 1
            continue
        if key in _UNIFIED_ONLY_VALUED:
            dropped.append(key)
            i += 1 if "=" in tok else 2  # skip flag (and its separate value)
            continue
        out.append(tok)
        i += 1
    if dropped:
        print(f"[WARN] Unified-only flags not supported by the generation trainer, dropped: {dropped}")
    # The archived trainer needs the flag itself to turn generation on.
    out.append("--enable_generation")
    return out


def _delegate_to_generation_trainer() -> None:
    script = _resolve_generation_trainer()
    fwd = _forward_argv_for_generation(sys.argv[1:])
    print("[INFO] --enable_generation set: delegating to the archived generation trainer.")
    print(f"[INFO] Trainer : {script}")
    print(f"[INFO] Command : {sys.executable} {script} {' '.join(fwd)}")
    os.execv(sys.executable, [sys.executable, str(script)] + fwd)


def main() -> None:
    parser = build_train_argparser()
    # Peek without failing on generation-specific flags (they are unknown to the
    # unified parser and land in `extra`).
    pre_args, _extra = parser.parse_known_args()

    if getattr(pre_args, "enable_generation", False):
        if pre_args.arch != "clip":
            raise SystemExit(
                "[ERROR] --enable_generation is only supported with --arch clip "
                f"(got --arch {pre_args.arch}). The archived generation trainer is CLIP-only."
            )
        _delegate_to_generation_trainer()
        return

    # Strict parse for the core (no-generation) path.
    args = parser.parse_args()
    # Imported after arg parsing so `--help` does not require torch/transformers.
    from angio_ft.engine import train
    train(args)


if __name__ == "__main__":
    main()
