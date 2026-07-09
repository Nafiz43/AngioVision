#!/usr/bin/env python3
"""
validate.py - unified AngioVision binary-QA validation entrypoint.

Replaces custom_framework_validate.py / custom_framework_validate_temporal.py.
Evaluates a single checkpoint file OR a whole run directory (epoch_*.pt / last.pt)
and copies the best-by-ORIGINAL-accuracy predictions to --out_csv.

The model-defining flags MUST match the training run that produced the
checkpoint, e.g.:

    python3 validate.py \
        --arch clip --temporal_mode sinusoidal \
        --checkpoint <run_dir> \
        --out_csv preds.csv --error_csv errs.csv --device cuda \
        --frame_chunk_size 32 --pooling logsumexp

Run ``python3 validate.py --help`` for all flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package next to this script is importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from angio_ft.cli import build_validate_argparser  # noqa: E402


def main() -> None:
    args = build_validate_argparser().parse_args()
    # Imported after arg parsing so `--help` does not require torch/transformers.
    from angio_ft.qa_eval import run
    run(args)


if __name__ == "__main__":
    main()
