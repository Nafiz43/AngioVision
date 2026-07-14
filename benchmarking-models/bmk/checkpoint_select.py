"""Best-checkpoint selection for the fine-tuned towers.

The eval suite, by default, evaluates the BEST available checkpoint for each
fine-tuned model — "best" = highest validation QA readout (the per-epoch cosine
YES/NO accuracy logged by the trainer into `<run>_epoch_metrics.csv`). Because
a collapsed-alignment tower can read INVERTED (its FLIPPED accuracy beats
ORIGINAL — e.g. MedSigLIP), the readout score is max(ORIGINAL, FLIPPED); the
`inverted` flag is carried through so the probe/labels can be flipped downstream.

Nothing here hardcodes an epoch. What IS fixed per model is its architecture
(objective + vision tower + temporal mode) — that's the model's definition, not
a tuning choice — encoded in MODEL_SPECS and matched to run dirs by prefix.

Discovery is dynamic: every run dir under `ckpt_root` whose name starts with a
known model prefix is considered; the globally best epoch (that also has a saved
`epoch_<n>.pt`) wins. Add a new tower by adding one MODEL_SPECS row.

Functional standalone:  python3 -m bmk.checkpoint_select <ckpt_root>
"""

from __future__ import annotations

import csv
import glob
import os
import sys

# label, run-name prefix, (arch, vit_name, temporal_mode) for validate.py --rich_probe
MODEL_SPECS = [
    ("SigLIP2",   "siglip2_",   ("siglip", "google/siglip2-base-patch16-224", "sinusoidal")),
    ("SigLIP",    "siglip_",    ("siglip", "google/siglip-base-patch16-224",  "sinusoidal")),
    ("MedSigLIP", "medsiglip_", ("siglip", "google/medsiglip-448",            "sinusoidal")),
    ("X-CLIP",    "xclip_",     ("clip",   "microsoft/xclip-base-patch32",    "none")),
    ("CLIP",      "clip_",      ("clip",   "microsoft/rad-dino",              "sinusoidal")),
]


def _spec_for(run_name: str):
    # Longest matching prefix wins so "siglip2_" beats "siglip_".
    best = None
    for label, prefix, arch in MODEL_SPECS:
        if run_name.startswith(prefix) and (best is None or len(prefix) > len(best[1])):
            best = (label, prefix, arch)
    return best


def _epoch_readouts(run_dir: str) -> list:
    """[(epoch, readout_score, inverted)] for epochs whose epoch_<n>.pt exists."""
    metrics = glob.glob(os.path.join(run_dir, "*_epoch_metrics.csv"))
    if not metrics:
        return []
    rows = []
    with open(metrics[0], newline="") as f:
        for r in csv.DictReader(f):
            try:
                ep = int(float(r["epoch"]))
                orig = float(r.get("ORIGINAL_ACCURACY", "nan"))
                flip = float(r.get("FLIPPED_ACCURACY", "nan"))
            except (KeyError, ValueError):
                continue
            if not os.path.exists(os.path.join(run_dir, f"epoch_{ep}.pt")):
                continue  # readout logged but checkpoint not on disk
            rows.append((ep, max(orig, flip), flip > orig))
    return rows


def select_checkpoints(ckpt_root: str) -> dict:
    """{label: {run, epoch, score, inverted, checkpoint, arch, vit_name, temporal_mode}}.

    Per model, the globally best (epoch, run) by validation QA readout among all
    runs of that model that have both a metrics row and a saved checkpoint.
    """
    chosen = {}
    for run_dir in sorted(glob.glob(os.path.join(ckpt_root, "*"))):
        if not os.path.isdir(run_dir):
            continue
        spec = _spec_for(os.path.basename(run_dir))
        if spec is None:
            continue
        label, _prefix, (arch, vit, temporal) = spec
        for ep, score, inverted in _epoch_readouts(run_dir):
            cur = chosen.get(label)
            if cur is None or score > cur["score"]:
                chosen[label] = {
                    "run": os.path.basename(run_dir), "epoch": ep, "score": round(score, 4),
                    "inverted": inverted,
                    "checkpoint": os.path.join(run_dir, f"epoch_{ep}.pt"),
                    "arch": arch, "vit_name": vit, "temporal_mode": temporal,
                }
    return chosen


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    sel = select_checkpoints(root)
    if not sel:
        print(f"no fine-tuned checkpoints with readouts under {root}")
    for label, s in sorted(sel.items()):
        inv = " (INVERTED→flip)" if s["inverted"] else ""
        print(f"{label:10s} {s['run']}  ep{s['epoch']}  readout={s['score']}{inv}")
        print(f"           {s['checkpoint']}")
