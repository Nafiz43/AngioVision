#!/usr/bin/env bash
# End-to-end SigLIP (sigmoid pairwise contrastive) pipeline: train -> per-epoch
# val loss + QA metrics CSV -> checkpoint validation -> scoring -> loss plot.
#
#   ./scripts/train_siglip.sh                  # training only (per-epoch metrics CSV included)
#   ./scripts/train_siglip.sh resume           # continue from last.pt (training only)
#   ./scripts/train_siglip.sh validate         # checkpoint validation + scoring + loss plot
#   ./scripts/train_siglip.sh eval <ckpt.pt>   # evaluate one checkpoint file
#   ./scripts/train_siglip.sh full             # train, then validate + score + plot
#
# ── Variants worth running (exact commands) ──────────────────────────────────
# 1. Baseline (recommended first run): rad-dino tower + sigmoid loss
#    EPOCHS=10 BATCH_SIZE=2 ./scripts/train_siglip.sh
#
# 2. Sigmoid loss shines with a larger effective batch - GradCache accumulation
#    (experimental - unit-tested, not yet GPU-validated):
#    EXTRA_TRAIN_ARGS="--contrastive_accum --grad_accum 8" \
#      RUN_NAME=siglip_gc8 EPOCHS=10 ./scripts/train_siglip.sh
#
# 3. Domain text tower:
#    BERT_NAME=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
#      RUN_NAME=siglip_biomedbert EPOCHS=10 ./scripts/train_siglip.sh
#
# 4. Temporal ablation - no frame-order encoding:
#    TEMPORAL_MODE=none RUN_NAME=siglip_notime EPOCHS=10 ./scripts/train_siglip.sh
#
# 5. Mean pooling (frames + sequences) instead of max:
#    POOLING=mean RUN_NAME=siglip_meanpool EPOCHS=10 ./scripts/train_siglip.sh
#
# Override any path/hyper-parameter via env vars (see scripts/train_common.sh),
# e.g.  EPOCHS=10 BATCH_SIZE=4 ./scripts/train_siglip.sh
ARCH=siglip
# Locate the shared driver next to this wrapper, or in a scripts/ subdir
# (works whether the wrappers live at the fine-tuning root or in scripts/).
_HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$_HERE/train_common.sh" ]]; then
    source "$_HERE/train_common.sh"
else
    source "$_HERE/scripts/train_common.sh"
fi
