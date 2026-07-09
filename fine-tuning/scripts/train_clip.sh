#!/usr/bin/env bash
# End-to-end CLIP (softmax contrastive) pipeline: train -> per-epoch val loss +
# QA metrics CSV -> checkpoint validation -> scoring -> loss plot.
#
#   ./scripts/train_clip.sh                  # training only (per-epoch metrics CSV included)
#   ./scripts/train_clip.sh resume           # continue from last.pt (training only)
#   ./scripts/train_clip.sh validate         # checkpoint validation + scoring + loss plot
#   ./scripts/train_clip.sh eval <ckpt.pt>   # evaluate one checkpoint file
#   ./scripts/train_clip.sh full             # train, then validate + score + plot
#
# ── Variants worth running (exact commands) ──────────────────────────────────
# 1. Baseline (recommended first run): rad-dino tower + sinusoidal temporal PE
#    EPOCHS=10 BATCH_SIZE=2 ./scripts/train_clip.sh
#
# 2. Temporal ablation - no frame-order encoding:
#    TEMPORAL_MODE=none RUN_NAME=clip_notime EPOCHS=10 ./scripts/train_clip.sh
#
# 3. Domain text tower (radiology reports; likely the best single change):
#    BERT_NAME=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
#      RUN_NAME=clip_biomedbert EPOCHS=10 ./scripts/train_clip.sh
#
# 4. Bigger general vision tower (scale vs domain ablation against rad-dino):
#    VIT_NAME=facebook/dinov2-large RUN_NAME=clip_dinov2L \
#      EPOCHS=10 BATCH_SIZE=1 ./scripts/train_clip.sh
#
# 5. Biomedical CLIP vision tower:
#    VIT_NAME=openai/clip-vit-base-patch16 RUN_NAME=clip_vitb16 EPOCHS=10 ./scripts/train_clip.sh
#
# 6. LiT-style (locked image tower, tune text + projections - strong on small
#    datasets, much cheaper):
#    EXTRA_TRAIN_ARGS="--freeze_vision" RUN_NAME=clip_lit EPOCHS=10 ./scripts/train_clip.sh
#
# 7. Larger effective contrastive batch via GradCache accumulation
#    (experimental - unit-tested, not yet GPU-validated):
#    EXTRA_TRAIN_ARGS="--contrastive_accum --grad_accum 8" \
#      RUN_NAME=clip_gc8 EPOCHS=10 ./scripts/train_clip.sh
#
# Override any path/hyper-parameter via env vars (see scripts/train_common.sh),
# e.g.  EPOCHS=10 BATCH_SIZE=4 ./scripts/train_clip.sh
ARCH=clip
# Locate the shared driver next to this wrapper, or in a scripts/ subdir
# (works whether the wrappers live at the fine-tuning root or in scripts/).
_HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$_HERE/train_common.sh" ]]; then
    source "$_HERE/train_common.sh"
else
    source "$_HERE/scripts/train_common.sh"
fi
