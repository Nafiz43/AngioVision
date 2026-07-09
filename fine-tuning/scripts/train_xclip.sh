#!/usr/bin/env bash
# End-to-end X-CLIP pipeline: softmax contrastive loss with an X-CLIP VIDEO
# vision tower (defaults to microsoft/xclip-base-patch32 unless VIT_NAME is
# set). Each sequence's frames are uniformly sampled/padded to the tower's
# clip length (config.num_frames, 8 for the base checkpoint) and encoded
# JOINTLY with cross-frame attention - temporality is modelled inside the
# tower, so TEMPORAL_MODE defaults to "none" for this arch.
#
#   ./scripts/train_xclip.sh                  # training only (per-epoch metrics CSV included)
#   ./scripts/train_xclip.sh resume           # continue from last.pt (training only)
#   ./scripts/train_xclip.sh validate         # checkpoint validation + scoring + loss plot
#   ./scripts/train_xclip.sh eval <ckpt.pt>   # evaluate one checkpoint file
#   ./scripts/train_xclip.sh full             # train, then validate + score + plot
#
# ── Variants worth running (exact commands) ──────────────────────────────────
# 1. Baseline (recommended first run): base-patch32 tower, 8-frame clips
#    EPOCHS=10 BATCH_SIZE=2 ./scripts/train_xclip.sh
#
# 2. Stronger tower - patch16 (finer spatial detail, ~4x slower):
#    VIT_NAME=microsoft/xclip-base-patch16 RUN_NAME=xclip_p16 \
#      EPOCHS=10 BATCH_SIZE=2 ./scripts/train_xclip.sh
#
# 3. Longer clips - 16-frame variant (better for long contrast runs):
#    VIT_NAME=microsoft/xclip-base-patch16-16-frames RUN_NAME=xclip_p16_16f \
#      EPOCHS=10 BATCH_SIZE=1 ./scripts/train_xclip.sh
#
# 4. Domain text tower (radiology reports; likely the best single change):
#    BERT_NAME=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
#      RUN_NAME=xclip_biomedbert EPOCHS=10 ./scripts/train_xclip.sh
#
# 5. Mean pooling over sequences (smoother study embedding than max):
#    POOLING=mean RUN_NAME=xclip_meanpool EPOCHS=10 ./scripts/train_xclip.sh
#
# 6. LiT-style (locked video tower, tune text + projections only - strong on
#    small datasets, and much cheaper):
#    EXTRA_TRAIN_ARGS="--freeze_vision" RUN_NAME=xclip_lit EPOCHS=10 ./scripts/train_xclip.sh
#
# Override any path/hyper-parameter via env vars (see scripts/train_common.sh).
ARCH=xclip
# Locate the shared driver next to this wrapper, or in a scripts/ subdir
# (works whether the wrappers live at the fine-tuning root or in scripts/).
_HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$_HERE/train_common.sh" ]]; then
    source "$_HERE/train_common.sh"
else
    source "$_HERE/scripts/train_common.sh"
fi
