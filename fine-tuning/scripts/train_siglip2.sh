#!/usr/bin/env bash
# End-to-end SigLIP2 pipeline: sigmoid pairwise loss with a SigLIP2 pretrained
# vision tower (defaults to google/siglip2-base-patch16-224 unless VIT_NAME is
# set). Train -> per-epoch val loss + QA metrics CSV -> checkpoint validation
# -> scoring -> loss plot.
#
#   ./scripts/train_siglip2.sh                  # training only (per-epoch metrics CSV included)
#   ./scripts/train_siglip2.sh resume           # continue from last.pt (training only)
#   ./scripts/train_siglip2.sh validate         # checkpoint validation + scoring + loss plot
#   ./scripts/train_siglip2.sh eval <ckpt.pt>   # evaluate one checkpoint file
#   ./scripts/train_siglip2.sh full             # train, then validate + score + plot
#
# ── Variants worth running (exact commands) ──────────────────────────────────
# 1. Baseline (recommended first run): siglip2-base tower
#    EPOCHS=10 BATCH_SIZE=2 ./scripts/train_siglip2.sh
#
# 2. Medical SigLIP tower (likely the strongest single swap in the whole
#    fine-tuning directory - SigLIP tuned on medical imaging):
#    VIT_NAME=google/medsiglip-448 RUN_NAME=medsiglip EPOCHS=10 \
#      BATCH_SIZE=1 ./scripts/train_siglip2.sh
#
# 3. Larger SigLIP2 tower:
#    VIT_NAME=google/siglip2-large-patch16-256 RUN_NAME=siglip2_large \
#      EPOCHS=10 BATCH_SIZE=1 ./scripts/train_siglip2.sh
#
# 4. Domain text tower:
#    BERT_NAME=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
#      RUN_NAME=siglip2_biomedbert EPOCHS=10 ./scripts/train_siglip2.sh
#
# 5. LiT-style (locked image tower - pairs well with the medsiglip swap):
#    VIT_NAME=google/medsiglip-448 EXTRA_TRAIN_ARGS="--freeze_vision" \
#      RUN_NAME=medsiglip_lit EPOCHS=10 ./scripts/train_siglip2.sh
#
# Override any path/hyper-parameter via env vars (see scripts/train_common.sh),
# e.g.  EPOCHS=10 BATCH_SIZE=4 VIT_NAME=google/siglip2-large-patch16-256 ./scripts/train_siglip2.sh
ARCH=siglip2
# Locate the shared driver next to this wrapper, or in a scripts/ subdir
# (works whether the wrappers live at the fine-tuning root or in scripts/).
_HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$_HERE/train_common.sh" ]]; then
    source "$_HERE/train_common.sh"
else
    source "$_HERE/scripts/train_common.sh"
fi
