#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_common.sh - shared driver sourced by train_clip.sh / train_siglip.sh /
# train_siglip2.sh / train_xclip.sh. Do not run directly; run a per-arch wrapper.
#
# The wrapper must set ARCH before sourcing this file.
#
# Modes (first CLI arg to the wrapper):
#   train              (default) training ONLY (per-epoch metrics still logged
#                      via --epoch_qa_eval; no post-hoc validation / plot)
#   resume             continue training from last.pt in the run dir (train only)
#   validate           checkpoint validation + scoring + loss plot for the run dir
#   eval <checkpoint>  evaluate one specific checkpoint file only
#   full               train, then validation + scoring + plot in one go
#
# Every path/hyper-parameter can be overridden via environment variables, e.g.:
#   EPOCHS=10 BATCH_SIZE=4 ./train_clip.sh
#   META_CSV=/my/meta.csv REPORTS_CSV=/my/reports.csv ./train_siglip.sh
#
# Outputs (for run name $RUN_NAME):
#   $OUT_DIR/$RUN_NAME/epoch_<n>.pt            per-epoch checkpoints
#   $OUT_DIR/$RUN_NAME/last.pt                 rolling checkpoint (resume point)
#   $OUT_DIR/$RUN_NAME/${RUN_NAME}_loss.csv    step-level training loss log
#   $OUT_DIR/$RUN_NAME/${RUN_NAME}_epoch_metrics.csv
#                                              one row per epoch: train loss,
#                                              val loss, and the full QA metric
#                                              set incl. ORIGINAL / FLIPPED /
#                                              ALL_YES / ALL_NO / RANDOM baselines
#   $OUT_DIR/$RUN_NAME/epoch_eval/             per-epoch prediction/error CSVs
#   $OUTPUT_DIR/preds_${RUN_NAME}.csv          best-checkpoint predictions
#   $OUTPUT_DIR/errors_${RUN_NAME}.csv         best-checkpoint skip/error log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [[ -z "${ARCH:-}" ]]; then
    echo "[ERROR] ARCH is not set - run train_clip.sh / train_siglip.sh / train_siglip2.sh / train_xclip.sh instead." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-train}"

# ── configurable knobs (env-overridable) ─────────────────────────────────────
PYTHON="${PYTHON:-python3}"

META_CSV="${META_CSV:-/data/Deep_Angiography/AngioVision/fine-tuning/consolidated_metadata_gt.csv}"
REPORTS_CSV="${REPORTS_CSV:-/data/Deep_Angiography/AngioVision/fine-tuning/cleaned_report_list.csv}"
BASE_FRAMES_DIR="${BASE_FRAMES_DIR:-/data/Deep_Angiography/DICOM_Sequence_Processed}"
VAL_DATA_DIR="${VAL_DATA_DIR:-/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed}"
VALIDATION_CSV="${VALIDATION_CSV:-}"   # empty -> central settings.py VALIDATION_CSV

OUT_DIR="${OUT_DIR:-/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/Deep_Angiography/AngioVision/fine-tuning/output}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
# xclip models temporality inside the vision tower (cross-frame attention),
# so its default skips the redundant sinusoidal frame PE.
if [[ "$ARCH" == "xclip" ]]; then
    TEMPORAL_MODE="${TEMPORAL_MODE:-none}"
else
    TEMPORAL_MODE="${TEMPORAL_MODE:-sinusoidal}"
fi
POOLING="${POOLING:-max}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FRAME_CHUNK_SIZE="${FRAME_CHUNK_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"               # cuda | cpu (validation device; training auto-detects)

# VIT_NAME / BERT_NAME: empty -> per-arch defaults inside the pipeline
# (clip/siglip -> microsoft/rad-dino; siglip2 -> google/siglip2-base-patch16-224;
#  xclip -> microsoft/xclip-base-patch32)
VIT_NAME="${VIT_NAME:-}"
BERT_NAME="${BERT_NAME:-}"

RUN_NAME="${RUN_NAME:-${ARCH}_${TEMPORAL_MODE}_e${EPOCHS}_bs${BATCH_SIZE}}"
RUN_DIR="$OUT_DIR/$RUN_NAME"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_VALIDATE_ARGS="${EXTRA_VALIDATE_ARGS:-}"

# ── assemble shared flag fragments ───────────────────────────────────────────
MODEL_FLAGS=(--arch "$ARCH" --temporal_mode "$TEMPORAL_MODE" --pooling "$POOLING")
[[ -n "$VIT_NAME" ]]  && MODEL_FLAGS+=(--vit_name "$VIT_NAME")
[[ -n "$BERT_NAME" ]] && MODEL_FLAGS+=(--bert_name "$BERT_NAME")

TRAIN_FLAGS=(
    "${MODEL_FLAGS[@]}"
    --meta_csv "$META_CSV"
    --reports_csv "$REPORTS_CSV"
    --base_frames_dir "$BASE_FRAMES_DIR"
    --val_data_dir "$VAL_DATA_DIR"
    --out_dir "$OUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --seed "$SEED"
    --num_workers "$NUM_WORKERS"
    --frame_chunk_size "$FRAME_CHUNK_SIZE"
    --val_fraction "$VAL_FRACTION"
    --epoch_qa_eval
)
[[ -n "$VALIDATION_CSV" ]] && TRAIN_FLAGS+=(--validation_csv "$VALIDATION_CSV")
[[ "$DEVICE" == "cpu" ]] && TRAIN_FLAGS+=(--cpu)

PRED_CSV="$OUTPUT_DIR/preds_${RUN_NAME}.csv"
ERR_CSV="$OUTPUT_DIR/errors_${RUN_NAME}.csv"

run_validation() {
    local target="$1"   # run dir or single checkpoint file
    VALIDATE_FLAGS=(
        --checkpoint "$target"
        --data_dir "$VAL_DATA_DIR"
        --out_csv "$PRED_CSV"
        --error_csv "$ERR_CSV"
        --device "$DEVICE"
        --frame_chunk_size "$FRAME_CHUNK_SIZE"
        --random_seed "$SEED"
        "${MODEL_FLAGS[@]}"
    )
    [[ -n "$VALIDATION_CSV" ]] && VALIDATE_FLAGS+=(--validation_csv "$VALIDATION_CSV")
    echo "[SCRIPT] validate.py on: $target"
    # shellcheck disable=SC2086
    "$PYTHON" validate.py "${VALIDATE_FLAGS[@]}" $EXTRA_VALIDATE_ARGS
    echo "[SCRIPT] Best predictions: $PRED_CSV"
}

plot_losses() {
    local loss_csv="$RUN_DIR/${RUN_NAME}_loss.csv"
    if [[ -f "$loss_csv" ]]; then
        "$PYTHON" plot_loss.py --source_path "$loss_csv" || echo "[SCRIPT][WARN] plot_loss.py failed (non-fatal)."
    else
        echo "[SCRIPT][WARN] Loss CSV not found, skipping plot: $loss_csv"
    fi
}

echo "[SCRIPT] arch=$ARCH mode=$MODE run=$RUN_NAME"
echo "[SCRIPT] run dir: $RUN_DIR"

run_training() {
    FLAGS=("${TRAIN_FLAGS[@]}")
    [[ "$MODE" == "resume" ]] && FLAGS+=(--resume)
    [[ "${FORCE:-0}" == "1" ]] && FLAGS+=(--force)
    # shellcheck disable=SC2086
    "$PYTHON" train.py "${FLAGS[@]}" $EXTRA_TRAIN_ARGS
}

case "$MODE" in
    train|resume)
        run_training
        echo "[SCRIPT] Training finished. Run '$0 validate' for checkpoint validation + scoring + plot."
        ;;
    full)
        run_training
        run_validation "$RUN_DIR"
        plot_losses
        ;;
    validate)
        run_validation "$RUN_DIR"
        plot_losses
        ;;
    eval)
        CKPT="${2:?usage: $0 eval <checkpoint.pt>}"
        run_validation "$CKPT"
        ;;
    *)
        echo "[ERROR] Unknown mode '$MODE'. Use: train | resume | validate | eval <ckpt> | full" >&2
        exit 1
        ;;
esac

echo "[SCRIPT] Done."
echo "[SCRIPT] Checkpoints:        $RUN_DIR"
echo "[SCRIPT] Epoch metrics CSV:  $RUN_DIR/${RUN_NAME}_epoch_metrics.csv"
echo "[SCRIPT] Step loss CSV:      $RUN_DIR/${RUN_NAME}_loss.csv"
echo "[SCRIPT] Final predictions:  $PRED_CSV"
