"""
angio_ft.cli
─────────────
Argument parsers for the unified entrypoints.

``build_train_argparser`` is a superset of every flag from the original
``custom_framework_train_temporal.py`` / ``siglip.py`` trainers, plus ``--arch``
to select the contrastive objective.  ``build_validate_argparser`` mirrors the
model-defining flags so evaluation reconstructs the exact training architecture.

Structural ablations
    --arch {clip,siglip}                 objective (softmax vs sigmoid)
    --temporal_mode {none,sinusoidal}    temporal component on/off
    --disable_frame_temporal             turn OFF frame-level temporal
    --enable_sequence_temporal           turn ON sequence-level temporal

Hyper-parameter ablations
    pooling, LR groups, batch/epochs/grad_accum/amp, ViT unfreezing, chunking...
"""

from __future__ import annotations

import argparse

from .constants import ARCH_CHOICES, POOL_CHOICES, TEMPORAL_MODE_CHOICES

_DEFAULT_CKPT_DIR = "/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints"
_DEFAULT_OUT_DIR = "/data/Deep_Angiography/AngioVision/fine-tuning/output"
_DEFAULT_VAL_DIR = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed"


def build_train_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified AngioVision contrastive fine-tuning (CLIP / SigLIP, +/- temporal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── architecture / objective (structural ablation) ─────────────────────
    ap.add_argument("--arch", default="clip", choices=ARCH_CHOICES,
                    help="Architecture: clip (softmax), siglip/siglip2 (sigmoid), "
                         "xclip (softmax + X-CLIP video tower with cross-frame attention).")

    # ── optional report-generation head ────────────────────────────────────
    # The GPT-2 decoder + hold-out generation machinery lives in the archived,
    # fully-tested generation trainer. Setting this flag makes train.py delegate
    # to it, forwarding all other args (generation-specific flags such as
    # --decoder_model_name / --gen_loss_weight / --clip_loss_weight /
    # --generation_max_length / --holdout_* are passed straight through).
    ap.add_argument("--enable_generation", action="store_true",
                    help="Train an optional report-generation (GPT-2) head. Delegates to the "
                         "archived generation trainer (CLIP objective only).")

    # ── required / data ────────────────────────────────────────────────────
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--reports_csv", required=True)
    ap.add_argument("--base_frames_dir", required=True)

    # ── column names ───────────────────────────────────────────────────────
    ap.add_argument("--report_text_col", default="cleaned_radrpt")
    ap.add_argument("--anon_col", default="Anon Acc #")
    ap.add_argument("--sop_col", default="SOPInstanceUIDs")
    ap.add_argument("--report_type_col", default="Type",
                    help="Column flagging Original / Augmented variants.")
    ap.add_argument("--report_sampling", default="uniform", choices=["uniform"])
    ap.add_argument("--report_sampling_seed", type=int, default=42)

    # ── model ──────────────────────────────────────────────────────────────
    ap.add_argument("--vit_name", default=None,
                    help="Vision tower. Defaults per --arch: clip/siglip -> microsoft/rad-dino, "
                         "siglip2 -> google/siglip2-base-patch16-224, xclip -> "
                         "microsoft/xclip-base-patch32. SigLIP/SigLIP2/CLIP/X-CLIP vision "
                         "checkpoints are supported in addition to plain ViT/DINO ones.")
    ap.add_argument("--bert_name", default="UCSD-VA-health/RadBERT-RoBERTa-4m")
    ap.add_argument("--embed_dim", type=int, default=256)

    # ── training ───────────────────────────────────────────────────────────
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--vision_backbone_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--text_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    ap.add_argument("--contrastive_accum", action="store_true",
                    help="GradCache-style accumulation: treat batch_size*grad_accum as ONE "
                         "contrastive batch (all micro-batch embeddings become mutual "
                         "negatives) at single-micro-batch activation memory. Default off = "
                         "legacy per-micro-batch loss (no extra negatives).")
    ap.add_argument("--warmup_steps", type=int, default=100,
                    help="Linear warmup steps for cosine LR scheduler.")
    ap.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    ap.add_argument("--vit_grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global seed (torch/numpy/random) for reproducible runs.")

    # ── per-epoch validation ───────────────────────────────────────────────
    ap.add_argument("--val_fraction", type=float, default=0.0,
                    help="Fraction of studies held out of training as a validation split "
                         "for the per-epoch validation loss (0 = disabled). Split is "
                         "study-level and seeded by --seed.")
    ap.add_argument("--epoch_qa_eval", action="store_true",
                    help="After every epoch, run binary-QA validation on --val_data_dir "
                         "against --validation_csv and log the full metric set (incl. "
                         "ALL_YES / ALL_NO / RANDOM / FLIPPED baselines) to the "
                         "epoch-metrics CSV in the run directory.")
    ap.add_argument("--validation_csv", default="",
                    help="GT CSV for --epoch_qa_eval. Empty = fall back to the central "
                         "settings.py VALIDATION_CSV if available.")

    # ── data loading ───────────────────────────────────────────────────────
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--io_threads", type=int, default=4,
                    help="ThreadPoolExecutor workers for parallel image loading.")
    ap.add_argument("--cpu", action="store_true")

    # ── sequence / frame limits ────────────────────────────────────────────
    frac = ap.add_mutually_exclusive_group()
    frac.add_argument("--20%", dest="frames_20pct", action="store_true",
                      help="Train on a random 20%% of frames per sequence "
                           "(80%% randomly skipped, seeded by --seed).")
    frac.add_argument("--full-data", dest="frames_20pct", action="store_false",
                      help="Train on all frames (default).")
    ap.set_defaults(frames_20pct=False)
    ap.add_argument("--frame_chunk_size", type=int, default=16)
    ap.add_argument("--min_frames_per_sequence", type=int, default=1)
    ap.add_argument("--max_sequences_per_study", type=int, default=16)
    ap.add_argument("--max_frames_per_sequence", type=int, default=16)
    ap.add_argument("--early_stop_patience", type=int, default=5,
                    help="Stop if val_loss shows no improvement for this many "
                         "consecutive epochs. 0 disables early stopping.")

    # ── ViT fine-tuning ────────────────────────────────────────────────────
    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")
    ap.add_argument("--freeze_vision_proj", action="store_true",
                    help="Freeze the vision projection MLP (trainable by default). Combine with "
                         "--freeze_vision for a fully locked image tower (LiT-style).")
    ap.add_argument("--freeze_text_proj", action="store_true",
                    help="Freeze the text projection MLP (trainable by default).")
    ap.add_argument("--vit_trainable_blocks", type=int, default=3)
    ap.add_argument("--vit_unfreeze_patch_embed", action="store_true")

    # ── temporal encoding (structural ablation) ────────────────────────────
    ap.add_argument("--temporal_mode", default="sinusoidal", choices=TEMPORAL_MODE_CHOICES,
                    help="'none' disables temporal encoding entirely.")
    ap.add_argument("--disable_frame_temporal", action="store_true",
                    help="Turn OFF frame-level temporal encoding (on by default).")
    ap.add_argument("--enable_sequence_temporal", action="store_true",
                    help="Turn ON sequence-level temporal encoding (off by default).")
    ap.add_argument("--frame_temporal_scale", type=float, default=0.25)
    ap.add_argument("--sequence_temporal_scale", type=float, default=0.25)

    # ── pooling ────────────────────────────────────────────────────────────
    ap.add_argument("--pooling", default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", default=None, choices=POOL_CHOICES)
    ap.add_argument("--sequence_pooling", default=None, choices=POOL_CHOICES)

    # ── loss / logits ──────────────────────────────────────────────────────
    ap.add_argument("--logits_chunk", type=int, default=4,
                    help="Row chunk size when building the BxB similarity matrix.")
    ap.add_argument("--loss_flush_every", type=int, default=50,
                    help="Flush loss CSV buffer every N steps.")

    # ── CUDA memory ────────────────────────────────────────────────────────
    ap.add_argument("--vit_image_size", type=int, default=None)
    ap.add_argument("--empty_cache_each_step", action="store_true")
    ap.add_argument("--cache_clear_interval", type=int, default=10,
                    help="Call torch.cuda.empty_cache() every N steps (requires --empty_cache_each_step).")

    # ── output / post-training pipeline ────────────────────────────────────
    ap.add_argument("--out_dir", default=_DEFAULT_CKPT_DIR)
    ap.add_argument("--output_dir", default=_DEFAULT_OUT_DIR)
    ap.add_argument("--val_data_dir", default=_DEFAULT_VAL_DIR)
    ap.add_argument("--run_name", default=None,
                    help="Override the auto-generated run-directory name "
                         "({arch}_{tempTAG}_{epochs}_{bs}_{maxseq}_{maxframes}_h{cfghash}).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last.pt in the run directory (model+optimizer+step).")
    ap.add_argument("--force", action="store_true",
                    help="Proceed even if the run directory already has checkpoints "
                         "(they will be overwritten as training progresses).")
    ap.add_argument("--keep_missing_reports", action="store_true")
    ap.add_argument("--run_post_pipeline", action="store_true",
                    help="After training, automatically run validate -> score -> plot.")
    ap.add_argument("--validate_script", default="validate.py")
    ap.add_argument("--calculate_score_script", default="calculate_score.py")
    ap.add_argument("--plot_loss_script", default="plot_loss.py")
    ap.add_argument("--validate_device", default=None)

    return ap


def build_validate_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified AngioVision binary-QA validation for CLIP / SigLIP (+/- temporal) checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--checkpoint", required=True, type=str,
                    help="Checkpoint .pt file OR run directory containing epoch_*.pt / last.pt.")
    ap.add_argument("--data_dir", default="/data/Deep_Angiography/Validation_Data/test-data", type=str,
                    help="Fallback data dir. Overridden if settings.py provides DATA_DIR.")
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--error_csv", default="", type=str,
                    help="Optional CSV logging skip/errors per sequence dir.")
    ap.add_argument("--validation_csv",
                    default="/data/Deep_Angiography/Validation_Data/test-data/gt.csv", type=str,
                    help="Fallback validation CSV (also passed to calculate_score.py as --gt_path). "
                         "Overridden if settings.py provides VALIDATION_CSV.")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ── model (MUST match the training run that produced the checkpoint) ────
    ap.add_argument("--arch", default="clip", choices=ARCH_CHOICES)
    ap.add_argument("--vit_name", default=None, type=str,
                    help="Vision tower. Defaults per --arch (clip/siglip -> microsoft/rad-dino, "
                         "siglip2 -> google/siglip2-base-patch16-224, xclip -> "
                         "microsoft/xclip-base-patch32); normally overridden by the "
                         "checkpoint's embedded config.")
    ap.add_argument("--bert_name", default="UCSD-VA-health/RadBERT-RoBERTa-4m", type=str)
    ap.add_argument("--embed_dim", default=256, type=int)
    ap.add_argument("--vit_image_size", type=int, default=None)

    # ── pooling ────────────────────────────────────────────────────────────
    ap.add_argument("--pooling", default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", default="", choices=("",) + POOL_CHOICES)
    ap.add_argument("--sequence_pooling", default="", choices=("",) + POOL_CHOICES)

    # ── temporal (MUST match training) ─────────────────────────────────────
    ap.add_argument("--temporal_mode", default="sinusoidal", choices=TEMPORAL_MODE_CHOICES)
    ap.add_argument("--disable_frame_temporal", action="store_true")
    ap.add_argument("--enable_sequence_temporal", action="store_true")
    ap.add_argument("--frame_temporal_scale", type=float, default=0.25)
    ap.add_argument("--sequence_temporal_scale", type=float, default=0.25)

    # ── inference behaviour ────────────────────────────────────────────────
    ap.add_argument("--sequence_repeat_factor", default=1, type=int,
                    help="Repeat the pooled sequence before study-level pooling "
                         "(legacy base-validator behaviour used 16; default 1 = no repeat).")
    ap.add_argument("--frame_chunk_size", default=64, type=int)
    ap.add_argument("--max_frames", default=0, type=int)

    ap.add_argument("--calculate_score_script", default="calculate_score.py", type=str,
                    help="Path to calculate_score.py script to run after validation.")
    ap.add_argument("--random_seed", default=42, type=int,
                    help="Random seed used for the RANDOM baseline in calculate_score.py.")

    # ── evaluation protocol ────────────────────────────────────────────────
    ap.add_argument("--selection_csv", default="", type=str,
                    help="Optional dev-split GT CSV used ONLY to select the best checkpoint. "
                         "When set, final metrics are reported once against the held-out "
                         "--validation_csv (see split_gt.py to create the split).")
    ap.add_argument("--margin_debias", action="store_true",
                    help="Debias yes/no decisions by subtracting the per-question-family "
                         "median similarity margin (unsupervised, no GT labels used).")
    ap.add_argument("--lr_probe", action="store_true",
                    help="After scoring, run a grouped-CV logistic-regression probe on the "
                         "buffered (sequence, question) embeddings against GT answers "
                         "(templates + LR readout; GroupKFold by accession).")
    ap.add_argument("--prompt_ensemble", action="store_true",
                    help="Average text embeddings over several yes/no hypothesis paraphrases "
                         "per question instead of a single sentence pair.")

    return ap
