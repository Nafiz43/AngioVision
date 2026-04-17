# this script checks how the trained model perfoms, while passing only black images 
# and then asking the questions

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import ViTModel, AutoModel, AutoTokenizer

# -----------------------------
# HF image processor compatibility
# -----------------------------
try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:
    _ViTProcessor = None

try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:
    _ViTFeatureExtractor = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
POOL_CHOICES = ("max", "mean", "logsumexp")
TEMPORAL_MODE_CHOICES = ("none", "sinusoidal")

# -----------------------------
# Black-image sanity-check defaults
# -----------------------------
DEFAULT_NUM_BLACK_FRAMES = 16
DEFAULT_BLACK_IMAGE_SIZE = 224


# -----------------------------
# Settings loader
# -----------------------------
def load_paths_from_settings(settings_py: str) -> Tuple[str, str]:
    settings_path = Path(settings_py)

    if not settings_path.exists():
        raise SystemExit(f"[ERROR] settings.py does not exist: {settings_path}")

    spec = importlib.util.spec_from_file_location("angio_settings", settings_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[ERROR] Could not load settings module from: {settings_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "VALIDATION_CSV"):
        raise SystemExit(f"[ERROR] VALIDATION_CSV not found in settings.py: {settings_path}")
    if not hasattr(module, "DATA_DIR"):
        raise SystemExit(f"[ERROR] DATA_DIR not found in settings.py: {settings_path}")

    validation_csv = str(module.VALIDATION_CSV)
    data_dir = str(module.DATA_DIR)

    print(f"[INFO] Loaded VALIDATION_CSV from settings.py: {validation_csv}")
    print(f"[INFO] Loaded DATA_DIR from settings.py: {data_dir}")

    return validation_csv, data_dir


# -----------------------------
# Utilities
# -----------------------------
def _run_subprocess_capture(cmd: List[str]) -> subprocess.CompletedProcess:
    print("\n[INFO] Running subprocess:")
    print("       " + " ".join(cmd))
    cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n")
    if cp.stderr:
        print(cp.stderr, end="" if cp.stderr.endswith("\n") else "\n", file=sys.stderr)
    return cp


def parse_score_output(text: str) -> Dict[str, Optional[float]]:
    out = {
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    text_lower = text.lower()

    def _parse_int(patterns: List[str]) -> Optional[int]:
        for pat in patterns:
            m = re.search(pat, text_lower, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return None

    def _parse_float_metric(patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text_lower, flags=re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if "%" in m.group(0) or val > 1.0:
                        val = val / 100.0
                    return val
                except Exception:
                    pass
        return None

    out["tp"] = _parse_int([
        r"\btp\b\s*[:=]\s*([0-9]+)",
        r"true\s*positives?\s*[:=]\s*([0-9]+)",
    ])
    out["tn"] = _parse_int([
        r"\btn\b\s*[:=]\s*([0-9]+)",
        r"true\s*negatives?\s*[:=]\s*([0-9]+)",
    ])
    out["fp"] = _parse_int([
        r"\bfp\b\s*[:=]\s*([0-9]+)",
        r"false\s*positives?\s*[:=]\s*([0-9]+)",
    ])
    out["fn"] = _parse_int([
        r"\bfn\b\s*[:=]\s*([0-9]+)",
        r"false\s*negatives?\s*[:=]\s*([0-9]+)",
    ])

    out["accuracy"] = _parse_float_metric([
        r"accuracy(?:\s+score)?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%",
        r"accuracy(?:\s+score)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ])

    out["precision"] = _parse_float_metric([
        r"precision\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%",
        r"precision\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ])

    out["recall"] = _parse_float_metric([
        r"recall\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%",
        r"recall\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ])

    out["f1"] = _parse_float_metric([
        r"f1(?:-score|\s*score)?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%",
        r"f1(?:-score|\s*score)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ])

    return out


def get_vit_processor(vit_name: str):
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(vit_name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(vit_name)
    raise ImportError("Neither ViTImageProcessor nor ViTFeatureExtractor is available in transformers.")


def generate_black_images(n: int, size: int) -> List[Image.Image]:
    """Generate `n` pure-black RGB PIL images of shape (size, size)."""
    if n <= 0:
        raise ValueError(f"num_black_frames must be > 0, got {n}")
    if size <= 0:
        raise ValueError(f"black_image_size must be > 0, got {size}")
    return [Image.new("RGB", (size, size), color=(0, 0, 0)) for _ in range(n)]


def normalize_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def normalize_question(q: Any) -> str:
    q = normalize_str(q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def sanitize_for_filename(s: str) -> str:
    s = s.replace("\\", "/")
    s = re.sub(r"[^\w.\-]+", "__", s)
    s = re.sub(r"__+", "__", s).strip("_")
    return s or "unnamed"


def parse_epoch_label(checkpoint_name: str) -> str:
    stem = Path(checkpoint_name).stem
    m = re.fullmatch(r"epoch_(\d+)", stem)
    if m:
        return m.group(1)
    if stem == "last":
        return "last"
    return stem


def load_validation_question_lookup(validation_csv: str) -> Dict[str, List[str]]:
    path = Path(validation_csv)
    if not path.exists():
        raise SystemExit(f"[ERROR] Validation CSV does not exist: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"[ERROR] Could not read validation CSV: {path} ({type(e).__name__}: {e})")

    required_cols = {"SOPInstanceUID", "Question"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] Validation CSV missing required columns: {sorted(missing)}")

    lookup: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        sop = normalize_str(row.get("SOPInstanceUID"))
        q = normalize_question(row.get("Question"))

        if not sop or not q:
            continue

        if sop not in lookup:
            lookup[sop] = []

        if q not in lookup[sop]:
            lookup[sop].append(q)

    print(f"[INFO] Loaded validation lookup from: {path}")
    print(f"[INFO] Unique SOPInstanceUIDs in GT: {len(lookup)}")
    total_q = sum(len(v) for v in lookup.values())
    print(f"[INFO] Total SOP-question pairs in GT: {total_q}")

    return lookup


def read_metadata_key_value_csv(seq_dir: Path) -> Tuple[Optional[str], Optional[str], str]:
    meta_path = seq_dir / "metadata.csv"
    if not meta_path.exists():
        return None, None, "missing_metadata_csv"

    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        return None, None, f"metadata_read_error:{type(e).__name__}"

    if df.empty or len(df.columns) < 2:
        return None, None, "metadata_empty_or_bad_format"

    cols = [c.strip().lower() for c in df.columns]
    try:
        info_idx = cols.index("information")
        val_idx = cols.index("value")
    except ValueError:
        info_idx, val_idx = 0, 1

    info_col = df.columns[info_idx]
    val_col = df.columns[val_idx]

    kv: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row.get(info_col, "")).strip()
        v = str(row.get(val_col, "")).strip()
        if k:
            kv[k] = v

    acc = kv.get("AccessionNumber") or kv.get("Accession Number") or kv.get("Accession")
    sop = kv.get("SOPInstanceUID") or kv.get("SOP Instance UID") or kv.get("SOPInstanceUid")

    if not acc or acc.lower() == "nan":
        return None, None, f"metadata_missing_accession(keys={list(kv.keys())[:20]})"
    if not sop or sop.lower() == "nan":
        return None, None, f"metadata_missing_sop(keys={list(kv.keys())[:20]})"

    return acc.strip(), sop.strip(), "ok"


def make_yes_no_hypotheses(question: str) -> Tuple[str, str]:
    q = question.strip().rstrip("?").lower()

    if "variant anatomy" in q:
        return (
            "Angiography demonstrates variant vascular anatomy.",
            "Angiography demonstrates no variant vascular anatomy.",
        )
    if "hemorrhage" in q or "extravasation" in q:
        return (
            "Angiography shows hemorrhage or contrast extravasation.",
            "Angiography shows no hemorrhage and no contrast extravasation.",
        )
    if "dissection" in q:
        return (
            "Angiography shows evidence of arterial or venous dissection.",
            "Angiography shows no evidence of arterial or venous dissection.",
        )
    if "stenosis" in q:
        return (
            "Angiography shows stenosis in a visualized vessel.",
            "Angiography shows no stenosis in any visualized vessel.",
        )
    if "stent" in q:
        return (
            "An endovascular stent is visible on angiography.",
            "No endovascular stent is visible on angiography.",
        )

    return (f"Yes. {question}", f"No. {question}")


def checkpoint_sort_key_name(name: str):
    stem = Path(name).stem
    m = re.fullmatch(r"epoch_(\d+)", stem)
    if m:
        return (0, int(m.group(1)))
    if stem == "last":
        return (1, float("inf"))
    return (2, stem)


def discover_experiments_and_checkpoints(checkpoint_root: str) -> List[Dict[str, Any]]:
    root = Path(checkpoint_root)

    if not root.exists():
        raise SystemExit(f"[ERROR] Checkpoint path does not exist: {root}")

    if root.is_file():
        exp_name = str(root.parent.name)
        return [{
            "experiment_name": exp_name,
            "experiment_dir": root.parent,
            "checkpoints": [root],
        }]

    if not root.is_dir():
        raise SystemExit(f"[ERROR] Checkpoint path is neither file nor directory: {root}")

    pt_files = sorted(root.rglob("*.pt"))
    if not pt_files:
        raise SystemExit(f"[ERROR] No .pt files found under: {root}")

    grouped: Dict[str, List[Path]] = {}
    for pt in pt_files:
        try:
            rel_parent = pt.parent.relative_to(root)
            exp_name = rel_parent.as_posix()
        except Exception:
            exp_name = pt.parent.name

        if exp_name == ".":
            exp_name = pt.parent.name

        grouped.setdefault(exp_name, []).append(pt)

    experiments: List[Dict[str, Any]] = []
    for exp_name, ckpts in sorted(grouped.items(), key=lambda kv: kv[0]):
        ckpts_sorted = sorted(ckpts, key=lambda p: checkpoint_sort_key_name(p.name))
        experiments.append({
            "experiment_name": exp_name,
            "experiment_dir": ckpts_sorted[0].parent,
            "checkpoints": ckpts_sorted,
        })

    return experiments


def build_sinusoidal_position_encoding(
    positions: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape={tuple(positions.shape)}")

    positions = positions.to(device=device, dtype=torch.float32)
    pe = torch.zeros((positions.size(0), dim), device=device, dtype=torch.float32)

    if dim == 1:
        pe[:, 0] = positions
        return pe.to(dtype=dtype)

    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    if dim > 1:
        cos_width = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term[:cos_width])

    return pe.to(dtype=dtype)


# -----------------------------
# Model
# -----------------------------
class PooledCLIP(nn.Module):
    def __init__(
        self,
        vit_name: str,
        text_model_name: str,
        embed_dim: int,
        frame_pooling: str,
        sequence_pooling: str,
        temporal_mode: str = "sinusoidal",
        temporal_on_frames: bool = True,
        temporal_on_sequences: bool = False,
        frame_temporal_scale: float = 0.25,
        sequence_temporal_scale: float = 0.25,
    ):
        super().__init__()
        if frame_pooling not in POOL_CHOICES:
            raise ValueError(f"frame_pooling must be one of {POOL_CHOICES}, got {frame_pooling}")
        if sequence_pooling not in POOL_CHOICES:
            raise ValueError(f"sequence_pooling must be one of {POOL_CHOICES}, got {sequence_pooling}")
        if temporal_mode not in TEMPORAL_MODE_CHOICES:
            raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODE_CHOICES}, got {temporal_mode}")

        self.vit = ViTModel.from_pretrained(vit_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.text_hidden = self.text_model.config.hidden_size
        self.frame_pooling = frame_pooling
        self.sequence_pooling = sequence_pooling

        self.temporal_mode = temporal_mode
        self.temporal_on_frames = bool(temporal_on_frames)
        self.temporal_on_sequences = bool(temporal_on_sequences)
        self.frame_temporal_scale = float(frame_temporal_scale)
        self.sequence_temporal_scale = float(sequence_temporal_scale)

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.vit_hidden),
            nn.GELU(),
            nn.Linear(self.vit_hidden, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden, self.text_hidden),
            nn.GELU(),
            nn.Linear(self.text_hidden, embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @torch.no_grad()
    def encode_text(self, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        out = self.text_model(**tok)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.text_proj(cls), dim=-1)

    def _pool_tensor(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"_pool_tensor expected 2D tensor [N, D], got shape {tuple(x.shape)}")
        if x.size(0) == 0:
            raise ValueError("Cannot pool an empty tensor.")

        if mode == "max":
            return x.max(dim=0).values
        if mode == "mean":
            return x.mean(dim=0)
        if mode == "logsumexp":
            return torch.logsumexp(x, dim=0)

        raise ValueError(f"Unsupported pooling mode: {mode}")

    def _add_temporal_encoding(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if self.temporal_mode == "none":
            return x
        if scale == 0.0:
            return x

        pe = build_sinusoidal_position_encoding(
            positions=positions,
            dim=x.size(-1),
            device=x.device,
            dtype=x.dtype,
        )
        return x + (scale * pe)

    @torch.no_grad()
    def encode_black_sequence(
        self,
        processor,
        num_frames: int,
        image_size: int,
        device: torch.device,
        frame_chunk_size: int,
        vit_image_size: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], str]:
        """
        Generate `num_frames` pure-black RGB images on the fly and encode them
        into a single sequence embedding, using the same pooling / temporal
        pipeline as real-frame encoding.
        """
        black_imgs = generate_black_images(num_frames, image_size)
        if not black_imgs:
            return None, "no_frames"

        collected_seq_feats: List[torch.Tensor] = []

        for i in range(0, len(black_imgs), frame_chunk_size):
            chunk = black_imgs[i : i + frame_chunk_size]
            valid_positions = list(range(i, i + len(chunk)))

            if vit_image_size is not None:
                inputs = processor(
                    images=chunk,
                    return_tensors="pt",
                    size={"height": vit_image_size, "width": vit_image_size},
                )
            else:
                inputs = processor(images=chunk, return_tensors="pt")

            pixel_values = inputs["pixel_values"].to(device)

            out = self.vit(pixel_values=pixel_values)
            feats = out.last_hidden_state[:, 0, :]

            if self.temporal_on_frames and self.temporal_mode != "none":
                pos_tensor = torch.tensor(valid_positions, device=device, dtype=torch.long)
                feats = self._add_temporal_encoding(
                    feats,
                    positions=pos_tensor,
                    scale=self.frame_temporal_scale,
                )

            collected_seq_feats.append(feats)

            del inputs, pixel_values, out

        if not collected_seq_feats:
            return None, "no_readable_frames"

        all_frame_feats = torch.cat(collected_seq_feats, dim=0)
        sequence_feat = self._pool_tensor(all_frame_feats, self.frame_pooling)

        seq_stack = sequence_feat.unsqueeze(0)

        if self.temporal_on_sequences and self.temporal_mode != "none":
            seq_positions = torch.arange(seq_stack.size(0), device=device, dtype=torch.long)
            seq_stack = self._add_temporal_encoding(
                seq_stack,
                positions=seq_positions,
                scale=self.sequence_temporal_scale,
            )

        study_like_feat = self._pool_tensor(seq_stack, self.sequence_pooling)
        emb = F.normalize(self.vision_proj(study_like_feat.unsqueeze(0)), dim=-1)
        return emb, "ok"


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (showing up to 20): {unexpected[:20]}")
    print(f"[INFO] Checkpoint loaded: {ckpt_path}")


def run_single_checkpoint(
    model: nn.Module,
    tokenizer,
    processor,
    ckpt_path: Path,
    experiment_name: str,
    device: torch.device,
    args,
    validation_lookup: Dict[str, List[str]],
    temp_pred_dir: Path,
    temp_err_dir: Path,
) -> Dict[str, Any]:
    model.eval()
    load_checkpoint(model, ckpt_path, device=device)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"[ERROR] Validation data dir does not exist: {data_dir}")

    checkpoint_name = ckpt_path.name
    epoch_label = parse_epoch_label(checkpoint_name)

    exp_safe = sanitize_for_filename(experiment_name)
    ckpt_safe = sanitize_for_filename(ckpt_path.stem)

    temp_out_csv = temp_pred_dir / f"{exp_safe}__{ckpt_safe}__black__predictions.csv"
    temp_err_csv = temp_err_dir / f"{exp_safe}__{ckpt_safe}__black__errors.csv"

    error_rows = []
    n_total = 0
    n_ok = 0
    total_predictions_written = 0
    skip_counts: Dict[str, int] = {}

    # --------------------------------------------------------------------
    # Compute the black-image embedding ONCE per checkpoint.
    # Since the input is deterministic (16 identical black frames) and the
    # model is in eval mode, this embedding is identical for every sequence,
    # so we avoid redundant ViT forward passes across all sequence dirs.
    # --------------------------------------------------------------------
    print(
        f"[INFO] Encoding {args.num_black_frames} black image(s) "
        f"({args.black_image_size}x{args.black_image_size}) once for this checkpoint..."
    )
    black_img_emb, black_status = model.encode_black_sequence(
        processor=processor,
        num_frames=args.num_black_frames,
        image_size=args.black_image_size,
        device=device,
        frame_chunk_size=args.frame_chunk_size,
        vit_image_size=args.vit_image_size,
    )
    if black_status != "ok" or black_img_emb is None:
        raise SystemExit(
            f"[ERROR] Failed to compute black-image embedding for checkpoint "
            f"{ckpt_path}: status={black_status}"
        )
    print("[INFO] Black-image embedding ready. Reusing across all sequences.")

    with open(temp_out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["AccessionNumber", "SOPInstanceUID", "Question", "Answer"])

        seq_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        for seq_dir in tqdm(
            seq_dirs,
            desc=f"{experiment_name} | {checkpoint_name} | BLACK",
            dynamic_ncols=True,
        ):
            n_total += 1

            acc, sop, meta_status = read_metadata_key_value_csv(seq_dir)
            if meta_status != "ok":
                skip_counts[meta_status] = skip_counts.get(meta_status, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": meta_status, "details": ""})
                continue

            sop_norm = normalize_str(sop)
            relevant_questions = validation_lookup.get(sop_norm, [])
            if not relevant_questions:
                s = "no_gt_questions_for_sop"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append(
                    {
                        "seq_dir": str(seq_dir),
                        "status": s,
                        "details": f"SOPInstanceUID={sop}",
                    }
                )
                continue

            # NOTE: Real frames are intentionally NOT loaded here — this is a
            # black-image sanity check. We reuse the precomputed black embedding.
            img_emb = black_img_emb

            for q in relevant_questions:
                yes_h, no_h = make_yes_no_hypotheses(q)
                txt_emb = model.encode_text(tokenizer, [yes_h, no_h], device=device)
                sims = (img_emb @ txt_emb.t()).squeeze(0)
                pred = "YES" if sims[0].item() > sims[1].item() else "NO"
                writer.writerow([acc, sop, q, pred])
                total_predictions_written += 1

            n_ok += 1

    print(f"[INFO] Wrote predictions to: {temp_out_csv}")

    pd.DataFrame(error_rows, columns=["seq_dir", "status", "details"]).to_csv(temp_err_csv, index=False)
    print(f"[INFO] Wrote error log to: {temp_err_csv}")

    if not temp_out_csv.exists():
        raise SystemExit(f"[ERROR] Prediction CSV not found, cannot run calculate_score.py: {temp_out_csv}")

    score_cmd = [
        sys.executable,
        args.calculate_score_script,
        "--pred_path",
        str(temp_out_csv),
    ]
    cp = _run_subprocess_capture(score_cmd)

    combined_output = ""
    if cp.stdout:
        combined_output += cp.stdout
    if cp.stderr:
        combined_output += "\n" + cp.stderr

    metrics = parse_score_output(combined_output)

    print("\n[CHECKPOINT METRICS - BLACK IMAGE SANITY CHECK]")
    print(f"  Experiment: {experiment_name}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Epoch: {epoch_label}")
    print(f"  Num black frames: {args.num_black_frames}")
    print(f"  Black image size: {args.black_image_size}")
    print(f"  TP: {metrics['tp'] if metrics['tp'] is not None else 'N/A'}")
    print(f"  TN: {metrics['tn'] if metrics['tn'] is not None else 'N/A'}")
    print(f"  FP: {metrics['fp'] if metrics['fp'] is not None else 'N/A'}")
    print(f"  FN: {metrics['fn'] if metrics['fn'] is not None else 'N/A'}")
    print(f"  Accuracy: {metrics['accuracy']:.6f}" if metrics["accuracy"] is not None else "  Accuracy: N/A")
    print(f"  Precision: {metrics['precision']:.6f}" if metrics["precision"] is not None else "  Precision: N/A")
    print(f"  Recall: {metrics['recall']:.6f}" if metrics["recall"] is not None else "  Recall: N/A")
    print(f"  F1-score: {metrics['f1']:.6f}" if metrics["f1"] is not None else "  F1-score: N/A")

    return {
        "experiment_name": experiment_name,
        "checkpoint_name": checkpoint_name,
        "epoch": epoch_label,
        "checkpoint_path": str(ckpt_path),
        "prediction_csv": str(temp_out_csv),
        "error_csv": str(temp_err_csv),
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_total_sequence_dirs": n_total,
        "n_successful_sequences": n_ok,
        "n_predictions_written": total_predictions_written,
        "skip_summary": "; ".join(f"{k}={v}" for k, v in sorted(skip_counts.items())),
        "num_black_frames": args.num_black_frames,
        "black_image_size": args.black_image_size,
        "frame_temporal_mode": args.temporal_mode,
        "frame_temporal_enabled": not args.disable_frame_temporal,
        "sequence_temporal_enabled": args.enable_sequence_temporal,
        "frame_temporal_scale": args.frame_temporal_scale,
        "sequence_temporal_scale": args.sequence_temporal_scale,
        "frame_pooling": args.frame_pooling if args.frame_pooling else args.pooling,
        "sequence_pooling": args.sequence_pooling if args.sequence_pooling else args.pooling,
        "status": "ok",
        "error": "",
    }


def build_leaderboards(
    results: List[Dict[str, Any]],
    leaderboard_dir: Path,
) -> Tuple[Path, Path]:
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    all_csv = leaderboard_dir / "experiments_all_black.csv"
    sorted_csv = leaderboard_dir / "experiments_sorted_black.csv"

    df = pd.DataFrame(results)

    desired_cols = [
        "experiment_name",
        "checkpoint_name",
        "epoch",
        "checkpoint_path",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "n_total_sequence_dirs",
        "n_successful_sequences",
        "n_predictions_written",
        "prediction_csv",
        "error_csv",
        "status",
        "error",
        "num_black_frames",
        "black_image_size",
        "frame_temporal_mode",
        "frame_temporal_enabled",
        "sequence_temporal_enabled",
        "frame_temporal_scale",
        "sequence_temporal_scale",
        "frame_pooling",
        "sequence_pooling",
        "skip_summary",
    ]

    existing_cols = [c for c in desired_cols if c in df.columns]
    df = df[existing_cols]

    df.to_csv(all_csv, index=False)

    sort_df = df.copy()
    if "f1" in sort_df.columns:
        sort_df = sort_df.sort_values(by=["f1"], ascending=[False], na_position="last")

    sort_df.to_csv(sorted_csv, index=False)

    return all_csv, sorted_csv


def print_final_leaderboard(results: List[Dict[str, Any]]):
    print("\n" + "#" * 180)
    print("[FINAL EXPERIMENT LEADERBOARD - BLACK IMAGE SANITY CHECK]")
    print("#" * 180)

    header = (
        f"{'Experiment':<45} "
        f"{'Checkpoint':<15} "
        f"{'Epoch':<8} "
        f"{'TP':>8} "
        f"{'TN':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Accuracy':>12} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F1':>12}"
    )
    print(header)
    print("-" * len(header))

    def sort_key(r: Dict[str, Any]):
        f1 = r.get("f1")
        return -(f1 if f1 is not None else -1e18)

    for r in sorted(results, key=sort_key):
        tp_str = "N/A" if r.get("tp") is None else str(r["tp"])
        tn_str = "N/A" if r.get("tn") is None else str(r["tn"])
        fp_str = "N/A" if r.get("fp") is None else str(r["fp"])
        fn_str = "N/A" if r.get("fn") is None else str(r["fn"])
        acc_str = "N/A" if r.get("accuracy") is None else f"{r['accuracy']:.6f}"
        prec_str = "N/A" if r.get("precision") is None else f"{r['precision']:.6f}"
        rec_str = "N/A" if r.get("recall") is None else f"{r['recall']:.6f}"
        f1_str = "N/A" if r.get("f1") is None else f"{r['f1']:.6f}"

        print(
            f"{r['experiment_name'][:45]:<45} "
            f"{r['checkpoint_name'][:15]:<15} "
            f"{str(r['epoch'])[:8]:<8} "
            f"{tp_str:>8} "
            f"{tn_str:>8} "
            f"{fp_str:>8} "
            f"{fn_str:>8} "
            f"{acc_str:>12} "
            f"{prec_str:>12} "
            f"{rec_str:>12} "
            f"{f1_str:>12}"
        )


def build_failed_checkpoint_result(
    experiment_name: str,
    ckpt_path: Path,
    args,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "checkpoint_name": ckpt_path.name,
        "epoch": parse_epoch_label(ckpt_path.name),
        "checkpoint_path": str(ckpt_path),
        "prediction_csv": "",
        "error_csv": "",
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "n_total_sequence_dirs": 0,
        "n_successful_sequences": 0,
        "n_predictions_written": 0,
        "skip_summary": "",
        "num_black_frames": args.num_black_frames,
        "black_image_size": args.black_image_size,
        "frame_temporal_mode": args.temporal_mode,
        "frame_temporal_enabled": not args.disable_frame_temporal,
        "sequence_temporal_enabled": args.enable_sequence_temporal,
        "frame_temporal_scale": args.frame_temporal_scale,
        "sequence_temporal_scale": args.sequence_temporal_scale,
        "frame_pooling": args.frame_pooling if args.frame_pooling else args.pooling,
        "sequence_pooling": args.sequence_pooling if args.sequence_pooling else args.pooling,
        "status": "failed",
        "error": error_message,
    }


def run(args):
    settings_validation_csv, settings_data_dir = load_paths_from_settings(args.settings_py)

    if not args.validation_csv:
        args.validation_csv = settings_validation_csv
    if not args.data_dir:
        args.data_dir = settings_data_dir

    print(f"[INFO] Final validation_csv: {args.validation_csv}")
    print(f"[INFO] Final data_dir: {args.data_dir}")
    print(f"[INFO] Black-image sanity check: num_black_frames={args.num_black_frames}, "
          f"black_image_size={args.black_image_size}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    experiments = discover_experiments_and_checkpoints(args.checkpoint_root)
    print(f"[INFO] Found {len(experiments)} experiment(s) under: {args.checkpoint_root}")
    for exp in experiments:
        print(f"  - {exp['experiment_name']} ({len(exp['checkpoints'])} checkpoints)")

    processor = get_vit_processor(args.vit_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    frame_pooling = args.frame_pooling if args.frame_pooling else args.pooling
    sequence_pooling = args.sequence_pooling if args.sequence_pooling else args.pooling

    validation_lookup = load_validation_question_lookup(args.validation_csv)

    model = PooledCLIP(
        vit_name=args.vit_name,
        text_model_name=args.bert_name,
        embed_dim=args.embed_dim,
        frame_pooling=frame_pooling,
        sequence_pooling=sequence_pooling,
        temporal_mode=args.temporal_mode,
        temporal_on_frames=(not args.disable_frame_temporal),
        temporal_on_sequences=args.enable_sequence_temporal,
        frame_temporal_scale=args.frame_temporal_scale,
        sequence_temporal_scale=args.sequence_temporal_scale,
    ).to(device)
    model.eval()

    leaderboard_dir = Path(args.leaderboard_dir)
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    temp_pred_dir = leaderboard_dir / "tmp_predictions_black"
    temp_err_dir = leaderboard_dir / "tmp_errors_black"
    temp_pred_dir.mkdir(parents=True, exist_ok=True)
    temp_err_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for exp in experiments:
        experiment_name = exp["experiment_name"]
        checkpoints = exp["checkpoints"]

        print("\n" + "=" * 120)
        print(f"[INFO] Experiment: {experiment_name}")
        print(f"[INFO] Number of checkpoints: {len(checkpoints)}")
        print("=" * 120)

        for ckpt_path in checkpoints:
            try:
                result = run_single_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    ckpt_path=ckpt_path,
                    experiment_name=experiment_name,
                    device=device,
                    args=args,
                    validation_lookup=validation_lookup,
                    temp_pred_dir=temp_pred_dir,
                    temp_err_dir=temp_err_dir,
                )
                results.append(result)
            except Exception as e:
                error_message = f"{type(e).__name__}: {e}"
                traceback_text = traceback.format_exc()

                print("\n" + "!" * 120, file=sys.stderr)
                print(f"[ERROR] Failed checkpoint: {ckpt_path}", file=sys.stderr)
                print(f"[ERROR] {error_message}", file=sys.stderr)
                print(traceback_text, file=sys.stderr)
                print("[WARN] Skipping failed checkpoint and continuing with the remaining checkpoints.", file=sys.stderr)
                print("!" * 120 + "\n", file=sys.stderr)

                results.append(
                    build_failed_checkpoint_result(
                        experiment_name=experiment_name,
                        ckpt_path=ckpt_path,
                        args=args,
                        error_message=error_message,
                    )
                )
                continue

    if not results:
        raise SystemExit("[ERROR] No checkpoints were evaluated.")

    all_csv, sorted_csv = build_leaderboards(results, leaderboard_dir)

    print_final_leaderboard(results)

    print("\n[LEADERBOARD FILES WRITTEN]")
    print(f"  experiments_all_black.csv    : {all_csv}")
    print(f"  experiments_sorted_black.csv : {sorted_csv}")

    failed_results = [r for r in results if r.get("status") == "failed"]
    if failed_results:
        print("\n[WARN] Some checkpoints failed but the run continued.")
        print(f"  Failed checkpoints: {len(failed_results)}")
        for r in failed_results:
            print(f"    - {r['experiment_name']} | {r['checkpoint_name']} | {r.get('error', '')}")

    valid_results = [r for r in results if r.get("accuracy") is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x["accuracy"])
        print("\n[BEST CHECKPOINT OVERALL - BLACK IMAGE SANITY CHECK]")
        print(f"  Experiment : {best['experiment_name']}")
        print(f"  Checkpoint : {best['checkpoint_name']}")
        print(f"  Epoch      : {best['epoch']}")
        print(f"  Accuracy   : {best['accuracy']:.6f}")
        print(f"  Precision  : {best['precision']:.6f}" if best.get("precision") is not None else "  Precision  : N/A")
        print(f"  Recall     : {best['recall']:.6f}" if best.get("recall") is not None else "  Recall     : N/A")
        print(f"  F1-score   : {best['f1']:.6f}" if best.get("f1") is not None else "  F1-score   : N/A")
        print(f"  Pred CSV   : {best['prediction_csv']}")
        print(f"  Error CSV  : {best['error_csv']}")
    else:
        print("\n[WARN] No valid accuracy values were parsed from calculate_score.py output.")


def build_argparser():
    ap = argparse.ArgumentParser(
        description=(
            "Black-image sanity check for temporal CLIP-style validation. "
            "For every experiment and every checkpoint under an outer checkpoint "
            "directory, pass N black images (generated on the fly) followed by "
            "the validation questions, and see how the model performs."
        )
    )

    ap.add_argument(
        "--checkpoint_root",
        required=True,
        type=str,
        help="Outer checkpoint directory. Everything between this directory and a .pt file is treated as experiment name.",
    )

    ap.add_argument(
        "--settings_py",
        default="/data/Deep_Angiography/AngioVision/configs/settings.py",
        type=str,
        help="Path to settings.py containing VALIDATION_CSV and DATA_DIR.",
    )

    ap.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Optional override for validation sequence directory. If empty, reads DATA_DIR from settings.py.",
    )

    ap.add_argument(
        "--validation_csv",
        default="",
        type=str,
        help="Optional override for validation CSV. If empty, reads VALIDATION_CSV from settings.py.",
    )

    ap.add_argument(
        "--leaderboard_dir",
        default="/data/Deep_Angiography/AngioVision/fine-tuning/statistical-test-result-black",
        type=str,
        help="Directory where experiments_all_black.csv and experiments_sorted_black.csv will be written.",
    )

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--vit_name", default="google/vit-base-patch16-224-in21k", type=str)
    ap.add_argument(
        "--bert_name",
        default="dmis-lab/biobert-base-cased-v1.1",
        type=str,
        help="Text encoder model name. Should match training.",
    )
    ap.add_argument("--embed_dim", default=256, type=int)

    ap.add_argument("--pooling", default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", default="", choices=("",) + POOL_CHOICES)
    ap.add_argument("--sequence_pooling", default="", choices=("",) + POOL_CHOICES)

    ap.add_argument(
        "--temporal_mode",
        type=str,
        default="sinusoidal",
        choices=TEMPORAL_MODE_CHOICES,
        help="How to inject order information. Should match training.",
    )
    ap.add_argument(
        "--disable_frame_temporal",
        action="store_true",
        help="Disable frame-order positional encoding inside each sequence.",
    )
    ap.add_argument(
        "--enable_sequence_temporal",
        action="store_true",
        help="Also add positional encoding across sequence order.",
    )
    ap.add_argument(
        "--frame_temporal_scale",
        type=float,
        default=0.25,
        help="Scale of frame-order positional encoding.",
    )
    ap.add_argument(
        "--sequence_temporal_scale",
        type=float,
        default=0.25,
        help="Scale of sequence-order positional encoding.",
    )

    ap.add_argument("--frame_chunk_size", default=64, type=int)
    ap.add_argument("--vit_image_size", default=None, type=int)

    # --- Black-image sanity-check controls ---
    ap.add_argument(
        "--num_black_frames",
        type=int,
        default=DEFAULT_NUM_BLACK_FRAMES,
        help="Number of black frames to generate per sequence (default: 16).",
    )
    ap.add_argument(
        "--black_image_size",
        type=int,
        default=DEFAULT_BLACK_IMAGE_SIZE,
        help="Spatial size (HxW) of each generated black image before ViT processing (default: 224).",
    )

    ap.add_argument(
        "--calculate_score_script",
        default="calculate_score.py",
        type=str,
        help="Path to calculate_score.py script to run after validation.",
    )

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)


# python3 custom_framework_sanity_check.py \
#   --checkpoint_root /data/Deep_Angiography/AngioVision/fine-tuning/d_checkpoints \
#   --num_black_frames 16 \
#   --black_image_size 224