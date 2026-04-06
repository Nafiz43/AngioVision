"""
custom_framework_validate_temporal.py

Run CLIP-style binary QA on validation sequence directories using one or more trained checkpoints.

Updated to match the temporal-aware training code:
- Uses AutoModel / AutoTokenizer so it matches BioBERT or any overridden text encoder
- Adds the same sinusoidal temporal encoding logic used during training
- Supports frame-level temporal encoding and optional sequence-level temporal encoding
- Keeps checkpoint directory evaluation and best-checkpoint selection behavior

Validation flow:
1) Read validation sequences
2) Build frame embeddings with ViT
3) Add temporal positional encoding if enabled
4) Pool frames into a sequence feature
5) Optionally add sequence-order encoding if enabled
6) Project image/text into shared CLIP-style embedding space
7) Compare YES/NO hypothesis similarities
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POOL_CHOICES = ("max", "mean", "logsumexp")
TEMPORAL_MODE_CHOICES = ("none", "sinusoidal")


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


def list_images(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []
    imgs = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)


def find_frames_dir(seq_dir: Path) -> Optional[Path]:
    for name in ("frames", "Frames"):
        d = seq_dir / name
        if d.exists() and d.is_dir():
            return d
    return None


def uniform_subsample(paths: List[Path], max_frames: Optional[int]) -> List[Path]:
    if max_frames is None or max_frames <= 0 or len(paths) <= max_frames:
        return paths
    idxs = torch.linspace(0, len(paths) - 1, steps=max_frames).long().tolist()
    return [paths[i] for i in idxs]


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


def checkpoint_sort_key(p: Path):
    name = p.stem
    m = re.match(r"epoch_(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    if name == "last":
        return (1, float("inf"))
    return (2, name)


def discover_checkpoints(checkpoint_arg: str) -> List[Path]:
    p = Path(checkpoint_arg)

    if p.is_file():
        return [p]

    if not p.exists():
        raise SystemExit(f"[ERROR] Checkpoint path does not exist: {p}")

    if not p.is_dir():
        raise SystemExit(f"[ERROR] Checkpoint path is neither file nor directory: {p}")

    epoch_ckpts = sorted(
        [x for x in p.iterdir() if x.is_file() and re.fullmatch(r"epoch_\d+\.pt", x.name)],
        key=checkpoint_sort_key,
    )
    if epoch_ckpts:
        return epoch_ckpts

    last_ckpt = p / "last.pt"
    if last_ckpt.exists() and last_ckpt.is_file():
        return [last_ckpt]

    raise SystemExit(f"[ERROR] No checkpoint files found in directory: {p}")


def checkpoint_label(ckpt_path: Path) -> str:
    return ckpt_path.stem


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
# Model (must match training architecture)
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
    def encode_sequence_from_frames(
        self,
        processor,
        frame_paths: List[Path],
        device: torch.device,
        frame_chunk_size: int,
        max_frames: Optional[int],
        vit_image_size: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], str]:
        """
        Matches training-side temporal logic:
        - uniform subsample while preserving order
        - encode frame CLS features
        - add frame positional encoding if enabled
        - pool across frames
        - optional sequence-level temporal encoding (for validation single sequence,
          this is only meaningful if enabled; it remains consistent structurally)
        - project + normalize
        """
        frame_paths = uniform_subsample(frame_paths, max_frames)
        if not frame_paths:
            return None, "no_frames"

        collected_seq_feats: List[torch.Tensor] = []

        for i in range(0, len(frame_paths), frame_chunk_size):
            chunk = frame_paths[i : i + frame_chunk_size]
            imgs: List[Image.Image] = []
            valid_positions: List[int] = []

            for local_idx, p in enumerate(chunk):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid_positions.append(i + local_idx)
                except Exception:
                    continue

            if not imgs:
                continue

            if vit_image_size is not None:
                inputs = processor(
                    images=imgs,
                    return_tensors="pt",
                    size={"height": vit_image_size, "width": vit_image_size},
                )
            else:
                inputs = processor(images=imgs, return_tensors="pt")

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

            del inputs, pixel_values, out, imgs

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
    device: torch.device,
    args,
    validation_lookup: Dict[str, List[str]],
) -> Dict[str, Any]:
    model.eval()
    load_checkpoint(model, ckpt_path, device=device)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"[ERROR] Validation data dir does not exist: {data_dir}")

    final_out_csv = Path(args.out_csv)
    final_out_csv.parent.mkdir(parents=True, exist_ok=True)

    error_csv_requested = bool(args.error_csv)
    final_error_csv = Path(args.error_csv) if error_csv_requested else None
    if final_error_csv is not None:
        final_error_csv.parent.mkdir(parents=True, exist_ok=True)

    label = checkpoint_label(ckpt_path)

    temp_out_csv = final_out_csv.parent / f"{final_out_csv.stem}__{label}{final_out_csv.suffix or '.csv'}"
    temp_err_csv = None
    if final_error_csv is not None:
        temp_err_csv = final_error_csv.parent / f"{final_error_csv.stem}__{label}{final_error_csv.suffix or '.csv'}"

    error_rows = []
    n_total = 0
    n_ok = 0
    skip_counts: Dict[str, int] = {}

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None

    with open(temp_out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["AccessionNumber", "SOPInstanceUID", "Question", "Answer"])

        seq_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        for seq_dir in tqdm(seq_dirs, desc=f"Sequences [{label}]", dynamic_ncols=True):
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

            frames_dir = find_frames_dir(seq_dir)
            if frames_dir is None:
                s = "missing_frames_dir"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": ""})
                continue

            frame_paths = list_images(frames_dir)
            if not frame_paths:
                s = "no_frame_files"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": str(frames_dir)})
                continue

            img_emb, emb_status = model.encode_sequence_from_frames(
                processor=processor,
                frame_paths=frame_paths,
                device=device,
                frame_chunk_size=args.frame_chunk_size,
                max_frames=max_frames,
                vit_image_size=args.vit_image_size,
            )
            if emb_status != "ok" or img_emb is None:
                skip_counts[emb_status] = skip_counts.get(emb_status, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": emb_status, "details": ""})
                continue

            for q in relevant_questions:
                yes_h, no_h = make_yes_no_hypotheses(q)
                txt_emb = model.encode_text(tokenizer, [yes_h, no_h], device=device)
                sims = (img_emb @ txt_emb.t()).squeeze(0)
                pred = "YES" if sims[0].item() > sims[1].item() else "NO"
                writer.writerow([acc, sop, q, pred])

            n_ok += 1

    print(f"[INFO] Wrote predictions to: {temp_out_csv}")

    print("\n[SUMMARY]")
    print(f"  Checkpoint:              {ckpt_path.name}")
    print(f"  Total sequence dirs:     {n_total}")
    print(f"  Successfully predicted:  {n_ok}")
    print(f"  Frame temporal mode:     {args.temporal_mode}")
    print(f"  Frame temporal enabled:  {not args.disable_frame_temporal}")
    print(f"  Sequence temporal:       {args.enable_sequence_temporal}")
    print(f"  Frame temporal scale:    {args.frame_temporal_scale}")
    print(f"  Sequence temporal scale: {args.sequence_temporal_scale}")
    print(f"  Sequence pooling:        {args.sequence_pooling if args.sequence_pooling else args.pooling}")
    if skip_counts:
        print("  Skipped counts:")
        for k, v in sorted(skip_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {k}: {v}")

    if temp_err_csv is not None:
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

    if metrics["accuracy"] is None:
        print(f"[WARN] Could not parse accuracy from calculate_score.py output for {ckpt_path.name}")
    if metrics["f1"] is None:
        print(f"[WARN] Could not parse F1-score from calculate_score.py output for {ckpt_path.name}")

    print("\n[CHECKPOINT METRICS]")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  TP: {metrics['tp'] if metrics['tp'] is not None else 'N/A'}")
    print(f"  TN: {metrics['tn'] if metrics['tn'] is not None else 'N/A'}")
    print(f"  FP: {metrics['fp'] if metrics['fp'] is not None else 'N/A'}")
    print(f"  FN: {metrics['fn'] if metrics['fn'] is not None else 'N/A'}")
    print(f"  Accuracy: {metrics['accuracy']:.6f}" if metrics["accuracy"] is not None else "  Accuracy: N/A")
    print(f"  F1-score: {metrics['f1']:.6f}" if metrics["f1"] is not None else "  F1-score: N/A")

    return {
        "checkpoint": ckpt_path,
        "label": label,
        "pred_csv": temp_out_csv,
        "error_csv": temp_err_csv,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "n_total": n_total,
        "n_ok": n_ok,
    }


def run(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    checkpoints = discover_checkpoints(args.checkpoint)
    print(f"[INFO] Found {len(checkpoints)} checkpoint(s) to evaluate:")
    for ck in checkpoints:
        print(f"  - {ck}")

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

    results: List[Dict[str, Any]] = []
    for ckpt_path in checkpoints:
        print("\n" + "=" * 100)
        print(f"[INFO] Evaluating checkpoint: {ckpt_path}")
        print("=" * 100)
        result = run_single_checkpoint(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            ckpt_path=ckpt_path,
            device=device,
            args=args,
            validation_lookup=validation_lookup,
        )
        results.append(result)

    if not results:
        raise SystemExit("[ERROR] No checkpoints were evaluated.")

    valid_results = [r for r in results if r["accuracy"] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x["accuracy"])
    else:
        best = results[-1]
        print("[WARN] No accuracy could be parsed from any calculate_score.py output.")
        print("[WARN] Falling back to the last evaluated checkpoint as best.")

    final_out_csv = Path(args.out_csv)
    final_out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["pred_csv"], final_out_csv)

    if args.error_csv and best["error_csv"] is not None and Path(best["error_csv"]).exists():
        final_error_csv = Path(args.error_csv)
        final_error_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["error_csv"], final_error_csv)

    print("\n" + "#" * 160)
    print("[FINAL CHECKPOINT COMPARISON]")
    print("#" * 160)
    print(
        f"{'Checkpoint':<20} "
        f"{'TP':>8} "
        f"{'TN':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Accuracy':>12} "
        f"{'F1':>12} "
        f"{'Pred CSV':<40}"
    )

    for r in results:
        tp_str = "N/A" if r.get("tp") is None else str(r["tp"])
        tn_str = "N/A" if r.get("tn") is None else str(r["tn"])
        fp_str = "N/A" if r.get("fp") is None else str(r["fp"])
        fn_str = "N/A" if r.get("fn") is None else str(r["fn"])
        acc_str = "N/A" if r.get("accuracy") is None else f"{r['accuracy']:.6f}"
        f1_str = "N/A" if r.get("f1") is None else f"{r['f1']:.6f}"

        print(
            f"{r['checkpoint'].name:<20} "
            f"{tp_str:>8} "
            f"{tn_str:>8} "
            f"{fp_str:>8} "
            f"{fn_str:>8} "
            f"{acc_str:>12} "
            f"{f1_str:>12} "
            f"{r['pred_csv'].name:<40}"
        )

    print("\n[BEST CHECKPOINT]")
    print(f"  Checkpoint: {best['checkpoint']}")
    print(f"  TP: {best['tp'] if best.get('tp') is not None else 'N/A'}")
    print(f"  TN: {best['tn'] if best.get('tn') is not None else 'N/A'}")
    print(f"  FP: {best['fp'] if best.get('fp') is not None else 'N/A'}")
    print(f"  FN: {best['fn'] if best.get('fn') is not None else 'N/A'}")
    print(f"  Best accuracy: {best['accuracy']:.6f}" if best.get("accuracy") is not None else "  Best accuracy: N/A")
    print(f"  Best F1-score: {best['f1']:.6f}" if best.get("f1") is not None else "  Best F1-score: N/A")
    print(f"  Best prediction CSV copied to: {final_out_csv}")
    if args.error_csv and best["error_csv"] is not None:
        print(f"  Best error CSV copied to: {args.error_csv}")


def build_argparser():
    ap = argparse.ArgumentParser(description="Run CLIP-style binary QA on sequence-level validation data.")

    ap.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Checkpoint .pt file OR directory containing epoch checkpoints.",
    )
    ap.add_argument(
        "--data_dir",
        default="/data/Deep_Angiography/Validation_Data/test-data",
        type=str,
    )
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--error_csv", default="", type=str, help="Optional CSV logging skip/errors per sequence dir.")

    ap.add_argument(
        "--validation_csv",
        default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv",
        type=str,
        help="Validation CSV containing ground-truth rows. Only questions present here will be asked per SOPInstanceUID.",
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
    ap.add_argument("--max_frames", default=0, type=int)
    ap.add_argument("--vit_image_size", default=None, type=int)

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

