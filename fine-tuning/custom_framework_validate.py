#!/usr/bin/env python3
"""
custom_framework_validate.py

Run CLIP-style binary QA on validation sequence directories using one or more trained checkpoints.
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

from transformers import ViTModel, BertModel, BertTokenizer

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


def _run_subprocess_capture(cmd: List[str]) -> subprocess.CompletedProcess:
    print("\n[INFO] Running subprocess:")
    print("       " + " ".join(cmd))
    cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if cp.stdout:
        print(cp.stdout, end="" if cp.stdout.endswith("\n") else "\n")
    if cp.stderr:
        print(cp.stderr, end="" if cp.stderr.endswith("\n") else "\n", file=sys.stderr)
    return cp


def _parse_int_metric(text: str, key: str) -> Optional[int]:
    m = re.search(rf"\b{re.escape(key)}\b\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_float_metric(text: str, key: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(key)}\b\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1))
        if "%" in m.group(0) or val > 1.0:
            val = val / 100.0
        return val
    except Exception:
        return None


def parse_score_output(text: str) -> Dict[str, Optional[float]]:
    prefixes = ["ORIGINAL", "FLIPPED", "ALL_YES", "ALL_NO", "RANDOM"]
    fields = ["TP", "TN", "FP", "FN", "ACCURACY", "PRECISION", "RECALL", "F1"]

    out = {}
    for prefix in prefixes:
        for field in fields:
            key = f"{prefix}_{field}"
            if field in {"TP", "TN", "FP", "FN"}:
                out[key] = _parse_int_metric(text, key)
            else:
                out[key] = _parse_float_metric(text, key)
    return out


def format_pair(x: Optional[float], y: Optional[float], decimals: int = 6) -> str:
    def _one(v):
        if v is None:
            return "N/A"
        if isinstance(v, int):
            return str(v)
        return f"{v:.{decimals}f}"
    return f"{_one(x)}/{_one(y)}"


def format_single(x: Optional[float], decimals: int = 6) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{decimals}f}"


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


class PooledCLIP(nn.Module):
    def __init__(self, vit_name: str, bert_name: str, embed_dim: int, frame_pooling: str, sequence_pooling: str):
        super().__init__()
        if frame_pooling not in POOL_CHOICES:
            raise ValueError(f"frame_pooling must be one of {POOL_CHOICES}, got {frame_pooling}")
        if sequence_pooling not in POOL_CHOICES:
            raise ValueError(f"sequence_pooling must be one of {POOL_CHOICES}, got {sequence_pooling}")

        self.vit = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.bert_hidden = self.bert.config.hidden_size
        self.frame_pooling = frame_pooling
        self.sequence_pooling = sequence_pooling

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.vit_hidden),
            nn.GELU(),
            nn.Linear(self.vit_hidden, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert_hidden, self.bert_hidden),
            nn.GELU(),
            nn.Linear(self.bert_hidden, embed_dim),
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
        out = self.bert(**tok)
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

    @torch.no_grad()
    def encode_sequence_from_frames(
        self,
        processor,
        frame_paths: List[Path],
        device: torch.device,
        frame_chunk_size: int,
        max_frames: Optional[int],
        sequence_repeat_factor: int,
    ) -> Tuple[Optional[torch.Tensor], str]:
        frame_paths = uniform_subsample(frame_paths, max_frames)
        if not frame_paths:
            return None, "no_frames"

        collected_frame_feats = []

        for i in range(0, len(frame_paths), frame_chunk_size):
            chunk = frame_paths[i : i + frame_chunk_size]
            imgs: List[Image.Image] = []
            for p in chunk:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue
            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            out = self.vit(pixel_values=pixel_values)
            feats = out.last_hidden_state[:, 0, :]
            collected_frame_feats.append(feats)

            del inputs, pixel_values, out, imgs

        if not collected_frame_feats:
            return None, "no_readable_frames"

        all_frame_feats = torch.cat(collected_frame_feats, dim=0)
        sequence_feat = self._pool_tensor(all_frame_feats, self.frame_pooling)

        repeat_n = max(1, int(sequence_repeat_factor))
        repeated_sequences = sequence_feat.unsqueeze(0).repeat(repeat_n, 1)

        study_like_feat = self._pool_tensor(repeated_sequences, self.sequence_pooling)

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
                sequence_repeat_factor=args.sequence_repeat_factor,
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
    print(f"  Sequence repeat factor:  {args.sequence_repeat_factor}")
    print(f"  Sequence pooling:        {args.sequence_pooling}")
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
        "--random_seed",
        str(args.random_seed),
    ]
    cp = _run_subprocess_capture(score_cmd)

    combined_output = ""
    if cp.stdout:
        combined_output += cp.stdout
    if cp.stderr:
        combined_output += "\n" + cp.stderr

    metrics = parse_score_output(combined_output)

    print("\n[CHECKPOINT METRICS]")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  TP (orig/flip): {format_pair(metrics.get('ORIGINAL_TP'), metrics.get('FLIPPED_TP'), decimals=0)}")
    print(f"  TN (orig/flip): {format_pair(metrics.get('ORIGINAL_TN'), metrics.get('FLIPPED_TN'), decimals=0)}")
    print(f"  FP (orig/flip): {format_pair(metrics.get('ORIGINAL_FP'), metrics.get('FLIPPED_FP'), decimals=0)}")
    print(f"  FN (orig/flip): {format_pair(metrics.get('ORIGINAL_FN'), metrics.get('FLIPPED_FN'), decimals=0)}")
    print(f"  Accuracy (orig/flip): {format_pair(metrics.get('ORIGINAL_ACCURACY'), metrics.get('FLIPPED_ACCURACY'))}")
    print(f"  F1-score (orig/flip): {format_pair(metrics.get('ORIGINAL_F1'), metrics.get('FLIPPED_F1'))}")
    print(f"  ALL_YES Accuracy/F1: {format_single(metrics.get('ALL_YES_ACCURACY'))} / {format_single(metrics.get('ALL_YES_F1'))}")
    print(f"  ALL_NO  Accuracy/F1: {format_single(metrics.get('ALL_NO_ACCURACY'))} / {format_single(metrics.get('ALL_NO_F1'))}")
    print(f"  RANDOM  Accuracy/F1: {format_single(metrics.get('RANDOM_ACCURACY'))} / {format_single(metrics.get('RANDOM_F1'))}")

    return {
        "checkpoint": ckpt_path,
        "label": label,
        "pred_csv": temp_out_csv,
        "error_csv": temp_err_csv,
        "ORIGINAL_TP": metrics.get("ORIGINAL_TP"),
        "ORIGINAL_TN": metrics.get("ORIGINAL_TN"),
        "ORIGINAL_FP": metrics.get("ORIGINAL_FP"),
        "ORIGINAL_FN": metrics.get("ORIGINAL_FN"),
        "ORIGINAL_ACCURACY": metrics.get("ORIGINAL_ACCURACY"),
        "ORIGINAL_PRECISION": metrics.get("ORIGINAL_PRECISION"),
        "ORIGINAL_RECALL": metrics.get("ORIGINAL_RECALL"),
        "ORIGINAL_F1": metrics.get("ORIGINAL_F1"),
        "FLIPPED_TP": metrics.get("FLIPPED_TP"),
        "FLIPPED_TN": metrics.get("FLIPPED_TN"),
        "FLIPPED_FP": metrics.get("FLIPPED_FP"),
        "FLIPPED_FN": metrics.get("FLIPPED_FN"),
        "FLIPPED_ACCURACY": metrics.get("FLIPPED_ACCURACY"),
        "FLIPPED_PRECISION": metrics.get("FLIPPED_PRECISION"),
        "FLIPPED_RECALL": metrics.get("FLIPPED_RECALL"),
        "FLIPPED_F1": metrics.get("FLIPPED_F1"),
        "ALL_YES_TP": metrics.get("ALL_YES_TP"),
        "ALL_YES_TN": metrics.get("ALL_YES_TN"),
        "ALL_YES_FP": metrics.get("ALL_YES_FP"),
        "ALL_YES_FN": metrics.get("ALL_YES_FN"),
        "ALL_YES_ACCURACY": metrics.get("ALL_YES_ACCURACY"),
        "ALL_YES_PRECISION": metrics.get("ALL_YES_PRECISION"),
        "ALL_YES_RECALL": metrics.get("ALL_YES_RECALL"),
        "ALL_YES_F1": metrics.get("ALL_YES_F1"),
        "ALL_NO_TP": metrics.get("ALL_NO_TP"),
        "ALL_NO_TN": metrics.get("ALL_NO_TN"),
        "ALL_NO_FP": metrics.get("ALL_NO_FP"),
        "ALL_NO_FN": metrics.get("ALL_NO_FN"),
        "ALL_NO_ACCURACY": metrics.get("ALL_NO_ACCURACY"),
        "ALL_NO_PRECISION": metrics.get("ALL_NO_PRECISION"),
        "ALL_NO_RECALL": metrics.get("ALL_NO_RECALL"),
        "ALL_NO_F1": metrics.get("ALL_NO_F1"),
        "RANDOM_TP": metrics.get("RANDOM_TP"),
        "RANDOM_TN": metrics.get("RANDOM_TN"),
        "RANDOM_FP": metrics.get("RANDOM_FP"),
        "RANDOM_FN": metrics.get("RANDOM_FN"),
        "RANDOM_ACCURACY": metrics.get("RANDOM_ACCURACY"),
        "RANDOM_PRECISION": metrics.get("RANDOM_PRECISION"),
        "RANDOM_RECALL": metrics.get("RANDOM_RECALL"),
        "RANDOM_F1": metrics.get("RANDOM_F1"),
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
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    frame_pooling = args.frame_pooling if args.frame_pooling else args.pooling
    sequence_pooling = args.sequence_pooling if args.sequence_pooling else args.pooling

    validation_lookup = load_validation_question_lookup(args.validation_csv)

    model = PooledCLIP(
        vit_name=args.vit_name,
        bert_name=args.bert_name,
        embed_dim=args.embed_dim,
        frame_pooling=frame_pooling,
        sequence_pooling=sequence_pooling,
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

    valid_results = [r for r in results if r["ORIGINAL_ACCURACY"] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x["ORIGINAL_ACCURACY"])
    else:
        best = results[-1]
        print("[WARN] No ORIGINAL accuracy could be parsed from any calculate_score.py output.")
        print("[WARN] Falling back to the last evaluated checkpoint as best.")

    final_out_csv = Path(args.out_csv)
    final_out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["pred_csv"], final_out_csv)

    if args.error_csv and best["error_csv"] is not None and Path(best["error_csv"]).exists():
        final_error_csv = Path(args.error_csv)
        final_error_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["error_csv"], final_error_csv)

    print("\n" + "#" * 220)
    print("[FINAL CHECKPOINT COMPARISON]")
    print("#" * 220)
    print(
        f"{'Checkpoint':<20} "
        f"{'TP(O/F)':>14} "
        f"{'TN(O/F)':>14} "
        f"{'FP(O/F)':>14} "
        f"{'FN(O/F)':>14} "
        f"{'Accuracy(O/F)':>20} "
        f"{'F1(O/F)':>20} "
        f"{'Pred CSV':<40}"
    )

    for r in results:
        print(
            f"{r['checkpoint'].name:<20} "
            f"{format_pair(r.get('ORIGINAL_TP'), r.get('FLIPPED_TP'), decimals=0):>14} "
            f"{format_pair(r.get('ORIGINAL_TN'), r.get('FLIPPED_TN'), decimals=0):>14} "
            f"{format_pair(r.get('ORIGINAL_FP'), r.get('FLIPPED_FP'), decimals=0):>14} "
            f"{format_pair(r.get('ORIGINAL_FN'), r.get('FLIPPED_FN'), decimals=0):>14} "
            f"{format_pair(r.get('ORIGINAL_ACCURACY'), r.get('FLIPPED_ACCURACY')):>20} "
            f"{format_pair(r.get('ORIGINAL_F1'), r.get('FLIPPED_F1')):>20} "
            f"{r['pred_csv'].name:<40}"
        )

    if results:
        baseline_source = results[0]

        print(
            f"{'ALL_YES':<20} "
            f"{format_single(baseline_source.get('ALL_YES_TP'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_YES_TN'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_YES_FP'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_YES_FN'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_YES_ACCURACY')):>20} "
            f"{format_single(baseline_source.get('ALL_YES_F1')):>20} "
            f"{'-':<40}"
        )

        print(
            f"{'ALL_NO':<20} "
            f"{format_single(baseline_source.get('ALL_NO_TP'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_NO_TN'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_NO_FP'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_NO_FN'), decimals=0):>14} "
            f"{format_single(baseline_source.get('ALL_NO_ACCURACY')):>20} "
            f"{format_single(baseline_source.get('ALL_NO_F1')):>20} "
            f"{'-':<40}"
        )

        print(
            f"{'RANDOM':<20} "
            f"{format_single(baseline_source.get('RANDOM_TP'), decimals=0):>14} "
            f"{format_single(baseline_source.get('RANDOM_TN'), decimals=0):>14} "
            f"{format_single(baseline_source.get('RANDOM_FP'), decimals=0):>14} "
            f"{format_single(baseline_source.get('RANDOM_FN'), decimals=0):>14} "
            f"{format_single(baseline_source.get('RANDOM_ACCURACY')):>20} "
            f"{format_single(baseline_source.get('RANDOM_F1')):>20} "
            f"{'-':<40}"
        )

    print("\n[BEST CHECKPOINT]")
    print(f"  Checkpoint: {best['checkpoint']}")
    print(f"  TP (orig/flip): {format_pair(best.get('ORIGINAL_TP'), best.get('FLIPPED_TP'), decimals=0)}")
    print(f"  TN (orig/flip): {format_pair(best.get('ORIGINAL_TN'), best.get('FLIPPED_TN'), decimals=0)}")
    print(f"  FP (orig/flip): {format_pair(best.get('ORIGINAL_FP'), best.get('FLIPPED_FP'), decimals=0)}")
    print(f"  FN (orig/flip): {format_pair(best.get('ORIGINAL_FN'), best.get('FLIPPED_FN'), decimals=0)}")
    print(f"  Accuracy (orig/flip): {format_pair(best.get('ORIGINAL_ACCURACY'), best.get('FLIPPED_ACCURACY'))}")
    print(f"  F1-score (orig/flip): {format_pair(best.get('ORIGINAL_F1'), best.get('FLIPPED_F1'))}")
    print(f"  Best checkpoint selected by ORIGINAL accuracy only.")
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
        default="/data/Deep_Angiography/Validation_Data/test-data/gt.csv",
        type=str,
        help="Validation CSV containing ground-truth rows. Only questions present here will be asked per SOPInstanceUID.",
    )

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--vit_name", default="google/vit-base-patch16-224-in21k", type=str)
    ap.add_argument("--bert_name", default="bert-base-uncased", type=str)
    ap.add_argument("--embed_dim", default=256, type=int)

    ap.add_argument("--pooling", default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", default="", choices=("",) + POOL_CHOICES)
    ap.add_argument("--sequence_pooling", default="", choices=("",) + POOL_CHOICES)

    ap.add_argument(
        "--sequence_repeat_factor",
        default=16,
        type=int,
        help="Repeat the pooled sequence representation this many times before final sequence-level pooling.",
    )

    ap.add_argument("--frame_chunk_size", default=64, type=int)
    ap.add_argument("--max_frames", default=0, type=int)

    ap.add_argument(
        "--calculate_score_script",
        default="calculate_score.py",
        type=str,
        help="Path to calculate_score.py script to run after validation.",
    )

    ap.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed used for the RANDOM baseline in calculate_score.py.",
    )

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)