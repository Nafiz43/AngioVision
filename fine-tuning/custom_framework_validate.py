#!/usr/bin/env python3
"""
custom_framework_validate.py

Run CLIP-style binary QA on validation sequence directories using one or more trained checkpoints.

DATA LAYOUT (your validation set):
  DATA_DIR/
    <sequence_dir_1>/
      frames/              # images
      metadata.csv         # key-value CSV: columns [Information, Value]
      mosaic.png           # optional
    <sequence_dir_2>/
      ...

metadata.csv format (IMPORTANT):
  Information,Value
  SOPInstanceUID,1.2.276...
  AccessionNumber,202510081160
  ...

OUTPUT:
- --out_csv: final/best prediction CSV
- --error_csv (optional): final/best error CSV

Binary QA method:
- For each question, we build YES/NO hypothesis texts
- Compare cosine similarity between image embedding and text embeddings
- Choose YES if sim_yes > sim_no else NO

NEW:
- --checkpoint can be either:
    1) a single checkpoint file (.pt), or
    2) a run directory containing epoch checkpoints
- If a directory is given:
    * evaluate all epoch_*.pt checkpoints
    * if none exist, evaluate last.pt if available
- For each checkpoint:
    * validation is run
    * calculate_score.py is run as a subprocess
    * stdout/stderr is parsed to extract TP/TN/FP/FN/Accuracy/F1
- The checkpoint with the best accuracy is reported
- The best checkpoint's prediction CSV is copied to --out_csv
- The best checkpoint's error CSV is copied to --error_csv if requested

EFFICIENCY UPDATE:
- Only asks questions for sequences whose SOPInstanceUID appears in --validation_csv
- Only asks the specific questions listed for that SOPInstanceUID in the validation CSV
- Sequences not present in validation CSV are skipped
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
    """
    Tries to parse TP, TN, FP, FN, Accuracy, and F1-score
    from calculate_score.py stdout/stderr.

    Returns:
        {
            "tp": int or None,
            "tn": int or None,
            "fp": int or None,
            "fn": int or None,
            "accuracy": float in [0,1] or None,
            "f1": float in [0,1] or None,
        }
    """
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
    """
    Build lookup:
      SOPInstanceUID -> list of questions present in GT

    Only those questions will be asked for each sequence.
    """
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
    """
    Reads seq_dir/metadata.csv where format is:
      Information,Value
      SOPInstanceUID,....
      AccessionNumber,....

    Returns: (AccessionNumber, SOPInstanceUID, status)
      status = "ok" if successful else reason
    """
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
    """
    If checkpoint_arg is:
      - a file: return [file]
      - a directory:
          * return sorted epoch_*.pt if any
          * else return [last.pt] if available
    """
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


# -----------------------------
# Model (must match your training architecture)
# -----------------------------
class PooledCLIP(nn.Module):
    """
    Inference version matching training:
    ViT + projection, BERT + projection, CLIP-style embedding space.
    Validation: each sequence dir is one "study" (single sequence), so we pool only over frames.
    """

    def __init__(self, vit_name: str, bert_name: str, embed_dim: int, frame_pooling: str):
        super().__init__()
        if frame_pooling not in POOL_CHOICES:
            raise ValueError(f"frame_pooling must be one of {POOL_CHOICES}, got {frame_pooling}")

        self.vit = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.bert_hidden = self.bert.config.hidden_size
        self.frame_pooling = frame_pooling

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

    @torch.no_grad()
    def encode_sequence_from_frames(
        self,
        processor,
        frame_paths: List[Path],
        device: torch.device,
        frame_chunk_size: int,
        max_frames: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], str]:
        frame_paths = uniform_subsample(frame_paths, max_frames)
        if not frame_paths:
            return None, "no_frames"

        if self.frame_pooling == "max":
            running = torch.full((self.vit_hidden,), -1e9, device=device)
            updated = False
        elif self.frame_pooling == "mean":
            running = torch.zeros((self.vit_hidden,), device=device)
            count = 0
        else:
            running = torch.full((self.vit_hidden,), -float("inf"), device=device)
            updated = False

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

            if self.frame_pooling == "max":
                running = torch.maximum(running, feats.max(dim=0).values)
                updated = True
            elif self.frame_pooling == "mean":
                running = running + feats.sum(dim=0)
                count += feats.size(0)
            else:
                running = torch.logaddexp(running, torch.logsumexp(feats, dim=0))
                updated = True

            del inputs, pixel_values, out, feats, imgs

        if self.frame_pooling == "mean":
            if count <= 0:
                return None, "no_readable_frames"
            running = running / float(count)
        else:
            if not updated:
                return None, "no_readable_frames"

        emb = F.normalize(self.vision_proj(running.unsqueeze(0)), dim=-1)
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
    """
    Runs prediction generation + calculate_score.py for one checkpoint.
    Returns dict with paths and parsed metrics.
    """
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
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    frame_pooling = args.frame_pooling if args.frame_pooling else args.pooling

    validation_lookup = load_validation_question_lookup(args.validation_csv)

    model = PooledCLIP(
        vit_name=args.vit_name,
        bert_name=args.bert_name,
        embed_dim=args.embed_dim,
        frame_pooling=frame_pooling,
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
        default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed",
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
    ap.add_argument("--bert_name", default="bert-base-uncased", type=str)
    ap.add_argument("--embed_dim", default=256, type=int)

    ap.add_argument("--pooling", default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", default="", choices=("",) + POOL_CHOICES)

    ap.add_argument("--frame_chunk_size", default=64, type=int)
    ap.add_argument("--max_frames", default=0, type=int)

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