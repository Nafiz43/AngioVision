"""Data utilities: SOP-UID parsing, frame discovery, image preprocessing,
holdout-study loading. Mirrors the training script's data handling."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image

try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:
    _ViTProcessor = None
try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:
    _ViTFeatureExtractor = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_sop_instance_uids(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    s = str(val).strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[0] == s[-1]:
        s = s[1:-1].strip()
    if not s:
        return []
    if s[0] in "[(":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [t.strip() for t in re.split(r"\s*,\s*", s) if t.strip()]


def _list_images(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_frame_files(base: Path, acc: str, sop: str) -> List[Path]:
    sop_dir = base / str(acc).strip() / str(sop).strip()
    for cand in [sop_dir / "frames", sop_dir]:
        imgs = _list_images(cand)
        if imgs:
            return imgs
    if sop_dir.is_dir():
        nested: List[Path] = []
        for ch in sop_dir.iterdir():
            if ch.is_dir():
                nested.extend(_list_images(ch))
        if nested:
            return sorted(nested)
    return []


def get_vit_processor(name: str):
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(name)
    raise ImportError("No ViT image processor available in transformers.")


def preprocess_frames(
    frame_files: List[Path],
    processor,
    vit_image_size: Optional[int],
) -> Tuple[Optional[torch.Tensor], List[int]]:
    imgs, valid = [], []
    for i, p in enumerate(frame_files):
        try:
            imgs.append(Image.open(p).convert("RGB"))
            valid.append(i)
        except Exception:
            continue
    if not imgs:
        return None, []
    kw: Dict[str, Any] = dict(images=imgs, return_tensors="pt")
    if vit_image_size:
        kw["size"] = {"height": vit_image_size, "width": vit_image_size}
    pv = processor(**kw)["pixel_values"]
    for img in imgs:
        try:
            img.close()
        except Exception:
            pass
    return pv, valid


def load_holdout_studies(
    meta_csv: Path,
    anon_col: str,
    sop_col: str,
    base_dir: Path,
) -> List[Dict]:
    df = pd.read_csv(meta_csv)
    studies = []
    for _, row in df.iterrows():
        acc = str(row.get(anon_col, "")).strip()
        if not acc:
            continue
        sops = parse_sop_instance_uids(row.get(sop_col, ""))
        seq_info = [
            {"sop": s, "n_frames": len(find_frame_files(base_dir, acc, s))}
            for s in sops
        ]
        studies.append({"acc": acc, "sop_uids": sops, "seq_info": seq_info})
    return studies
