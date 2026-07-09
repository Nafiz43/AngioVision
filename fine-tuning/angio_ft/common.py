"""
angio_ft.common
────────────────
Shared, dependency-light building blocks used by both training and validation:

  • SOPInstanceUIDs parsing
  • frame-file discovery on disk
  • pooling helpers (max / mean / logsumexp)
  • deterministic sinusoidal temporal encoding
  • ViT image-processor loading + native-resolution auto-detection
  • small string / path normalisation helpers

Every function here is lifted verbatim (behaviour-preserving) from the original
``custom_framework_train_temporal.py`` / ``custom_framework_validate.py`` scripts
so that the unified pipeline is a drop-in replacement with identical numerics.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .constants import POOL_CHOICES, TEMPORAL_MODE_CHOICES  # noqa: F401 (re-exported)

# ── Transformers image-processor compat ──────────────────────────────────────
try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:  # pragma: no cover - depends on transformers version
    _ViTProcessor = None

try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:  # pragma: no cover
    _ViTFeatureExtractor = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# SOPInstanceUIDs parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_sop_instance_uids(val) -> List[str]:
    """Parse a SOPInstanceUIDs cell into a list of UID strings.

    Accepts ``"uid1,uid2"``, ``"['uid1','uid2']"`` and ``"('uid1','uid2')"``.
    """
    if val is None:
        return []
    try:
        import pandas as pd  # local import keeps common.py import-light
        if isinstance(val, float) and pd.isna(val):
            return []
    except Exception:
        if isinstance(val, float) and val != val:  # NaN fallback
            return []
    s = str(val).strip()
    if len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
    ):
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
    return [tok.strip() for tok in re.split(r"\s*,\s*", s) if tok.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Frame discovery
# ─────────────────────────────────────────────────────────────────────────────

def list_images_in_dir(d: Path) -> List[Path]:
    """Return sorted image paths in *d*. Sorting is explicit for reproducibility."""
    if not d.exists() or not d.is_dir():
        return []
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)  # explicit sort - os.listdir order is not guaranteed


def find_frame_files_for_sop(base_frames_dir: Path, acc: str, sop_uid: str) -> List[Path]:
    """Probe filesystem for frame images for a single SOP UID.

    Called once per (acc, sop_uid) pair during dataset initialisation; results
    are cached so ``__getitem__`` never touches the filesystem.
    """
    acc = str(acc).strip()
    sop_uid = str(sop_uid).strip()

    sop_dir = base_frames_dir / acc / sop_uid

    # Priority 1 - canonical frames/ sub-dir
    imgs = list_images_in_dir(sop_dir / "frames")
    if imgs:
        return imgs

    # Priority 2 - images directly inside the SOP dir
    imgs = list_images_in_dir(sop_dir)
    if imgs:
        return imgs

    # Priority 3 - any single-level sub-dir
    if sop_dir.exists() and sop_dir.is_dir():
        nested: List[Path] = []
        for child in sop_dir.iterdir():
            if child.is_dir():
                nested.extend(list_images_in_dir(child))
        if nested:
            return sorted(nested)

    # Priority 4 - glob pattern fallback
    try:
        for candidate in base_frames_dir.glob(f"*/{acc}/{sop_uid}/frames"):
            imgs = list_images_in_dir(candidate)
            if imgs:
                return imgs
    except Exception:
        pass

    return []


# ── Validation-side frame helpers ────────────────────────────────────────────

def find_frames_dir(seq_dir: Path) -> Optional[Path]:
    for name in ("frames", "Frames"):
        d = seq_dir / name
        if d.exists() and d.is_dir():
            return d
    return None


def uniform_subsample(paths: List[Path], max_frames: Optional[int]) -> List[Path]:
    """Uniformly subsample *paths* to at most *max_frames*, preserving order."""
    if max_frames is None or max_frames <= 0 or len(paths) <= max_frames:
        return paths
    idxs = torch.linspace(0, len(paths) - 1, steps=max_frames).long().tolist()
    return [paths[i] for i in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# String normalisation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Pooling helpers
# ─────────────────────────────────────────────────────────────────────────────

def pool_stack(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool a stacked ``(N, D)`` tensor down to ``(D,)`` using *mode*."""
    if x.ndim != 2:
        raise ValueError(f"pool_stack expects (N,D), got {tuple(x.shape)}")
    if x.size(0) == 0:
        raise ValueError("pool_stack got empty N dimension")
    if mode == "max":
        return x.max(dim=0).values
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "logsumexp":
        return torch.logsumexp(x, dim=0)
    raise ValueError(f"Unknown pooling mode: {mode}. Choose from {POOL_CHOICES}.")


# ─────────────────────────────────────────────────────────────────────────────
# Temporal encoding (deterministic - adds NO learnable parameters, so temporal
# on/off checkpoints remain mutually loadable)
# ─────────────────────────────────────────────────────────────────────────────

def build_sinusoidal_position_encoding(
    positions: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """positions: (N,)  ->  returns (N, dim)."""
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got {tuple(positions.shape)}")
    pos = positions.to(device=device, dtype=torch.float32)
    pe = torch.zeros((pos.size(0), dim), device=device, dtype=torch.float32)
    if dim == 1:
        pe[:, 0] = pos
        return pe.to(dtype)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
    cos_w = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term[:cos_w])
    return pe.to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# ViT processor + native-resolution resolution
# ─────────────────────────────────────────────────────────────────────────────

# Vision checkpoints whose preprocessing must NOT be forced through
# ViTImageProcessor (their preprocessor configs use different classes/keys).
_NON_VIT_MODEL_TYPES = {
    "siglip", "siglip2", "siglip_vision_model", "siglip2_vision_model",
    "clip", "clip_vision_model",
    "xclip", "xclip_vision_model",
}

# X-CLIP checkpoints ship a VideoMAE-style processor that returns 5-D video
# tensors; per-frame preprocessing is CLIP-style, so a CLIPImageProcessor
# loaded from the same repo reproduces it while keeping 4-D (N,C,H,W) output.
_XCLIP_MODEL_TYPES = {"xclip", "xclip_vision_model"}


def vision_model_type(vit_name: str) -> str:
    """Return the HF ``model_type`` for a vision checkpoint ('' if unknown)."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(vit_name)
        return str(getattr(cfg, "model_type", "") or "")
    except Exception:
        return ""


def get_vit_processor(vit_name: str):
    # SigLIP / SigLIP2 / CLIP checkpoints ship their own image processors;
    # forcing ViTImageProcessor onto them mangles normalisation. Plain ViT /
    # DINO checkpoints keep the legacy ViTImageProcessor path so numerics of
    # existing runs are unchanged.
    mtype = vision_model_type(vit_name)
    if mtype in _XCLIP_MODEL_TYPES:
        from transformers import CLIPImageProcessor
        return CLIPImageProcessor.from_pretrained(vit_name)
    if mtype in _NON_VIT_MODEL_TYPES:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(vit_name)
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(vit_name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(vit_name)
    raise ImportError("Neither ViTImageProcessor nor ViTFeatureExtractor found.")


def resolve_vit_image_size(vit_name: str, override: Optional[int] = None) -> int:
    """Return the image size the ViT model was trained with.

    Resolution priority:
      1. Explicit ``--vit_image_size`` CLI override (always wins).
      2. ``image_size`` field inside the processor / feature-extractor config.
      3. ``image_size`` field on the ViT model config itself.
      4. Hard-coded per-model fallback table for well-known checkpoints.
      5. Safe default of 224 with a visible warning.

    Prevents the ``Input image size (X*X) doesn't match model (Y*Y)`` error that
    occurs when the processor default (224) differs from the model's native
    resolution (e.g. ``microsoft/rad-dino`` expects 518x518).
    """
    if override is not None:
        print(f"[INFO] vit_image_size: using CLI override = {override}")
        return override

    # ── 1. Try to read from the processor config ──────────────────────────
    try:
        proc = get_vit_processor(vit_name)
        if proc is not None:
            size_val = getattr(proc, "size", None)
            if isinstance(size_val, dict):
                sz = size_val.get("height") or size_val.get("shortest_edge") or next(iter(size_val.values()), None)
                if sz is not None:
                    print(f"[INFO] vit_image_size: auto-detected {sz} from processor config ({vit_name})")
                    return int(sz)
            elif isinstance(size_val, int) and size_val > 0:
                print(f"[INFO] vit_image_size: auto-detected {size_val} from processor config ({vit_name})")
                return size_val
            crop = getattr(proc, "crop_size", None)
            if isinstance(crop, dict):
                sz = crop.get("height") or crop.get("width")
                if sz is not None:
                    print(f"[INFO] vit_image_size: auto-detected {sz} from processor crop_size ({vit_name})")
                    return int(sz)
    except Exception as e:
        print(f"[WARN] Could not read image size from processor config: {e}")

    # ── 2. Try to read from the ViT model config ──────────────────────────
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(vit_name)
        # Composite VLM configs (SigLIP/SigLIP2/CLIP) nest the vision settings.
        cfg_candidates = [cfg]
        if getattr(cfg, "vision_config", None) is not None:
            cfg_candidates.insert(0, cfg.vision_config)
        for c in cfg_candidates:
            for attr in ("image_size", "input_size"):
                val = getattr(c, attr, None)
                if val is not None:
                    sz = val[0] if isinstance(val, (list, tuple)) else int(val)
                    print(f"[INFO] vit_image_size: auto-detected {sz} from model config.{attr} ({vit_name})")
                    return sz
    except Exception as e:
        print(f"[WARN] Could not read image size from model config: {e}")

    # ── 3. Hard-coded fallback table ──────────────────────────────────────
    _KNOWN_SIZES: Dict[str, int] = {
        "microsoft/rad-dino":                       518,
        "microsoft/rad-dino-8b":                    518,
        "facebook/dinov2-base":                     518,
        "facebook/dinov2-large":                    518,
        "facebook/dinov2-giant":                    518,
        "google/vit-base-patch16-224":              224,
        "google/vit-base-patch16-224-in21k":        224,
        "google/vit-large-patch16-224":             224,
        "google/vit-huge-patch14-224-in21k":        224,
        "google/vit-base-patch32-384":              384,
        "google/vit-large-patch32-384":             384,
        "WinKawaks/vit-small-patch16-224":          224,
    }
    vit_lower = vit_name.lower()
    for key, sz in _KNOWN_SIZES.items():
        if key.lower() in vit_lower or vit_lower in key.lower():
            print(f"[INFO] vit_image_size: matched fallback table entry '{key}' -> {sz}")
            return sz

    # ── 4. Final safe default ─────────────────────────────────────────────
    default = 224
    print(
        f"[WARN] vit_image_size: could not auto-detect for '{vit_name}'. "
        f"Defaulting to {default}. Pass --vit_image_size if this is wrong."
    )
    return default
