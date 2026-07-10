"""DICOM pixel loading and mode-aware frame extraction.

Frame modes:
  best  only the annotated Best_Image frame (middle frame if unannotated/out of range)
  fl    First_Diag_Image → Last_Diag_Image window (indices clamped to the file)
  all   every frame, optionally uniformly subsampled to --max-frames

Heavy deps (pydicom, numpy, PIL) are imported at module level — the CLI
checks their presence up front; only import this module when they exist.
"""

import io
import itertools
import logging
from typing import Optional

import numpy as np
import pydicom
from PIL import Image as PILImage

log = logging.getLogger(__name__)


def _to_uint8_rgb(frame: np.ndarray, photometric: str) -> np.ndarray:
    f = frame.astype(np.float32); lo, hi = f.min(), f.max()
    f = ((f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f)).astype(np.uint8)
    if "MONOCHROME1" in photometric.upper():
        f = 255 - f
    return np.stack([f, f, f], axis=-1)


def _load_pixels(path_str: str) -> tuple[Optional[np.ndarray], str]:
    """Read a DICOM file → (frames array [N,H,W(,C)], photometric interpretation)."""
    try:
        ds = pydicom.dcmread(path_str, force=False)
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        pixels = ds.pixel_array
        if pixels.ndim == 2:
            pixels = pixels[np.newaxis]
        elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
            pixels = pixels[np.newaxis]
        return pixels, photometric
    except Exception as e:
        log.debug(f"DICOM read failed [{path_str}]: {e}")
        return None, ""


def _raw_to_rgb(raw: np.ndarray, photometric: str) -> np.ndarray:
    if raw.ndim == 3:
        f = raw.astype(np.float32); lo, hi = f.min(), f.max()
        f = ((f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f)).astype(np.uint8)
        return f[:, :, :3]
    return _to_uint8_rgb(raw, photometric)


def _extract_best_frame(path_str: str, frame_idx: int) -> list[np.ndarray]:
    pixels, photometric = _load_pixels(path_str)
    if pixels is None:
        return []
    total = pixels.shape[0]
    idx = total // 2 if frame_idx < 0 or frame_idx >= total else frame_idx
    return [_raw_to_rgb(pixels[idx], photometric)]


def _extract_frame_range(path_str: str, first_idx: int, last_idx: int) -> list[np.ndarray]:
    pixels, photometric = _load_pixels(path_str)
    if pixels is None:
        return []
    total = pixels.shape[0]
    lo = max(0, min(first_idx, total - 1)); hi = max(0, min(last_idx, total - 1))
    if hi < lo:
        hi = lo
    return [_raw_to_rgb(pixels[i], photometric) for i in range(lo, hi + 1)]


def _extract_all_frames(path_str: str, max_frames: int = 0) -> list[np.ndarray]:
    pixels, photometric = _load_pixels(path_str)
    if pixels is None:
        return []
    total = pixels.shape[0]
    if max_frames > 0 and total > max_frames:
        indices = ([total // 2] if max_frames == 1 else
                   sorted(set(round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames))))
    else:
        indices = list(range(total))
    return [_raw_to_rgb(pixels[i], photometric) for i in indices]


def extract_frames_for_seq(seq: dict, mode: str, max_frames: int = 0) -> list[np.ndarray]:
    """Extract RGB uint8 frames for one sequence according to *mode*."""
    if mode == "best":
        return _extract_best_frame(seq["dicom_path"], seq.get("best_image_idx", -1))
    elif mode == "fl":
        return _extract_frame_range(seq["dicom_path"], seq["first_diag_idx"], seq["last_diag_idx"])
    else:
        return _extract_all_frames(seq["dicom_path"], max_frames)


def get_display_frame_png(seq: dict, size: int = 256) -> Optional[bytes]:
    """Load one representative frame as PNG bytes for docx thumbnails."""
    pixels, photometric = _load_pixels(seq["dicom_path"])
    if pixels is None:
        return None
    total = pixels.shape[0]; best_idx = seq.get("best_image_idx", -1)
    idx = best_idx if 0 <= best_idx < total else total // 2
    frame = _raw_to_rgb(pixels[idx], photometric)
    img = PILImage.fromarray(frame).resize((size, size), PILImage.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


# ── Threading helpers ─────────────────────────────────────────────────────────

def chunked(iterable, size: int):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def make_frame_reader(mode: str, max_frames: int = 0):
    """Bind (mode, max_frames) into a seq → (seq, frames) callable for thread pools."""
    def reader(seq: dict) -> tuple[dict, list]:
        return seq, extract_frames_for_seq(seq, mode, max_frames)
    return reader
