"""Sequence directory discovery, mosaic loading, and UID-based mosaic resolution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------- Directory discovery ----------

def find_sequence_dirs(base_path: Path, frames_subdir: str = "frames") -> List[Path]:
    """Find all directories under *base_path* that contain a *frames_subdir*
    sub-directory with at least one image file.
    """
    seq_dirs: List[Path] = []
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists():
            continue
        try:
            if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()):
                seq_dirs.append(d)
        except PermissionError:
            continue
    return sorted(seq_dirs, key=lambda p: p.as_posix())


# ---------- SequenceMosaicInfo ----------

@dataclass
class SequenceMosaicInfo:
    """Metadata about a sequence directory and its mosaic image."""
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    ok: bool
    error: Optional[str] = None


def load_mosaics(
    seq_dirs: List[Path],
    base_path: Path,
    mosaic_name: str = "mosaic.png",
) -> List[SequenceMosaicInfo]:
    """Check which *seq_dirs* contain a mosaic and return structured info."""
    infos: List[SequenceMosaicInfo] = []
    for d in seq_dirs:
        rel = d.relative_to(base_path).as_posix()
        mp = d / mosaic_name
        infos.append(
            SequenceMosaicInfo(
                seq_dir=d,
                seq_rel=rel,
                mosaic_path=mp,
                ok=mp.exists(),
                error=None if mp.exists() else "Missing mosaic",
            )
        )
    return infos


# ---------- UID-based mosaic resolution ----------

@dataclass
class ResolvedMosaic:
    """Result of resolving a UID to its mosaic image on disk."""
    uid: str
    seq_dir: Optional[Path]
    mosaic_path: Optional[Path]
    ok: bool
    error: Optional[str] = None


def resolve_uid_dir(base_path: Path, uid: str) -> Optional[Path]:
    """Try ``base_path/uid`` first, then fall back to an ``rglob`` search."""
    direct = base_path / uid
    if direct.exists() and direct.is_dir():
        return direct

    try:
        for p in base_path.rglob("*"):
            if p.is_dir() and p.name == uid:
                return p
    except Exception:
        return None
    return None


def resolve_mosaic_for_uid(
    base_path: Path,
    uid: str,
    mosaic_name: str = "mosaic.png",
    mosaic_relative_dir: str = "",
) -> ResolvedMosaic:
    """Locate the mosaic image for a given *uid* under *base_path*."""
    uid_dir = resolve_uid_dir(base_path, uid)
    if not uid_dir:
        return ResolvedMosaic(uid=uid, seq_dir=None, mosaic_path=None, ok=False, error="UID directory not found")

    candidate = uid_dir / mosaic_relative_dir / mosaic_name if mosaic_relative_dir else uid_dir / mosaic_name
    if candidate.exists():
        return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=candidate, ok=True, error=None)

    try:
        hits = list(uid_dir.rglob(mosaic_name))
        if hits:
            return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=hits[0], ok=True, error=None)
    except Exception:
        pass

    return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=None, ok=False, error=f"Missing {mosaic_name}")


# ---------- Resume helpers ----------

def load_already_done_indices(out_path: Path, index_col: str = "input_row_index") -> Set[int]:
    """Return the set of already-processed row indices from an existing output CSV."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    try:
        done = pd.read_csv(out_path, usecols=[index_col])
        vals = done[index_col].dropna().astype(int).tolist()
        return set(vals)
    except Exception:
        return set()
