"""
Shared helpers for the visual-data-preparation pipeline.

Everything the original utils/ scripts each carried their own copy of
(DICOM header helpers, leaf-dir discovery, output-path construction,
pixel windowing, frame saving, eligibility filters, CSV/report writers,
plotting) lives here exactly once.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image
from pydicom.multival import MultiValue
from pydicom.pixels import get_decoder

NA_VALUE = "NA"
MAX_VALUE_CHARS = 2000
FRAME_FORMAT = "png"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

SKIP_KEYWORDS = {
    "PixelData", "WaveformData", "OverlayData",
    "EncapsulatedDocument", "CurveData", "AudioSampleData",
}
SKIP_TAGS = {(0x7FE0, 0x0010)}  # PixelData

REQUIRED_RADIATION_SETTING = "GR"
REQUIRED_POSITIONER_MOTION = "STATIC"
SERIES_DESCRIPTION_KEYWORDS = ("DSA", "CO 2")


# =========================================================
# DICOM header helpers
# =========================================================
def is_probably_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except Exception:
        return False
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def get_tag_str(ds, tag: str) -> str:
    val = getattr(ds, tag, None)
    if val is None:
        return ""
    if isinstance(val, MultiValue):
        val = val[0] if val else ""
    return str(val).strip()


def is_nullish(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, tuple, MultiValue)) and len(v) == 0:
        return True
    return False


def safe_str(x: object) -> str:
    try:
        return str(x)
    except Exception:
        return NA_VALUE


def normalize_value(value) -> str:
    if is_nullish(value):
        return NA_VALUE
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary:{len(value)} bytes>"
    if isinstance(value, MultiValue):
        s = ";".join(safe_str(v) for v in value if v is not None).strip()
        return s or NA_VALUE
    s = safe_str(value).strip()
    if not s:
        return NA_VALUE
    if len(s) > MAX_VALUE_CHARS:
        s = s[:MAX_VALUE_CHARS] + "...<truncated>"
    return s


def extract_metadata_pairs(ds) -> List[Dict[str, str]]:
    rows = []
    for elem in ds:
        if elem.VR == "SQ":
            continue
        if elem.keyword in SKIP_KEYWORDS:
            continue
        if (int(elem.tag.group), int(elem.tag.element)) in SKIP_TAGS:
            continue
        if is_nullish(elem.value):
            continue
        key = elem.keyword if elem.keyword else str(elem.tag)
        rows.append({"Information": key, "Value": normalize_value(elem.value)})
    return rows


# =========================================================
# Filesystem helpers
# =========================================================
def sanitize_dirname(name: str, max_len: int = 150) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return (name or "unknown")[:max_len]


def collect_leaf_dirs(root_dir: Path) -> List[Path]:
    """Directories that directly contain at least one file. Units of parallel work."""
    leaf_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if filenames:
            leaf_dirs.append(Path(dirpath))
    return leaf_dirs


def output_dir_for(output_root: Path, accession_number: str, sop_instance_uid: str) -> Path:
    acc = sanitize_dirname(accession_number) if accession_number else "NO_ACCESSION"
    sop = sop_instance_uid if sop_instance_uid else "NO_UID"
    return output_root / acc / sop


def dest_already_exists(output_root: Path, accession_number: str, sop_instance_uid: str) -> bool:
    d = output_dir_for(output_root, accession_number, sop_instance_uid)
    return (d / "metadata.csv").exists() and (d / "frames").exists()


def find_sequence_dirs(root: Path) -> List[Path]:
    """Every directory that owns a frames/ subdir containing at least one image."""
    seq_dirs = []
    for frames_dir in root.rglob("frames"):
        if not frames_dir.is_dir():
            continue
        try:
            has_image = any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                for p in frames_dir.iterdir()
            )
        except PermissionError:
            continue
        if has_image:
            seq_dirs.append(frames_dir.parent)
    return sorted(seq_dirs, key=lambda p: p.as_posix())


# =========================================================
# Eligibility filter (step 01)
# =========================================================
def passes_eligibility_filter(ds, min_frames: int, mode: str) -> Tuple[bool, str]:
    """
    strict : RadiationSetting GR → SeriesDescription DSA/CO 2 →
             PositionerMotion STATIC → NumberOfFrames > min_frames
    relaxed: same, without the SeriesDescription step
    """
    def _get(tag: str) -> str:
        return get_tag_str(ds, tag).upper()

    # ponytail: relaxed radiation gate — reject only an explicit non-GR setting
    # (e.g. SC fluoro). A missing/empty RadiationSetting tag passes through to the
    # downstream DSA-mask classifier (s06), which is the real DSA detector. This
    # recovered ~1.8k accessions whose DICOMs simply lack the tag. To restore the
    # strict gate, require == REQUIRED_RADIATION_SETTING again.
    rad = _get("RadiationSetting")
    if rad and rad != REQUIRED_RADIATION_SETTING:
        return False, "bad_radiation"

    if mode == "strict":
        sd = _get("SeriesDescription")
        if not any(kw.upper() in sd for kw in SERIES_DESCRIPTION_KEYWORDS):
            return False, "bad_series"

    if _get("PositionerMotion") != REQUIRED_POSITIONER_MOTION:
        return False, "bad_motion"

    nof_raw = getattr(ds, "NumberOfFrames", None)
    if nof_raw is not None:
        try:
            if int(str(nof_raw).strip()) <= min_frames:
                return False, "too_few_frames"
        except (ValueError, TypeError):
            pass

    return True, "ok"


# =========================================================
# Pixel conversion + frame saving (step 01)
# =========================================================
def to_uint8_windowed(arr: np.ndarray, ds) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if isinstance(center, MultiValue):
        center = center[0]
    if isinstance(width, MultiValue):
        width = width[0]

    if center is not None and width is not None and width > 0:
        lo = center - width / 2
        hi = center + width / 2
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
        arr = 1.0 - arr

    return (arr * 255).astype(np.uint8)


def configure_pixel_backend(ds, backend: str) -> None:
    """Force pixel-data decoding through a specific plugin so a pydicom-only
    vs GDCM comparison run is meaningful regardless of which packages happen
    to be installed. 'gdcm' forces the GDCM plugin; 'pydicom' forces
    whichever non-GDCM plugin pydicom already ships with (pylibjpeg/pillow/
    native), so re-running the 'pydicom' backend after python-gdcm gets
    installed for the other pass still reproduces the original behavior."""
    if backend == "gdcm":
        ds.pixel_array_options(decoding_plugin="gdcm")
        return
    try:
        available = get_decoder(ds.file_meta.TransferSyntaxUID)._available
        plugin = next((p for p in available if p != "gdcm"), "")
    except Exception:
        plugin = ""
    ds.pixel_array_options(decoding_plugin=plugin)


def save_frames(ds, frames_dir: Path, base_name: str, backend: str = "pydicom") -> int:
    """
    Read pixel_array BEFORE creating any directory so a failed pixel read
    never leaves an empty dir on disk. Raises on any failure.
    """
    configure_pixel_backend(ds, backend)
    px = ds.pixel_array
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _save(img, idx):
        Image.fromarray(img, mode="L").save(
            frames_dir / f"{base_name}_frame_{idx:04d}.{FRAME_FORMAT}"
        )

    if px.ndim == 2:
        _save(to_uint8_windowed(px, ds), 1)
        return 1
    if px.ndim == 3:
        for i in range(px.shape[0]):
            _save(to_uint8_windowed(px[i], ds), i + 1)
        return px.shape[0]

    raise ValueError(f"Unexpected pixel_array shape: {px.shape}")


# =========================================================
# Reporting helpers
# =========================================================
def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_annotated_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_png: Path,
    threshold: Optional[float] = None,
    bins: int = 30,
) -> None:
    """Histogram with mean line + mean±1SD band (matplotlib imported lazily)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(values, bins=bins, edgecolor="black", color="#4878CF", alpha=0.85,
            label="Frequency")
    ax.axvspan(max(0.0, mean_val - std_val), mean_val + std_val,
               alpha=0.15, color="orange",
               label=f"Mean ± 1 SD  ({mean_val - std_val:.1f} – {mean_val + std_val:.1f})")
    ax.axvline(mean_val, color="orange", linewidth=2.0, linestyle="--",
               label=f"Mean = {mean_val:.2f}")
    if threshold is not None:
        ax.axvline(threshold, color="red", linewidth=2.0, linestyle=":",
                   label=f"Threshold = {threshold}")

    ax.text(0.97, 0.97,
            f"n   = {len(values):,}\nMean = {mean_val:.2f}\nSD   = {std_val:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.8))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def describe_distribution(values: np.ndarray, label: str, output_txt: Path) -> None:
    """Descriptive stats + normality tests to a text file (scipy lazily imported)."""
    from scipy import stats as sps

    n = len(values)
    lines = [
        f"Basic Statistics for '{label}'",
        "=" * 50,
        f"Total valid observations : {n}",
        "",
        f"Mean               : {np.mean(values):.6f}",
        f"Median             : {np.median(values):.6f}",
        f"Standard Deviation : {(np.std(values, ddof=1) if n > 1 else 0.0):.6f}",
        f"Minimum            : {np.min(values):.6f}",
        f"25th Percentile    : {np.percentile(values, 25):.6f}",
        f"75th Percentile    : {np.percentile(values, 75):.6f}",
        f"Maximum            : {np.max(values):.6f}",
    ]
    if n > 2:
        lines.append(f"Skewness           : {sps.skew(values, bias=False):.6f}")
    if n > 3:
        lines.append(f"Kurtosis           : {sps.kurtosis(values, bias=False):.6f}")
    lines.append("")

    if 3 <= n <= 5000:
        stat, p = sps.shapiro(values)
        verdict = "reject normality" if p < 0.05 else "consistent with normality"
        lines.append(f"Shapiro-Wilk       : stat={stat:.6f}  p={p:.6g}  -> {verdict}")
    if n >= 8:
        stat, p = sps.normaltest(values)
        verdict = "reject normality" if p < 0.05 else "consistent with normality"
        lines.append(f"D'Agostino-Pearson : stat={stat:.6f}  p={p:.6g}  -> {verdict}")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def distribution_report(
    values: np.ndarray,
    label: str,
    out_dir: Path,
    prefix: str,
    threshold: Optional[float] = None,
    bins: int = 30,
) -> None:
    """
    Full distribution report for one numeric variable:
    annotated histogram, boxplot, Q-Q plot, and descriptive-stats text.
    Files land in out_dir as <prefix>_{histogram,boxplot,qqplot}.png
    and <prefix>_basic_stats.txt.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_annotated_histogram(
        values, f"Distribution of {label}", label,
        out_dir / f"{prefix}_histogram.png", threshold=threshold, bins=bins,
    )
    describe_distribution(values, label, out_dir / f"{prefix}_basic_stats.txt")

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, vert=False)
    plt.title(f"Boxplot of {label}")
    plt.xlabel(label)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_boxplot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 8))
    sps.probplot(values, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {label}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_qqplot.png", dpi=150)
    plt.close()
