"""
DICOM parsing, file discovery, indexing, and frame extraction.

The functions in this module run in worker subprocesses (via
ProcessPoolExecutor), so everything here must be importable at module level
and free of unpicklable global state.
"""

import os
import logging
import datetime
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from .schema import ALL_COLUMNS

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker-process globals — set once per process via _init_worker()
# ═══════════════════════════════════════════════════════════════════════════════

# Populated by _init_worker(); never written to after that.
_existing_sop_uids: frozenset = frozenset()


def _init_worker(existing_uids: frozenset) -> None:
    """
    ProcessPoolExecutor initializer.

    Runs exactly ONCE per worker process at startup. Stores the frozenset
    of already-ingested SOPInstanceUIDs in a module-level global so every
    subsequent call to parse_dicom_file() can do an O(1) membership check
    without any IPC overhead.
    """
    global _existing_sop_uids
    _existing_sop_uids = existing_uids


# ═══════════════════════════════════════════════════════════════════════════════
# DICOM metadata parsing  (runs in subprocess — must be module-level)
# ═══════════════════════════════════════════════════════════════════════════════

def safe_str(ds, tag_name: str) -> str:
    try:
        val = getattr(ds, tag_name, None)
        if val is None:
            return ""
        if hasattr(val, "sequence_of_items"):
            return ""
        if hasattr(val, "__iter__") and not isinstance(val, str):
            return str(val).strip()
        return str(val).strip()
    except Exception:
        return ""


def parse_dicom_file(path_str: str) -> Optional[dict]:
    """
    Parse one .dcm file and return a flat metadata dict.

    SOPInstanceUID-based skip
    ─────────────────────────
    Before the full 80-field header parse we do a minimal read that asks
    pydicom to load ONLY the SOPInstanceUID tag. If the returned UID is
    already in _existing_sop_uids (populated by _init_worker once per worker
    process from SQLite) we return None immediately — same sentinel value as
    "not a DICOM file" — so the full parse is skipped entirely.
    """
    path = Path(path_str)
    now  = datetime.datetime.utcnow().isoformat()

    # ── Quick SOPInstanceUID check ────────────────────────────────────────────
    try:
        ds_quick  = pydicom.dcmread(
            path_str,
            specific_tags=["SOPInstanceUID"],
            stop_before_pixels=True,
            force=False,
        )
        quick_uid = str(getattr(ds_quick, "SOPInstanceUID", "") or "").strip()
        if quick_uid and quick_uid in _existing_sop_uids:
            return None   # already in SQLite — skip full parse
    except Exception:
        pass   # fall through; full parse will handle the error properly

    # ── Full parse ────────────────────────────────────────────────────────────
    try:
        ds = pydicom.dcmread(path_str, stop_before_pixels=True, force=False)
    except InvalidDicomError:
        return None
    except Exception as exc:
        return {col: None for col in ALL_COLUMNS} | {
            "sop_instance_uid":   f"UNREADABLE_{path.stem[:40]}",
            "source_file":        path.name,
            "source_path":        path_str,
            "parse_error":        str(exc)[:500],
            "sqlite_inserted_at": now,
        }

    sop_uid = safe_str(ds, "SOPInstanceUID")
    if not sop_uid:
        return None

    accession = safe_str(ds, "AccessionNumber") or f"MISSING_{sop_uid[:12]}"

    try:
        frame_count = int(ds.NumberOfFrames)
    except Exception:
        frame_count = 1

    return {
        "sop_instance_uid":                    sop_uid,
        "accession_number":                    accession,
        "study_instance_uid":                  safe_str(ds, "StudyInstanceUID"),
        "series_instance_uid":                 safe_str(ds, "SeriesInstanceUID"),
        "patient_id":                          safe_str(ds, "PatientID"),
        "patient_name":                        safe_str(ds, "PatientName"),
        "patient_sex":                         safe_str(ds, "PatientSex"),
        "patient_age":                         safe_str(ds, "PatientAge"),
        "pregnancy_status":                    safe_str(ds, "PregnancyStatus"),
        "patient_identity_removed":            safe_str(ds, "PatientIdentityRemoved"),
        "deidentification_method":             safe_str(ds, "DeidentificationMethod"),
        "study_date":                          safe_str(ds, "StudyDate"),
        "study_time":                          safe_str(ds, "StudyTime"),
        "study_description":                   safe_str(ds, "StudyDescription"),
        "referring_physician":                 safe_str(ds, "ReferringPhysicianName"),
        "requested_procedure_description":     safe_str(ds, "RequestedProcedureDescription"),
        "performed_procedure_step_start_date": safe_str(ds, "PerformedProcedureStepStartDate"),
        "performed_procedure_step_start_time": safe_str(ds, "PerformedProcedureStepStartTime"),
        "performed_procedure_step_description":safe_str(ds, "PerformedProcedureStepDescription"),
        "series_date":                         safe_str(ds, "SeriesDate"),
        "series_time":                         safe_str(ds, "SeriesTime"),
        "series_description":                  safe_str(ds, "SeriesDescription"),
        "series_number":                       safe_str(ds, "SeriesNumber"),
        "acquisition_number":                  safe_str(ds, "AcquisitionNumber"),
        "modality":                            safe_str(ds, "Modality"),
        "protocol_name":                       safe_str(ds, "ProtocolName"),
        "sop_class_uid":                       safe_str(ds, "SOPClassUID"),
        "instance_number":                     safe_str(ds, "InstanceNumber"),
        "image_type":                          safe_str(ds, "ImageType"),
        "acquisition_date":                    safe_str(ds, "AcquisitionDate"),
        "acquisition_time":                    safe_str(ds, "AcquisitionTime"),
        "content_date":                        safe_str(ds, "ContentDate"),
        "content_time":                        safe_str(ds, "ContentTime"),
        "rows":                                safe_str(ds, "Rows"),
        "columns":                             safe_str(ds, "Columns"),
        "bits_allocated":                      safe_str(ds, "BitsAllocated"),
        "bits_stored":                         safe_str(ds, "BitsStored"),
        "high_bit":                            safe_str(ds, "HighBit"),
        "samples_per_pixel":                   safe_str(ds, "SamplesPerPixel"),
        "pixel_representation":                safe_str(ds, "PixelRepresentation"),
        "photometric_interpretation":          safe_str(ds, "PhotometricInterpretation"),
        "number_of_frames":                    safe_str(ds, "NumberOfFrames") or str(frame_count),
        "frame_count":                         str(frame_count),
        "frame_time":                          safe_str(ds, "FrameTime"),
        "cine_rate":                           safe_str(ds, "CineRate"),
        "images_in_acquisition":               safe_str(ds, "ImagesInAcquisition"),
        "representative_frame_number":         safe_str(ds, "RepresentativeFrameNumber"),
        "start_trim":                          safe_str(ds, "StartTrim"),
        "stop_trim":                           safe_str(ds, "StopTrim"),
        "recommended_display_frame_rate":      safe_str(ds, "RecommendedDisplayFrameRate"),
        "kvp":                                 safe_str(ds, "KVP"),
        "exposure_time":                       safe_str(ds, "ExposureTime"),
        "xray_tube_current":                   safe_str(ds, "XRayTubeCurrent"),
        "avg_pulse_width":                     safe_str(ds, "AveragePulseWidth"),
        "radiation_setting":                   safe_str(ds, "RadiationSetting"),
        "radiation_mode":                      safe_str(ds, "RadiationMode"),
        "dose_product":                        safe_str(ds, "ImageAndFluoroscopyAreaDoseProduct"),
        "distance_source_to_detector":         safe_str(ds, "DistanceSourceToDetector"),
        "distance_source_to_patient":          safe_str(ds, "DistanceSourceToPatient"),
        "est_magnification_factor":            safe_str(ds, "EstimatedRadiographicMagnificationFactor"),
        "intensifier_size":                    safe_str(ds, "IntensifierSize"),
        "imager_pixel_spacing":                safe_str(ds, "ImagerPixelSpacing"),
        "focal_spots":                         safe_str(ds, "FocalSpots"),
        "positioner_motion":                   safe_str(ds, "PositionerMotion"),
        "positioner_primary_angle":            safe_str(ds, "PositionerPrimaryAngle"),
        "positioner_secondary_angle":          safe_str(ds, "PositionerSecondaryAngle"),
        "patient_position":                    safe_str(ds, "PatientPosition"),
        "window_center":                       safe_str(ds, "WindowCenter"),
        "window_width":                        safe_str(ds, "WindowWidth"),
        "voi_lut_function":                    safe_str(ds, "VOILUTFunction"),
        "lossy_image_compression":             safe_str(ds, "LossyImageCompression"),
        "longitudinal_temporal_info_modified": safe_str(ds, "LongitudinalTemporalInformationModified"),
        "pixel_intensity_relationship":        safe_str(ds, "PixelIntensityRelationship"),
        "contrast_bolus_agent":                safe_str(ds, "ContrastBolusAgent"),
        "contrast_bolus_ingredient":           safe_str(ds, "ContrastBolusIngredient"),
        "manufacturer":                        safe_str(ds, "Manufacturer"),
        "manufacturer_model_name":             safe_str(ds, "ManufacturerModelName"),
        "station_name":                        safe_str(ds, "StationName"),
        "software_versions":                   safe_str(ds, "SoftwareVersions"),
        "device_serial_number":                safe_str(ds, "DeviceSerialNumber"),
        "detector_id":                         safe_str(ds, "DetectorID"),
        "detector_description":                safe_str(ds, "DetectorDescription"),
        "specific_character_set":              safe_str(ds, "SpecificCharacterSet"),
        "source_file":                         path.name,
        "source_path":                         path_str,
        "parse_error":                         None,
        "sqlite_inserted_at":                  now,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# File discovery & chunking utilities
# ═══════════════════════════════════════════════════════════════════════════════

def iter_dicom_files(root: Path):
    """Yield absolute path strings for every .dcm file under root."""
    for dirpath, _, filenames in os.walk(str(root)):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                yield str(Path(dirpath) / fname)


def chunked(iterable, size: int):
    """Yield successive fixed-size chunks from any iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


# ═══════════════════════════════════════════════════════════════════════════════
# Image ingestion helpers — indexing & frame extraction
# ═══════════════════════════════════════════════════════════════════════════════

def build_dicom_index(dicom_root: Path, con=None) -> dict[str, str]:
    """
    Build a {stem → full_path} index for all .dcm files.

    Strategy (fastest first):
      1. If the SQLite dicom_files table has entries, build from there — O(rows).
      2. Otherwise fall back to an os.walk() of the filesystem.
    """
    if con is not None:
        count = con.execute(
            "SELECT COUNT(*) FROM dicom_files WHERE source_file IS NOT NULL"
        ).fetchone()[0]
        if count > 0:
            log.info(f"Building DICOM index from SQLite ({count:,} rows) ...")
            index: dict[str, str] = {}
            for row in con.execute(
                "SELECT source_file, source_path FROM dicom_files "
                "WHERE source_file IS NOT NULL AND source_path IS NOT NULL"
            ):
                stem = Path(row[0]).stem
                if stem and stem not in index:
                    index[stem] = row[1]
            log.info(f"Index ready: {len(index):,} unique stems")
            return index

    log.info(f"Building DICOM index by scanning {dicom_root} (this may take a while) …")
    index = {}
    duplicates = 0
    for dirpath, _, filenames in os.walk(str(dicom_root)):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                stem = Path(fname).stem
                if stem in index:
                    duplicates += 1
                else:
                    index[stem] = str(Path(dirpath) / fname)
    log.info(
        f"Index ready: {len(index):,} unique stems "
        f"({duplicates} duplicate filenames ignored)"
    )
    return index


def parse_dicom_stem_from_csv_path(file_path_str: str) -> Optional[str]:
    """
    Extract the DICOM file stem from a labeled CSV file_path value.

    Example:
      input : '/Deep_Angiography/.../02_DSA 3 LD/2.16.840.1...247242.dcm'
      output: '2.16.840.1...247242'
    """
    if not file_path_str or not file_path_str.strip():
        return None
    p     = file_path_str.strip()
    fname = p.rsplit("/", 1)[-1] if "/" in p else p
    if fname.lower().endswith(".dcm"):
        return fname[:-4]
    return fname if fname else None


def dicom_frame_to_uint8_rgb(frame: np.ndarray, photometric: str = "") -> np.ndarray:
    """
    Normalise a single 2-D DICOM frame (any numeric dtype) to uint8 RGB (HxWx3).
    Handles MONOCHROME1 (inverts so high values appear bright).
    """
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    if hi > lo:
        f = (f - lo) / (hi - lo) * 255.0
    else:
        f = np.zeros_like(f)
    f = f.astype(np.uint8)

    if "MONOCHROME1" in photometric.upper():
        f = 255 - f

    return np.stack([f, f, f], axis=-1)   # HxWx3


def extract_frames(path_str: str) -> tuple[list[np.ndarray], dict]:
    """
    Open a multi-frame DICOM file and return:
      (list_of_uint8_rgb_frames, metadata_dict)
    """
    ds          = pydicom.dcmread(path_str, force=False)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    pixels      = ds.pixel_array

    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]
    elif pixels.ndim == 3:
        if pixels.shape[2] in (3, 4):
            pixels = pixels[np.newaxis, ...]

    frames: list[np.ndarray] = []
    for i in range(pixels.shape[0]):
        raw = pixels[i]
        if raw.ndim == 3:
            f = raw.astype(np.float32)
            lo, hi = f.min(), f.max()
            if hi > lo:
                f = (f - lo) / (hi - lo) * 255.0
            f = f.astype(np.uint8)
            if f.shape[2] == 4:
                f = f[:, :, :3]
        else:
            f = dicom_frame_to_uint8_rgb(raw, photometric)
        frames.append(f)

    sop_uid = str(getattr(ds, "SOPInstanceUID",   "") or "")
    acc     = str(getattr(ds, "AccessionNumber",   "") or "").strip()
    if not acc:
        acc = f"MISSING_{sop_uid[:12]}"

    meta = {
        "accession_number": acc,
        "series_uid":       str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "sop_uid":          sop_uid,
        "total_frames":     int(len(frames)),
        "rows":             int(getattr(ds, "Rows",    0) or 0),
        "columns":          int(getattr(ds, "Columns", 0) or 0),
        "modality":         str(getattr(ds, "Modality", "") or ""),
        "photometric":      photometric,
    }
    return frames, meta
