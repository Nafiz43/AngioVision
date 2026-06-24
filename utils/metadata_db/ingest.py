#!/usr/bin/env python3
"""
DICOM → SQLite Ingestion Pipeline  +  DICOM Images → ChromaDB

Recursively walks a DICOM directory, parses metadata from every .dcm file
in parallel (ProcessPoolExecutor), and stores everything in a flat SQLite
table keyed on SOPInstanceUID.

Also ingests a radiology report CSV (Anon Acc #, radrpt) into a separate
`radiology_reports` table, joinable to `dicom_files` via accession_number.

NEW ── Image ingestion into ChromaDB:
  • Reads labeled_DSA_2023_10_24.csv to identify labeled sequences
  • Skips rows where angio_run contains the word 'other'
  • Parses the DICOM file stem from the file_path column
      e.g. ".../02_DSA 3 LD/2.16.840.1...dcm"  →  "2.16.840.1..."
  • Locates the actual .dcm file under the DICOM root (index built from
    SQLite first; falls back to a full filesystem walk)
  • Extracts every frame via pydicom, normalises to uint8 RGB (HxWx3)
  • Embeds frames with microsoft/rad-dino (ViT-B/14 DINOv2, 768-dim)
    — model loaded ONCE at startup, reused across all batches
  • Upserts into a persistent ChromaDB collection called "dicom_images"
    using pre-computed embeddings (embeddings= kwarg)
  • Tracks progress per sequence in SQLite (image_ingestion_status)
    → safe to kill and restart at any point

Embedding model — microsoft/rad-dino:
  • Domain-specific ViT-B/14 (DINOv2) pre-trained on chest X-rays,
    CTs, MRIs, and fluoroscopy — better than generic CLIP for DSA.
  • Produces 768-dim CLS-token embeddings, L2-normalised before storage.
  • HuggingFace model ID: "microsoft/rad-dino"
  • Loaded via AutoModel + AutoImageProcessor from `transformers`.

NOTE — ChromaDB collection dimension:
  If you previously used OpenCLIP (512-dim) and are switching to
  RAD-DINO (768-dim), you must delete or rename the existing ChromaDB
  collection before the first run, otherwise ChromaDB will raise a
  dimension mismatch error.

CHANGES vs previous version:
  1. SOPInstanceUID-based skip for DICOM metadata re-runs
       • Existing SOPInstanceUIDs are loaded from SQLite into a frozenset
         before the parallel parse loop starts.
       • Each worker receives this frozenset via a pool initializer
         (_init_worker) — shipped once per process, not once per file.
       • At the top of parse_dicom_file() a minimal pydicom.dcmread()
         with specific_tags=['SOPInstanceUID'] reads only that one tag.
         If the UID is already in the frozenset → return None immediately,
         skipping the full 80-field header parse entirely.
       • Files that are genuinely new proceed to the full parse as before.

  2. RAD-DINO replaces OpenCLIP for image embedding
       • microsoft/rad-dino loaded once via load_rad_dino_model().
       • Frames embedded in batches via embed_frames_rad_dino() which
         uses AutoImageProcessor + AutoModel from transformers.
       • CLS token from last_hidden_state[:,0,:] is the frame embedding.
       • Vectors are L2-normalised; ChromaDB collection uses cosine space.
       • ChromaDB receives embeddings= directly — no wrapper ever called.

Design principles (unchanged from original):
  • SOPInstanceUID is the primary key / deduplification key
  • Missing AccessionNumber → synthetic key MISSING_{sop_uid[:12]}
  • INSERT OR IGNORE — re-runs are safe and idempotent
  • Full audit trail: parse_error, sqlite_inserted_at columns
  • Chunked submission — tqdm starts immediately, no upfront blocking

Requirements:
    pip install pydicom tqdm chromadb transformers torch pillow numpy
    (pillow is needed by pydicom for compressed JPEG/JP2 transfers)
"""

import os
import sys
import csv
import sqlite3
import logging
import argparse
import datetime
import itertools
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print("ERROR: pydicom not installed.  Run: pip install pydicom")
    sys.exit(1)

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore[assignment]

# RAD-DINO (microsoft/rad-dino) via HuggingFace transformers.
# Imported only when image ingestion is needed; availability checked
# at runtime via RAD_DINO_AVAILABLE.
try:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel
    RAD_DINO_AVAILABLE = True
except ImportError:
    RAD_DINO_AVAILABLE = False
    torch              = None  # type: ignore[assignment]
    PILImage           = None  # type: ignore[assignment]
    AutoImageProcessor = None  # type: ignore[assignment]
    AutoModel          = None  # type: ignore[assignment]

RAD_DINO_MODEL_ID = "microsoft/rad-dino"

# ── Constants ──────────────────────────────────────────────────────────────────
DICOM_ROOT        = Path("/data/Deep_Angiography/DICOM")
SQLITE_DB         = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
REPORTS_CSV       = Path("/data/Deep_Angiography/Reports/Report_List_v01_01_merged_raw.csv")
LABELED_CSV       = Path("/data/Deep_Angiography/labeled_DSA_2023_10_24.csv")
CHROMADB_PATH     = Path("/data/Deep_Angiography/AngioVision/chromadb")
CHROMA_COLLECTION = "dicom_images"

PARSE_WORKERS = max(1, os.cpu_count() - 1)
SQL_FLUSH     = 1000   # DICOM rows buffered before each SQLite commit
SUBMIT_CHUNK  = 2000   # futures submitted per chunk
CHROMA_BATCH  = 32     # frames per ChromaDB add() call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── SQLite schema ──────────────────────────────────────────────────────────────
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dicom_files (

    sop_instance_uid                     TEXT PRIMARY KEY,

    accession_number                     TEXT,
    study_instance_uid                   TEXT,
    series_instance_uid                  TEXT,

    patient_id                           TEXT,
    patient_name                         TEXT,
    patient_sex                          TEXT,
    patient_age                          TEXT,
    pregnancy_status                     TEXT,
    patient_identity_removed             TEXT,
    deidentification_method              TEXT,

    study_date                           TEXT,
    study_time                           TEXT,
    study_description                    TEXT,
    referring_physician                  TEXT,
    requested_procedure_description      TEXT,
    performed_procedure_step_start_date  TEXT,
    performed_procedure_step_start_time  TEXT,
    performed_procedure_step_description TEXT,

    series_date                          TEXT,
    series_time                          TEXT,
    series_description                   TEXT,
    series_number                        TEXT,
    acquisition_number                   TEXT,
    modality                             TEXT,
    protocol_name                        TEXT,

    sop_class_uid                        TEXT,
    instance_number                      TEXT,
    image_type                           TEXT,
    acquisition_date                     TEXT,
    acquisition_time                     TEXT,
    content_date                         TEXT,
    content_time                         TEXT,

    rows                                 TEXT,
    columns                              TEXT,
    bits_allocated                       TEXT,
    bits_stored                          TEXT,
    high_bit                             TEXT,
    samples_per_pixel                    TEXT,
    pixel_representation                 TEXT,
    photometric_interpretation           TEXT,
    number_of_frames                     TEXT,
    frame_count                          TEXT,
    frame_time                           TEXT,
    cine_rate                            TEXT,
    images_in_acquisition                TEXT,
    representative_frame_number          TEXT,
    start_trim                           TEXT,
    stop_trim                            TEXT,
    recommended_display_frame_rate       TEXT,

    kvp                                  TEXT,
    exposure_time                        TEXT,
    xray_tube_current                    TEXT,
    avg_pulse_width                      TEXT,
    radiation_setting                    TEXT,
    radiation_mode                       TEXT,
    dose_product                         TEXT,

    distance_source_to_detector          TEXT,
    distance_source_to_patient           TEXT,
    est_magnification_factor             TEXT,
    intensifier_size                     TEXT,
    imager_pixel_spacing                 TEXT,
    focal_spots                          TEXT,
    positioner_motion                    TEXT,
    positioner_primary_angle             TEXT,
    positioner_secondary_angle           TEXT,
    patient_position                     TEXT,

    window_center                        TEXT,
    window_width                         TEXT,
    voi_lut_function                     TEXT,
    lossy_image_compression              TEXT,
    longitudinal_temporal_info_modified  TEXT,
    pixel_intensity_relationship         TEXT,

    contrast_bolus_agent                 TEXT,
    contrast_bolus_ingredient            TEXT,

    manufacturer                         TEXT,
    manufacturer_model_name              TEXT,
    station_name                         TEXT,
    software_versions                    TEXT,
    device_serial_number                 TEXT,
    detector_id                          TEXT,
    detector_description                 TEXT,
    specific_character_set               TEXT,

    source_file                          TEXT,
    source_path                          TEXT,

    parse_error                          TEXT,
    sqlite_inserted_at                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_accession_number   ON dicom_files (accession_number);
CREATE INDEX IF NOT EXISTS idx_patient_id         ON dicom_files (patient_id);
CREATE INDEX IF NOT EXISTS idx_study_uid          ON dicom_files (study_instance_uid);
CREATE INDEX IF NOT EXISTS idx_series_uid         ON dicom_files (series_instance_uid);
CREATE INDEX IF NOT EXISTS idx_modality           ON dicom_files (modality);
CREATE INDEX IF NOT EXISTS idx_series_description ON dicom_files (series_description);
CREATE INDEX IF NOT EXISTS idx_frame_count        ON dicom_files (frame_count);
CREATE INDEX IF NOT EXISTS idx_source_path        ON dicom_files (source_path);
CREATE INDEX IF NOT EXISTS idx_patient_sex        ON dicom_files (patient_sex);
CREATE INDEX IF NOT EXISTS idx_study_date         ON dicom_files (study_date);
CREATE INDEX IF NOT EXISTS idx_acquisition_date   ON dicom_files (acquisition_date);
CREATE INDEX IF NOT EXISTS idx_parse_error        ON dicom_files (parse_error)
    WHERE parse_error IS NOT NULL;

-- ── Radiology reports ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS radiology_reports (
    accession_number   TEXT PRIMARY KEY,
    radrpt             TEXT,
    source_csv         TEXT,
    csv_inserted_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_rpt_accession ON radiology_reports (accession_number);

-- ── Image ingestion tracking ─────────────────────────────────────────────────
-- One row per DICOM sequence (one .dcm file = one sequence).
-- sequence_id  = the DICOM file stem (guaranteed unique per file).
-- ChromaDB frame IDs are "{sequence_id}_f{frame_idx:06d}", so partial
-- ingestion from a crashed run can be cleaned up deterministically on retry.
CREATE TABLE IF NOT EXISTS image_ingestion_status (
    sequence_id      TEXT PRIMARY KEY,
    accession_number TEXT,
    series_uid       TEXT,
    source_path      TEXT,
    frames_ingested  INTEGER DEFAULT 0,
    status           TEXT,      -- 'in_progress' | 'completed' | 'error'
    error_msg        TEXT,
    ingested_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_img_status    ON image_ingestion_status (status);
CREATE INDEX IF NOT EXISTS idx_img_accession ON image_ingestion_status (accession_number);
"""

ALL_COLUMNS = [
    "sop_instance_uid", "accession_number", "study_instance_uid",
    "series_instance_uid", "patient_id", "patient_name", "patient_sex",
    "patient_age", "pregnancy_status", "patient_identity_removed",
    "deidentification_method", "study_date", "study_time", "study_description",
    "referring_physician", "requested_procedure_description",
    "performed_procedure_step_start_date", "performed_procedure_step_start_time",
    "performed_procedure_step_description", "series_date", "series_time",
    "series_description", "series_number", "acquisition_number", "modality",
    "protocol_name", "sop_class_uid", "instance_number", "image_type",
    "acquisition_date", "acquisition_time", "content_date", "content_time",
    "rows", "columns", "bits_allocated", "bits_stored", "high_bit",
    "samples_per_pixel", "pixel_representation", "photometric_interpretation",
    "number_of_frames", "frame_count", "frame_time", "cine_rate",
    "images_in_acquisition", "representative_frame_number", "start_trim",
    "stop_trim", "recommended_display_frame_rate", "kvp", "exposure_time",
    "xray_tube_current", "avg_pulse_width", "radiation_setting", "radiation_mode",
    "dose_product", "distance_source_to_detector", "distance_source_to_patient",
    "est_magnification_factor", "intensifier_size", "imager_pixel_spacing",
    "focal_spots", "positioner_motion", "positioner_primary_angle",
    "positioner_secondary_angle", "patient_position", "window_center",
    "window_width", "voi_lut_function", "lossy_image_compression",
    "longitudinal_temporal_info_modified", "pixel_intensity_relationship",
    "contrast_bolus_agent", "contrast_bolus_ingredient", "manufacturer",
    "manufacturer_model_name", "station_name", "software_versions",
    "device_serial_number", "detector_id", "detector_description",
    "specific_character_set", "source_file", "source_path",
    "parse_error", "sqlite_inserted_at",
]

INSERT_SQL = (
    f"INSERT OR IGNORE INTO dicom_files ({', '.join(ALL_COLUMNS)}) "
    f"VALUES ({', '.join(['?'] * len(ALL_COLUMNS))})"
)

INSERT_RPT_SQL = """
    INSERT OR IGNORE INTO radiology_reports
        (accession_number, radrpt, source_csv, csv_inserted_at)
    VALUES (?, ?, ?, ?)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Worker-process globals — set once per process via _init_worker()
# ═══════════════════════════════════════════════════════════════════════════════

# Populated by _init_worker(); never written to after that.
_existing_sop_uids: frozenset = frozenset()


def _init_worker(existing_uids: frozenset) -> None:
    """
    ProcessPoolExecutor initializer.

    Runs exactly ONCE per worker process at startup.  Stores the frozenset
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
    except (AttributeError, TypeError, ValueError) as exc:
        log.debug(f"Could not read DICOM tag {tag_name!r}: {exc}")
        return ""


def parse_dicom_file(path_str: str) -> Optional[dict]:
    """
    Parse one .dcm file and return a flat metadata dict.

    SOPInstanceUID-based skip (Option 2)
    ─────────────────────────────────────
    Before the full 80-field header parse we do a minimal read that asks
    pydicom to load ONLY the SOPInstanceUID tag.  This is cheap: pydicom
    stops scanning the binary stream as soon as the tag is found.

    If the returned UID is already in _existing_sop_uids (populated by
    _init_worker once per worker process from SQLite) we return None
    immediately — same sentinel value as "not a DICOM file" — so the
    caller's INSERT OR IGNORE path is never reached and the full 80-field
    parse is skipped entirely.

    Files with a UID not yet in the DB proceed to the normal full parse.
    Files that fail the quick read (corrupt, not DICOM, etc.) fall through
    to the full parse which records a proper error stub row.
    """
    path = Path(path_str)
    now  = datetime.datetime.utcnow().isoformat()

    # ── Quick SOPInstanceUID check ────────────────────────────────────────────
    # specific_tags instructs pydicom to stop reading as soon as the
    # requested tag is consumed — dramatically less I/O than a full header
    # parse.  Any exception here is silently ignored; the full parse below
    # will surface a proper error if the file is genuinely unreadable.
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
    except (AttributeError, ValueError, TypeError):
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
# SQLite helpers
# ═══════════════════════════════════════════════════════════════════════════════

def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SQLITE_SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-65536")
    con.commit()
    log.info(f"SQLite DB opened: {db_path}")
    return con


def flush_batch(con: sqlite3.Connection, batch: list[dict]) -> tuple[int, int]:
    """INSERT OR IGNORE batch. Returns (inserted, ignored)."""
    if not batch:
        return 0, 0
    before = con.execute("SELECT COUNT(*) FROM dicom_files").fetchone()[0]
    con.executemany(INSERT_SQL, [[r.get(col) for col in ALL_COLUMNS] for r in batch])
    con.commit()
    after   = con.execute("SELECT COUNT(*) FROM dicom_files").fetchone()[0]
    new     = after - before
    ignored = len(batch) - new
    return new, ignored


# ═══════════════════════════════════════════════════════════════════════════════
# Radiology report ingestion
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_reports(con: sqlite3.Connection, csv_path: Path) -> tuple[int, int]:
    """
    Load Report_List_v01_01_merged_raw.csv into radiology_reports.
    Expected columns (case-insensitive): 'Anon Acc #', 'radrpt'.
    Returns (inserted, ignored).
    """
    if not csv_path.exists():
        log.warning(f"Reports CSV not found — skipping: {csv_path}")
        return 0, 0

    now      = datetime.datetime.utcnow().isoformat()
    source   = str(csv_path)
    bad_rows = 0

    log.info(f"Ingesting radiology reports from: {csv_path}")
    before = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            log.error("CSV appears to be empty — no headers found.")
            return 0, 0

        norm    = {h.strip().lower(): h for h in reader.fieldnames}
        acc_key = norm.get("anon acc #")
        rpt_key = norm.get("radrpt")

        if acc_key is None or rpt_key is None:
            log.error(
                f"Expected columns 'Anon Acc #' and 'radrpt' — "
                f"found: {list(reader.fieldnames)}"
            )
            return 0, 0

        rows_to_insert = []
        for row in reader:
            acc = (row.get(acc_key) or "").strip()
            rpt = (row.get(rpt_key) or "").strip()
            if not acc:
                bad_rows += 1
                continue
            rows_to_insert.append((acc, rpt or None, source, now))

        con.executemany(INSERT_RPT_SQL, rows_to_insert)
        con.commit()

    after    = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]
    inserted = after - before
    ignored  = len(rows_to_insert) - inserted

    log.info(
        f"Reports CSV — total rows: {len(rows_to_insert):,} | "
        f"inserted: {inserted:,} | duplicates skipped: {ignored:,} | "
        f"bad (no accession): {bad_rows:,}"
    )
    return inserted, ignored


# ═══════════════════════════════════════════════════════════════════════════════
# Image ingestion helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_dicom_index(
    dicom_root: Path,
    con: Optional[sqlite3.Connection] = None,
) -> dict[str, str]:
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
    p    = file_path_str.strip()
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


# ═══════════════════════════════════════════════════════════════════════════════
# RAD-DINO (microsoft/rad-dino) — loaded ONCE per ingestion run
# ═══════════════════════════════════════════════════════════════════════════════

def load_rad_dino_model(device: Optional[str] = None):
    """
    Load microsoft/rad-dino exactly once and return (model, processor, device).

    RAD-DINO is a ViT-B/14 DINOv2 model pre-trained on a large collection
    of radiology images (chest X-ray, CT, MRI, fluoroscopy/DSA).  It
    produces 768-dim CLS-token embeddings that are substantially more
    discriminative for angiographic sequences than generic CLIP features.

    The model and its AutoImageProcessor are both fetched from HuggingFace
    on the first call (cached locally thereafter).  Subsequent runs load
    entirely from the local cache with no network traffic.
    """
    if not RAD_DINO_AVAILABLE:
        raise ImportError(
            "transformers or torch not installed. "
            "Run: pip install transformers torch pillow"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading {RAD_DINO_MODEL_ID} on {device} (once) …")
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_ID)
    model     = AutoModel.from_pretrained(RAD_DINO_MODEL_ID)
    model.eval()
    model = model.to(device)
    log.info("RAD-DINO model ready  (embedding dim: 768).")
    return model, processor, device


def embed_frames_rad_dino(
    frames: list[np.ndarray],
    model,
    processor,
    device: str,
) -> list[list[float]]:
    """
    Embed a list of uint8 RGB numpy frames with RAD-DINO.

    Steps:
      1. Convert each HxWx3 uint8 array to a PIL Image (processor expects PIL).
      2. Run AutoImageProcessor — handles resize, normalisation, tensor stack.
      3. Forward through the ViT encoder; take the CLS token from the last
         hidden state (index 0 along the sequence dimension).
      4. L2-normalise so ChromaDB cosine similarity == dot product.

    Returns a list of float32 vectors, one per frame, each of length 768.
    ChromaDB receives these via collection.add(embeddings=…) so the model
    is never re-loaded or re-initialised by ChromaDB internals.
    """
    pil_images = [PILImage.fromarray(f) for f in frames]

    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token: shape (batch, 768)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        # L2 normalise → cosine similarity == dot product in ChromaDB
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)

    return cls_emb.cpu().float().numpy().tolist()


def setup_chromadb(chroma_path: Path):
    """
    Initialise a persistent ChromaDB client and return (client, collection).

    No embedding function is attached to the collection — embeddings are
    pre-computed by embed_frames_rad_dino() and passed via the embeddings=
    kwarg in collection.add().  This prevents ChromaDB from ever invoking
    its internal model-loading code paths.
    """
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not installed. Run: pip install chromadb")
    chroma_path.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        f"ChromaDB '{CHROMA_COLLECTION}' at {chroma_path} — "
        f"existing items: {collection.count():,}"
    )
    return client, collection


def load_labeled_csv(csv_path: Path) -> list[dict]:
    """
    Load labeled_DSA_2023_10_24.csv, returning one dict per unique file_path.

    Filtering rules:
      • Rows where angio_run contains 'other' (case-insensitive) → skipped
      • Rows with an empty file_path                              → skipped
      • Duplicate file_path values                                → first kept
    """
    if not csv_path.exists():
        log.error(f"Labeled CSV not found: {csv_path}")
        return []

    rows: list[dict]      = []
    seen_paths: set[str]  = set()
    skipped_other = skipped_dup = skipped_nopath = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            log.error("Labeled CSV appears to be empty.")
            return []

        norm = {h.strip().lower(): h for h in reader.fieldnames}

        angio_key = norm.get("angio_run")
        path_key  = norm.get("file_path")
        acc_key   = norm.get("accession")
        ser_key   = norm.get("seriesuid")
        run_key   = norm.get("run_type")

        if path_key is None:
            log.error(f"'file_path' column not found. Available: {list(reader.fieldnames)}")
            return []

        for row in reader:
            fp        = (row.get(path_key) or "").strip()
            angio_val = (row.get(angio_key) or "").strip().lower() if angio_key else ""

            if not fp:
                skipped_nopath += 1
                continue
            if "other" in angio_val:
                skipped_other += 1
                continue
            if fp in seen_paths:
                skipped_dup += 1
                continue
            seen_paths.add(fp)

            rows.append({
                "accession":  (row.get(acc_key)  or "").strip() if acc_key  else "",
                "series_uid": (row.get(ser_key)  or "").strip() if ser_key  else "",
                "run_type":   (row.get(run_key)  or "").strip() if run_key  else "",
                "angio_run":  angio_val,
                "file_path":  fp,
            })

    log.info(
        f"Labeled CSV — kept: {len(rows):,} | "
        f"skipped (other): {skipped_other:,} | "
        f"skipped (dup file_path): {skipped_dup:,} | "
        f"skipped (no path): {skipped_nopath:,}"
    )
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main image ingestion pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_images_to_chromadb(
    con: sqlite3.Connection,
    labeled_csv: Path,
    dicom_root: Path,
    chroma_path: Path,
    limit_sequences: int = 0,
    chroma_batch_size: int = CHROMA_BATCH,
) -> None:
    """
    End-to-end image ingestion:
      1. Load and filter the labeled CSV
      2. Build a DICOM stem→path index (SQLite-backed, filesystem fallback)
      3. Load microsoft/rad-dino ONCE
      4. Connect to ChromaDB (no embedding function — we supply embeddings)
      5. Iterate sequences: extract frames → embed → store
      6. Record status in image_ingestion_status for safe resume

    Resume behaviour:
      • Sequences with status='completed' in SQLite are skipped.
      • Sequences with status='error' or 'in_progress' are retried;
        any previously written ChromaDB frames are deleted first.
    """
    if not CHROMA_AVAILABLE:
        log.error("chromadb not installed. Run: pip install chromadb")
        return
    if not RAD_DINO_AVAILABLE:
        log.error("transformers or torch not installed. Run: pip install transformers torch pillow")
        return

    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    sequences = load_labeled_csv(labeled_csv)
    if not sequences:
        log.warning("No sequences to ingest after filtering.")
        return

    # ── 2. Build DICOM index ─────────────────────────────────────────────────
    dicom_index = build_dicom_index(dicom_root, con=con)

    # ── 3. Load RAD-DINO once — model stays in memory for all batches ───────
    rad_dino_model, rad_dino_processor, rad_dino_device = load_rad_dino_model()

    # ── 4. Connect to ChromaDB ───────────────────────────────────────────────
    _, collection = setup_chromadb(chroma_path)

    # ── 5. Determine already-completed sequences ─────────────────────────────
    completed: set[str] = {
        row[0]
        for row in con.execute(
            "SELECT sequence_id FROM image_ingestion_status WHERE status = 'completed'"
        ).fetchall()
    }
    log.info(f"Already completed sequences (will be skipped): {len(completed):,}")

    # ── 6. Build work list ───────────────────────────────────────────────────
    todo: list[dict] = []
    not_found = 0

    for seq in sequences:
        stem = parse_dicom_stem_from_csv_path(seq["file_path"])
        if not stem:
            continue

        actual_path = dicom_index.get(stem)
        if actual_path is None:
            log.debug(f"Not found on disk: {stem}")
            not_found += 1
            continue

        seq_id             = stem
        seq["dicom_path"]  = actual_path
        seq["stem"]        = stem
        seq["sequence_id"] = seq_id

        if seq_id in completed:
            continue

        todo.append(seq)

    log.info(f"Files not found on disk : {not_found:,}")
    log.info(f"Sequences queued        : {len(todo):,}  (after resume filter)")

    if limit_sequences > 0:
        todo = todo[:limit_sequences]
        log.info(f"Capped to {limit_sequences:,} sequences via --limit-sequences")

    if not todo:
        log.info("Nothing to ingest — all sequences are already completed.")
        return

    # ── 7. Ingest loop ───────────────────────────────────────────────────────
    succeeded         = 0
    failed            = 0
    total_frames_done = 0

    with tqdm(total=len(todo), unit="seq", desc="Ingesting images → ChromaDB") as pbar:
        for seq in todo:
            seq_id     = seq["sequence_id"]
            path_str   = seq["dicom_path"]
            accession  = seq["accession"]
            series_uid = seq["series_uid"]
            now        = datetime.datetime.utcnow().isoformat()

            con.execute(
                """
                INSERT OR REPLACE INTO image_ingestion_status
                    (sequence_id, accession_number, series_uid, source_path,
                     frames_ingested, status, error_msg, ingested_at)
                VALUES (?, ?, ?, ?, 0, 'in_progress', NULL, ?)
                """,
                (seq_id, accession, series_uid, path_str, now),
            )
            con.commit()

            try:
                frames, meta = extract_frames(path_str)
                n_frames     = len(frames)

                if n_frames == 0:
                    raise ValueError("No frames extracted from DICOM file")

                # Clean up any frames from a previous partial run
                prev_row = con.execute(
                    "SELECT frames_ingested FROM image_ingestion_status "
                    "WHERE sequence_id = ?",
                    (seq_id,),
                ).fetchone()
                prev_n = int(prev_row[0]) if prev_row and prev_row[0] else 0
                if prev_n > 0:
                    stale_ids = [f"{seq_id}_f{i:06d}" for i in range(prev_n)]
                    try:
                        collection.delete(ids=stale_ids)
                    except Exception as exc:
                        log.debug(f"Could not delete stale ChromaDB entries for {seq_id}: {exc}")

                # Embed and add in batches — model already loaded, no reload
                frames_added = 0
                for batch_start in range(0, n_frames, chroma_batch_size):
                    batch = frames[batch_start: batch_start + chroma_batch_size]

                    # ── embed with the already-loaded model ──────────────────
                    embeddings = embed_frames_rad_dino(
                        batch, rad_dino_model, rad_dino_processor, rad_dino_device
                    )

                    ids = [
                        f"{seq_id}_f{(batch_start + j):06d}"
                        for j in range(len(batch))
                    ]
                    metas = [
                        {
                            "accession_number": str(accession or meta["accession_number"]),
                            "series_uid":       str(series_uid or meta["series_uid"]),
                            "sop_uid":          str(meta["sop_uid"]),
                            "frame_index":      int(batch_start + j),
                            "total_frames":     int(n_frames),
                            "source_path":      str(path_str),
                            "angio_run":        str(seq.get("angio_run", "")),
                            "run_type":         str(seq.get("run_type", "")),
                            "modality":         str(meta.get("modality", "")),
                        }
                        for j in range(len(batch))
                    ]

                    # Pass pre-computed embeddings — wrapper never called
                    collection.add(embeddings=embeddings, ids=ids, metadatas=metas)
                    frames_added += len(batch)

                con.execute(
                    """
                    UPDATE image_ingestion_status
                    SET frames_ingested = ?, status = 'completed', ingested_at = ?
                    WHERE sequence_id = ?
                    """,
                    (frames_added, datetime.datetime.utcnow().isoformat(), seq_id),
                )
                con.commit()

                succeeded         += 1
                total_frames_done += frames_added

            except Exception as exc:
                err_msg = str(exc)[:500]
                log.warning(f"Failed [{seq_id}]: {err_msg}")
                con.execute(
                    """
                    UPDATE image_ingestion_status
                    SET status = 'error', error_msg = ?, ingested_at = ?
                    WHERE sequence_id = ?
                    """,
                    (err_msg, datetime.datetime.utcnow().isoformat(), seq_id),
                )
                con.commit()
                failed += 1

            pbar.update(1)
            pbar.set_postfix(ok=succeeded, fail=failed, frames=total_frames_done)

    log.info("─" * 60)
    log.info("Image ingestion complete")
    log.info(f"  Sequences succeeded : {succeeded:>8,}")
    log.info(f"  Sequences failed    : {failed:>8,}")
    log.info(f"  Total frames stored : {total_frames_done:>8,}")
    log.info(f"  ChromaDB total      : {collection.count():>8,}  items")
    log.info("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# DB summary
# ═══════════════════════════════════════════════════════════════════════════════

def db_summary(con: sqlite3.Connection):
    total       = con.execute("SELECT COUNT(*) FROM dicom_files").fetchone()[0]
    with_error  = con.execute("SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NOT NULL").fetchone()[0]
    missing_acc = con.execute("SELECT COUNT(*) FROM dicom_files WHERE accession_number LIKE 'MISSING_%'").fetchone()[0]
    rpt_total   = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]
    linked      = con.execute(
        "SELECT COUNT(DISTINCT d.accession_number) "
        "FROM dicom_files d "
        "JOIN radiology_reports r USING (accession_number)"
    ).fetchone()[0]

    img_completed = con.execute(
        "SELECT COUNT(*) FROM image_ingestion_status WHERE status = 'completed'"
    ).fetchone()[0]
    img_error = con.execute(
        "SELECT COUNT(*) FROM image_ingestion_status WHERE status = 'error'"
    ).fetchone()[0]
    img_frames = con.execute(
        "SELECT COALESCE(SUM(frames_ingested), 0) FROM image_ingestion_status "
        "WHERE status = 'completed'"
    ).fetchone()[0]

    log.info("─" * 60)
    log.info(f"Total DICOM rows    : {total:>10,}")
    log.info(f"Parse errors        : {with_error:>10,}")
    log.info(f"Missing accession   : {missing_acc:>10,}")
    log.info("─" * 60)
    log.info(f"Radiology reports   : {rpt_total:>10,}")
    log.info(f"Accessions linked   : {linked:>10,}  (DICOM ∩ reports)")
    log.info("─" * 60)
    log.info(f"Image seqs done     : {img_completed:>10,}")
    log.info(f"Image seqs errored  : {img_error:>10,}")
    log.info(f"Frames in ChromaDB  : {img_frames:>10,}  (completed seqs)")
    log.info("─" * 60)
    log.info("Modality breakdown:")
    for r in con.execute(
        "SELECT modality, COUNT(*) AS n FROM dicom_files "
        "WHERE parse_error IS NULL GROUP BY modality ORDER BY n DESC LIMIT 15"
    ).fetchall():
        log.info(f"  {(r['modality'] or 'NULL'):12s}  {r['n']:>10,}")
    log.info("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ingest DICOM metadata + radiology reports into SQLite "
            "AND/OR ingest labeled DICOM sequences into ChromaDB (parallel, resumable)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (metadata + reports + images)
  python dicom_ingest_sqlite.py

  # Metadata + reports only (no images)
  python dicom_ingest_sqlite.py --skip-images

  # Image ingestion only (metadata already loaded)
  python dicom_ingest_sqlite.py --images-only

  # Image ingestion — process at most 50 sequences then stop
  python dicom_ingest_sqlite.py --images-only --limit-sequences 50

  # Dry-run the DICOM metadata walk (nothing written)
  python dicom_ingest_sqlite.py --dry-run --skip-images

  # Custom paths
  python dicom_ingest_sqlite.py \\
      --root /data/DICOM \\
      --db   /data/meta.db \\
      --labeled-csv /data/labels.csv \\
      --chromadb    /data/chroma

  # Just print DB + ChromaDB statistics
  python dicom_ingest_sqlite.py --summary-only

Useful join query after full ingestion:
  SELECT d.accession_number, d.series_description, d.frame_count,
         d.modality, d.study_date, r.radrpt
  FROM   dicom_files d
  JOIN   radiology_reports r USING (accession_number)
  WHERE  d.modality = 'XA'
  LIMIT  20;

Query ChromaDB for visually similar frames:
  results = collection.query(
      query_embeddings=[embed_frames_rad_dino([query_frame], model, processor, device)[0]],
      n_results=10,
      where={"modality": "XA"},
  )
        """,
    )

    parser.add_argument("--root",            default=str(DICOM_ROOT))
    parser.add_argument("--db",              default=str(SQLITE_DB))
    parser.add_argument("--reports",         default=str(REPORTS_CSV))
    parser.add_argument("--labeled-csv",     default=str(LABELED_CSV))
    parser.add_argument("--chromadb",        default=str(CHROMADB_PATH))
    parser.add_argument("--workers",         type=int, default=PARSE_WORKERS)
    parser.add_argument("--flush",           type=int, default=SQL_FLUSH)
    parser.add_argument("--chunk",           type=int, default=SUBMIT_CHUNK)
    parser.add_argument("--chroma-batch",    type=int, default=CHROMA_BATCH)
    parser.add_argument("--limit",           type=int, default=0)
    parser.add_argument("--limit-sequences", type=int, default=0)
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--summary-only",    action="store_true")
    parser.add_argument("--reports-only",    action="store_true")
    parser.add_argument("--images-only",     action="store_true")
    parser.add_argument("--skip-reports",    action="store_true")
    parser.add_argument("--skip-images",     action="store_true")

    args = parser.parse_args()

    dicom_root  = Path(args.root)
    db_path     = Path(args.db)
    reports_csv = Path(args.reports)
    labeled_csv = Path(args.labeled_csv)
    chroma_path = Path(args.chromadb)

    log.info(f"DICOM root     : {dicom_root}")
    log.info(f"SQLite DB      : {db_path}")
    log.info(f"Reports CSV    : {reports_csv}")
    log.info(f"Labeled CSV    : {labeled_csv}")
    log.info(f"ChromaDB path  : {chroma_path}")
    log.info(f"Workers        : {args.workers}")
    log.info(f"Flush size     : {args.flush}")
    log.info(f"Chunk size     : {args.chunk}")
    log.info(f"Chroma batch   : {args.chroma_batch}")
    log.info(f"Limit (meta)   : {args.limit or 'none'}")
    log.info(f"Limit (seqs)   : {args.limit_sequences or 'none'}")
    log.info(f"Dry run        : {args.dry_run}")

    if args.summary_only:
        db_summary(open_db(db_path))
        return

    con = open_db(db_path) if not args.dry_run else None

    run_metadata = not args.images_only and not args.reports_only
    run_reports  = not args.images_only and not args.skip_reports
    run_images   = not args.skip_images

    if run_reports and con is not None:
        ingest_reports(con, reports_csv)

    if args.reports_only:
        if con:
            db_summary(con)
            con.close()
        return

    # ── DICOM metadata ───────────────────────────────────────────────────────
    if run_metadata:
        if not dicom_root.exists():
            log.error(f"DICOM root does not exist: {dicom_root}")
            raise SystemExit(1)

        # ── Load existing SOPInstanceUIDs for skip check ─────────────────────
        # Shipped to every worker process once via _init_worker.
        # Workers do a minimal single-tag pydicom read; if the UID is in
        # this frozenset the full 80-field parse is skipped entirely.
        existing_uids: frozenset = frozenset()
        if con is not None:
            log.info("Loading existing SOPInstanceUIDs from SQLite …")
            existing_uids = frozenset(
                row[0]
                for row in con.execute(
                    "SELECT sop_instance_uid FROM dicom_files "
                    "WHERE sop_instance_uid IS NOT NULL"
                ).fetchall()
            )
            log.info(
                f"  {len(existing_uids):,} UIDs loaded — "
                "matching files will skip full parse"
            )

        log.info("Collecting .dcm file paths …")
        all_paths = list(iter_dicom_files(dicom_root))
        if args.limit:
            all_paths = all_paths[: args.limit]
        total = len(all_paths)
        log.info(f"Found {total:,} .dcm files")

        if total == 0:
            log.warning("No .dcm files found — skipping metadata ingestion.")
        else:
            inserted  = 0
            duplicate = 0
            skipped   = 0
            errored   = 0
            buffer: list[dict] = []

            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,    # runs once per worker process
                initargs=(existing_uids,),   # frozenset shipped here
            ) as pool:
                with tqdm(
                    total=total, unit="file",
                    desc="Ingesting DICOM metadata",
                    dynamic_ncols=True,
                ) as pbar:

                    for path_chunk in chunked(all_paths, args.chunk):
                        futures = {
                            pool.submit(parse_dicom_file, p): p
                            for p in path_chunk
                        }

                        for fut in as_completed(futures):
                            try:
                                result = fut.result()
                            except Exception as exc:
                                log.warning(f"Worker crashed: {exc}")
                                skipped += 1
                                pbar.update(1)
                                continue

                            if result is None:
                                # Either not DICOM, no SOPInstanceUID,
                                # or UID already in DB (quick skip)
                                skipped += 1
                            else:
                                if result.get("parse_error"):
                                    errored += 1
                                if not args.dry_run:
                                    buffer.append(result)

                            if con and len(buffer) >= args.flush:
                                new, ign = flush_batch(con, buffer)
                                inserted  += new
                                duplicate += ign
                                buffer = []

                            pbar.update(1)
                            pbar.set_postfix(
                                ins=inserted, dup=duplicate,
                                skip=skipped, err=errored,
                            )

            if con and buffer:
                new, ign = flush_batch(con, buffer)
                inserted  += new
                duplicate += ign

            log.info("─" * 60)
            log.info(f"Files found         : {total:>10,}")
            log.info(f"Rows inserted       : {inserted:>10,}")
            log.info(f"Duplicates skipped  : {duplicate:>10,}  (INSERT OR IGNORE)")
            log.info(f"No-UID / pre-exist  : {skipped:>10,}  (not DICOM, no UID, or already in DB)")
            log.info(f"Parse errors stored : {errored:>10,}  (stub rows with parse_error set)")
            if args.dry_run:
                log.info("(dry run — nothing written to SQLite)")

    # ── Image ingestion into ChromaDB ────────────────────────────────────────
    if run_images and con is not None:
        ingest_images_to_chromadb(
            con               = con,
            labeled_csv       = labeled_csv,
            dicom_root        = dicom_root,
            chroma_path       = chroma_path,
            limit_sequences   = args.limit_sequences,
            chroma_batch_size = args.chroma_batch,
        )

    if con:
        db_summary(con)
        con.close()


if __name__ == "__main__":
    main()