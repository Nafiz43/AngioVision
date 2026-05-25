#!/usr/bin/env python3
"""
DICOM → SQLite Ingestion Pipeline

Recursively walks a DICOM directory, parses metadata from every .dcm file
in parallel (ProcessPoolExecutor), and stores everything in a flat SQLite
table keyed on SOPInstanceUID.

Also ingests a radiology report CSV (Anon Acc #, radrpt) into a separate
`radiology_reports` table, joinable to `dicom_files` via accession_number.

Design principles:
  • SOPInstanceUID is the only skip condition — every parseable file is stored
  • Missing AccessionNumber → synthetic key MISSING_{sop_uid[:12]}
  • INSERT OR IGNORE — re-runs are safe and idempotent
  • Full audit trail: parse_error, sqlite_inserted_at columns
  • Resume-friendly: re-run skips already-inserted UIDs via INSERT OR IGNORE
  • Chunked submission — tqdm starts immediately, no upfront blocking
  • No external services required

Requirements:
    pip install pydicom tqdm
"""

import os
import sys
import csv
import sqlite3
import logging
import argparse
import datetime
import itertools
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

# ── Constants ──────────────────────────────────────────────────────────────────
DICOM_ROOT    = Path("/data/Deep_Angiography/DICOM")
SQLITE_DB     = Path("/data/Deep_Angiography/AngioVision/dicom_staging.db")
REPORTS_CSV   = Path("/data/Deep_Angiography/Reports/Report_List_v01_01_merged_raw.csv")
PARSE_WORKERS = max(1, os.cpu_count() - 1)
SQL_FLUSH     = 1000   # rows buffered before each SQLite commit
SUBMIT_CHUNK  = 2000   # futures submitted per chunk — keeps queue shallow

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
-- One row per accession number; joins to dicom_files on accession_number.
-- Duplicate accession numbers in the CSV are collapsed: the first radrpt
-- encountered is kept (INSERT OR IGNORE), matching the idempotent design of
-- the DICOM ingestion path.
--
-- Join example:
--   SELECT d.accession_number, d.series_description, d.frame_count,
--          r.radrpt
--   FROM   dicom_files d
--   JOIN   radiology_reports r USING (accession_number)
--   WHERE  d.modality = 'XA';
CREATE TABLE IF NOT EXISTS radiology_reports (
    accession_number   TEXT PRIMARY KEY,
    radrpt             TEXT,
    source_csv         TEXT,
    csv_inserted_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_rpt_accession ON radiology_reports (accession_number);
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


# ── DICOM parsing (runs in subprocess) ────────────────────────────────────────
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
    Module-level function — required for ProcessPoolExecutor pickle compatibility.

    Returns:
        dict with parse_error=None  — good row
        dict with parse_error set   — unreadable file stored as audit stub
        None                        — not DICOM or no SOPInstanceUID; skip silently
    """
    path = Path(path_str)
    now  = datetime.datetime.utcnow().isoformat()

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


# ── File discovery ─────────────────────────────────────────────────────────────
def iter_dicom_files(root: Path):
    """Yield absolute path strings for every .dcm file under root."""
    for dirpath, _, filenames in os.walk(str(root)):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                yield str(Path(dirpath) / fname)


# ── Chunked iterator ───────────────────────────────────────────────────────────
def chunked(iterable, size: int):
    """Yield successive fixed-size chunks from any iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


# ── SQLite helpers ─────────────────────────────────────────────────────────────
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


# ── Radiology report ingestion ─────────────────────────────────────────────────
def ingest_reports(con: sqlite3.Connection, csv_path: Path) -> tuple[int, int]:
    """
    Load Report_List_v01_01_merged_raw.csv into radiology_reports.

    Expected columns (case-insensitive):
        Anon Acc #   →  accession_number  (PRIMARY KEY)
        radrpt       →  radrpt

    Duplicate accession numbers in the CSV are silently ignored
    (INSERT OR IGNORE keeps the first occurrence).

    Returns:
        (inserted, ignored) counts
    """
    if not csv_path.exists():
        log.warning(f"Reports CSV not found — skipping: {csv_path}")
        return 0, 0

    now        = datetime.datetime.utcnow().isoformat()
    source     = str(csv_path)
    inserted   = 0
    ignored    = 0
    bad_rows   = 0

    log.info(f"Ingesting radiology reports from: {csv_path}")

    before = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        # utf-8-sig strips the BOM that Excel sometimes adds
        reader = csv.DictReader(fh)

        # Normalise header names so column lookup is case-insensitive
        # and strips surrounding whitespace.
        if reader.fieldnames is None:
            log.error("CSV appears to be empty — no headers found.")
            return 0, 0

        norm = {h.strip().lower(): h for h in reader.fieldnames}
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

    log.info("─" * 60)
    log.info(f"Total DICOM rows    : {total:>10,}")
    log.info(f"Parse errors        : {with_error:>10,}")
    log.info(f"Missing accession   : {missing_acc:>10,}")
    log.info("─" * 60)
    log.info(f"Radiology reports   : {rpt_total:>10,}")
    log.info(f"Accessions linked   : {linked:>10,}  (DICOM ∩ reports)")
    log.info("─" * 60)
    log.info("Modality breakdown:")
    for r in con.execute(
        "SELECT modality, COUNT(*) AS n FROM dicom_files "
        "WHERE parse_error IS NULL GROUP BY modality ORDER BY n DESC LIMIT 15"
    ).fetchall():
        log.info(f"  {(r['modality'] or 'NULL'):12s}  {r['n']:>10,}")
    log.info("─" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Ingest DICOM metadata + radiology reports into SQLite (parallel, resumable).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dicom_ingest_sqlite.py
  python dicom_ingest_sqlite.py --workers 26
  python dicom_ingest_sqlite.py --root /data/DICOM --db /data/meta.db
  python dicom_ingest_sqlite.py --dry-run
  python dicom_ingest_sqlite.py --limit 5000
  python dicom_ingest_sqlite.py --summary-only
  python dicom_ingest_sqlite.py --reports-only
  python dicom_ingest_sqlite.py --reports /path/to/custom_reports.csv

Useful join query after ingestion:
  SELECT d.accession_number, d.series_description, d.frame_count,
         d.modality, d.study_date, r.radrpt
  FROM   dicom_files d
  JOIN   radiology_reports r USING (accession_number)
  WHERE  d.modality = 'XA'
  LIMIT  20;
        """,
    )
    parser.add_argument("--root",         default=str(DICOM_ROOT))
    parser.add_argument("--db",           default=str(SQLITE_DB))
    parser.add_argument("--reports",      default=str(REPORTS_CSV),
                        help="Path to the radiology reports CSV (default: %(default)s)")
    parser.add_argument("--workers",      type=int, default=PARSE_WORKERS)
    parser.add_argument("--flush",        type=int, default=SQL_FLUSH)
    parser.add_argument("--chunk",        type=int, default=SUBMIT_CHUNK,
                        help="Futures submitted per chunk (default: 2000)")
    parser.add_argument("--limit",        type=int, default=0)
    parser.add_argument("--dry-run",      action="store_true",
                        help="Parse DICOM files but write nothing to SQLite")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print DB summary and exit (no ingestion)")
    parser.add_argument("--reports-only", action="store_true",
                        help="Skip DICOM ingestion; only load the reports CSV")
    parser.add_argument("--skip-reports", action="store_true",
                        help="Skip reports CSV ingestion entirely")
    args = parser.parse_args()

    dicom_root  = Path(args.root)
    db_path     = Path(args.db)
    reports_csv = Path(args.reports)

    log.info(f"DICOM root  : {dicom_root}")
    log.info(f"SQLite DB   : {db_path}")
    log.info(f"Reports CSV : {reports_csv}")
    log.info(f"Workers     : {args.workers}")
    log.info(f"Flush size  : {args.flush}")
    log.info(f"Chunk size  : {args.chunk}")
    log.info(f"Limit       : {args.limit or 'none'}")
    log.info(f"Dry run     : {args.dry_run}")

    if args.summary_only:
        db_summary(open_db(db_path))
        return

    con = open_db(db_path) if not args.dry_run else None

    # ── Reports CSV ingestion (fast — runs first) ──────────────────────
    if not args.skip_reports and not args.dry_run:
        ingest_reports(con, reports_csv)

    if args.reports_only:
        if con:
            db_summary(con)
            con.close()
        return

    # ── DICOM ingestion ────────────────────────────────────────────────
    if not dicom_root.exists():
        log.error(f"DICOM root does not exist: {dicom_root}")
        raise SystemExit(1)

    log.info("Collecting .dcm file paths...")
    all_paths = list(iter_dicom_files(dicom_root))
    if args.limit:
        all_paths = all_paths[: args.limit]
    total = len(all_paths)
    log.info(f"Found {total:,} .dcm files")

    if total == 0:
        log.warning("No .dcm files found — nothing to do.")
        if con:
            db_summary(con)
            con.close()
        return

    inserted  = 0
    duplicate = 0
    skipped   = 0
    errored   = 0
    buffer: list[dict] = []

    # ── Key fix: submit in chunks so tqdm starts immediately ───────────
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=total, unit="file", desc="Ingesting DICOM",
                  dynamic_ncols=True) as pbar:

            for path_chunk in chunked(all_paths, args.chunk):

                futures = {pool.submit(parse_dicom_file, p): p
                           for p in path_chunk}

                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                    except Exception as exc:
                        log.warning(f"Worker crashed: {exc}")
                        skipped += 1
                        pbar.update(1)
                        continue

                    if result is None:
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
                        ins=inserted,
                        dup=duplicate,
                        skip=skipped,
                        err=errored,
                    )

    # ── Final flush ────────────────────────────────────────────────────
    if con and buffer:
        new, ign = flush_batch(con, buffer)
        inserted  += new
        duplicate += ign

    log.info("─" * 60)
    log.info(f"Files found         : {total:>10,}")
    log.info(f"Rows inserted       : {inserted:>10,}")
    log.info(f"Duplicates skipped  : {duplicate:>10,}  (INSERT OR IGNORE)")
    log.info(f"No-UID skipped      : {skipped:>10,}  (not DICOM or no SOPInstanceUID)")
    log.info(f"Parse errors stored : {errored:>10,}  (stub rows with parse_error set)")
    if args.dry_run:
        log.info("(dry run — nothing written to SQLite)")
    elif con:
        db_summary(con)
        con.close()


if __name__ == "__main__":
    main()