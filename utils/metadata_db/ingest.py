#!/usr/bin/env python3
"""
DICOM → Neo4j Ingestion Pipeline
Recursively walks /data/Deep_Angiography/DICOM, extracts metadata from every
.dcm file, and loads it into Neo4j as a property graph.

Graph schema:
  (:Patient {patient_id, patient_name, patient_sex, patient_age})
      -[:UNDERWENT]->
  (:Study {study_instance_uid, study_date, study_time, study_description,
           accession_number, referring_physician})
      -[:HAS_SERIES]->
  (:Series {series_instance_uid, series_number, series_date, series_time,
            series_description, modality, body_part, protocol_name})
      -[:HAS_INSTANCE]->
  (:Instance {sop_instance_uid, sop_class_uid, instance_number,
              acquisition_date, acquisition_time, content_date, content_time,
              image_type, rows, columns, bits_allocated, bits_stored,
              number_of_frames, frame_time, cine_rate, frame_count,
              kvp, exposure_time, xray_tube_current, avg_pulse_width,
              distance_source_to_detector, distance_source_to_patient,
              radiation_setting, radiation_mode, dose_product,
              intensifier_size, focal_spots, imager_pixel_spacing,
              positioner_motion, positioner_primary_angle, positioner_secondary_angle,
              patient_position, window_center, window_width,
              lossy_image_compression, photometric_interpretation,
              samples_per_pixel, pixel_representation,
              contrast_bolus_agent, contrast_bolus_ingredient,
              manufacturer, manufacturer_model, station_name,
              software_versions, device_serial_number,
              detector_id, detector_description,
              specific_character_set, longitudinal_temporal_info_modified,
              source_file, source_path, form_type})

  (:AccessionNumber {value})   ← primary key node for easy lookup
      -[:BELONGS_TO]-> (:Study)

Requirements:
    pip install pydicom neo4j tqdm
    Neo4j running and config.py present with NEO4J_URI/USER/PASSWORD/DATABASE
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print("ERROR: pydicom not installed.  Run: pip install pydicom")
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j driver not installed.  Run: pip install neo4j")
    sys.exit(1)

try:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
except ImportError:
    # Fallback defaults — override via CLI or config.py
    NEO4J_URI      = "bolt://localhost:7687"
    NEO4J_USER     = "neo4j"
    NEO4J_PASSWORD = "neo4j-admin"
    NEO4J_DATABASE = "neo4j"

# ── Constants ─────────────────────────────────────────────────────────────────
DICOM_ROOT = Path("/data/Deep_Angiography/DICOM")
BATCH_SIZE = 200   # instances per transaction; lower if RAM-constrained

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA_QUERIES = [
    # Primary key / uniqueness constraints
    "CREATE CONSTRAINT accession_unique   IF NOT EXISTS FOR (a:AccessionNumber) REQUIRE a.value IS UNIQUE",
    "CREATE CONSTRAINT patient_id_unique  IF NOT EXISTS FOR (p:Patient)         REQUIRE p.patient_id IS UNIQUE",
    "CREATE CONSTRAINT study_uid_unique   IF NOT EXISTS FOR (s:Study)           REQUIRE s.study_instance_uid IS UNIQUE",
    "CREATE CONSTRAINT series_uid_unique  IF NOT EXISTS FOR (r:Series)          REQUIRE r.series_instance_uid IS UNIQUE",
    "CREATE CONSTRAINT instance_uid_unique IF NOT EXISTS FOR (i:Instance)       REQUIRE i.sop_instance_uid IS UNIQUE",
    # Lookup indexes
    "CREATE INDEX study_accession_idx  IF NOT EXISTS FOR (s:Study)    ON (s.accession_number)",
    "CREATE INDEX study_date_idx       IF NOT EXISTS FOR (s:Study)    ON (s.study_date)",
    "CREATE INDEX instance_acq_date    IF NOT EXISTS FOR (i:Instance) ON (i.acquisition_date)",
    "CREATE INDEX series_modality_idx  IF NOT EXISTS FOR (r:Series)   ON (r.modality)",
    "CREATE INDEX patient_sex_idx      IF NOT EXISTS FOR (p:Patient)  ON (p.patient_sex)",
]

# ── Cypher upsert ─────────────────────────────────────────────────────────────
INGEST_QUERY = """
UNWIND $rows AS row

// ── AccessionNumber node (primary key anchor) ─────────────────
MERGE (acc:AccessionNumber {value: row.accession_number})

// ── Patient ───────────────────────────────────────────────────
MERGE (p:Patient {patient_id: row.patient_id})
SET   p.patient_name       = row.patient_name,
      p.patient_sex        = row.patient_sex,
      p.patient_age        = row.patient_age,
      p.pregnancy_status   = row.pregnancy_status,
      p.identity_removed   = row.patient_identity_removed,
      p.deidentification   = row.deidentification_method

// ── Study ─────────────────────────────────────────────────────
MERGE (st:Study {study_instance_uid: row.study_instance_uid})
SET   st.accession_number         = row.accession_number,
      st.study_date               = row.study_date,
      st.study_time               = row.study_time,
      st.study_description        = row.study_description,
      st.referring_physician      = row.referring_physician,
      st.requested_procedure_desc = row.requested_procedure_description,
      st.performed_step_date      = row.performed_procedure_step_start_date,
      st.performed_step_time      = row.performed_procedure_step_start_time,
      st.performed_step_desc      = row.performed_procedure_step_description

// ── Series ────────────────────────────────────────────────────
MERGE (se:Series {series_instance_uid: row.series_instance_uid})
SET   se.series_number      = row.series_number,
      se.series_date        = row.series_date,
      se.series_time        = row.series_time,
      se.series_description = row.series_description,
      se.modality           = row.modality,
      se.protocol_name      = row.protocol_name,
      se.acquisition_number = row.acquisition_number

// ── Instance ──────────────────────────────────────────────────
MERGE (ins:Instance {sop_instance_uid: row.sop_instance_uid})
SET   ins.sop_class_uid                    = row.sop_class_uid,
      ins.instance_number                  = row.instance_number,
      ins.image_type                       = row.image_type,
      ins.acquisition_date                 = row.acquisition_date,
      ins.acquisition_time                 = row.acquisition_time,
      ins.content_date                     = row.content_date,
      ins.content_time                     = row.content_time,
      // Image geometry
      ins.rows                             = toInteger(row.rows),
      ins.columns                          = toInteger(row.columns),
      ins.bits_allocated                   = toInteger(row.bits_allocated),
      ins.bits_stored                      = toInteger(row.bits_stored),
      ins.high_bit                         = toInteger(row.high_bit),
      ins.samples_per_pixel                = toInteger(row.samples_per_pixel),
      ins.pixel_representation             = row.pixel_representation,
      ins.photometric_interpretation       = row.photometric_interpretation,
      ins.number_of_frames                 = toInteger(row.number_of_frames),
      ins.frame_count                      = toInteger(row.frame_count),
      ins.frame_time                       = toFloat(row.frame_time),
      ins.cine_rate                        = toInteger(row.cine_rate),
      ins.images_in_acquisition            = toInteger(row.images_in_acquisition),
      ins.representative_frame_number      = toInteger(row.representative_frame_number),
      ins.start_trim                       = toInteger(row.start_trim),
      ins.stop_trim                        = toInteger(row.stop_trim),
      ins.recommended_display_frame_rate   = toInteger(row.recommended_display_frame_rate),
      // Acquisition / radiation
      ins.kvp                              = toFloat(row.kvp),
      ins.exposure_time                    = toInteger(row.exposure_time),
      ins.xray_tube_current                = toInteger(row.xray_tube_current),
      ins.avg_pulse_width                  = toFloat(row.avg_pulse_width),
      ins.radiation_setting                = row.radiation_setting,
      ins.radiation_mode                   = row.radiation_mode,
      ins.dose_product                     = toFloat(row.dose_product),
      // Geometry
      ins.distance_source_to_detector      = toFloat(row.distance_source_to_detector),
      ins.distance_source_to_patient       = toFloat(row.distance_source_to_patient),
      ins.est_magnification_factor         = toFloat(row.est_magnification_factor),
      ins.intensifier_size                 = toFloat(row.intensifier_size),
      ins.imager_pixel_spacing             = row.imager_pixel_spacing,
      ins.focal_spots                      = row.focal_spots,
      ins.positioner_motion                = row.positioner_motion,
      ins.positioner_primary_angle         = toFloat(row.positioner_primary_angle),
      ins.positioner_secondary_angle       = toFloat(row.positioner_secondary_angle),
      ins.patient_position                 = row.patient_position,
      // Display
      ins.window_center                    = toFloat(row.window_center),
      ins.window_width                     = toFloat(row.window_width),
      ins.voi_lut_function                 = row.voi_lut_function,
      ins.lossy_image_compression          = row.lossy_image_compression,
      ins.longitudinal_temporal_info       = row.longitudinal_temporal_info_modified,
      ins.pixel_intensity_relationship     = row.pixel_intensity_relationship,
      // Contrast
      ins.contrast_bolus_agent             = row.contrast_bolus_agent,
      ins.contrast_bolus_ingredient        = row.contrast_bolus_ingredient,
      // Equipment
      ins.manufacturer                     = row.manufacturer,
      ins.manufacturer_model               = row.manufacturer_model_name,
      ins.station_name                     = row.station_name,
      ins.software_versions                = row.software_versions,
      ins.device_serial_number             = row.device_serial_number,
      ins.detector_id                      = row.detector_id,
      ins.detector_description             = row.detector_description,
      ins.specific_character_set           = row.specific_character_set,
      // File location
      ins.source_file                      = row.source_file,
      ins.source_path                      = row.source_path

// ── Relationships ─────────────────────────────────────────────
MERGE (p)-[:UNDERWENT]->(st)
MERGE (acc)-[:BELONGS_TO]->(st)
MERGE (st)-[:HAS_SERIES]->(se)
MERGE (se)-[:HAS_INSTANCE]->(ins)
"""

# ── DICOM tag extraction ───────────────────────────────────────────────────────
def safe_str(ds, tag_name: str, default: str = "") -> str:
    """Safely get a DICOM tag value as a clean string."""
    try:
        val = getattr(ds, tag_name, None)
        if val is None:
            return default
        # Handle sequences — skip them
        if hasattr(val, "__iter__") and not isinstance(val, str):
            try:
                return str(val).strip()
            except Exception:
                return default
        return str(val).strip()
    except Exception:
        return default

def extract_metadata(dcm_path: Path) -> Optional[dict]:
    """Read a .dcm file and return a flat metadata dict, or None on failure."""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=False)
    except InvalidDicomError:
        log.debug(f"Not a valid DICOM file (skipping): {dcm_path}")
        return None
    except Exception as e:
        log.warning(f"Failed to read {dcm_path}: {e}")
        return None

    # Derive accession number — mandatory primary key
    accession = safe_str(ds, "AccessionNumber")
    if not accession:
        log.debug(f"No AccessionNumber in {dcm_path.name} — skipping")
        return None

    # Frame count: prefer NumberOfFrames tag, fallback to 1
    try:
        frame_count = int(ds.NumberOfFrames)
    except Exception:
        frame_count = 1

    return {
        # ── Keys / UIDs
        "accession_number":           accession,
        "sop_instance_uid":           safe_str(ds, "SOPInstanceUID"),
        "sop_class_uid":              safe_str(ds, "SOPClassUID"),
        "study_instance_uid":         safe_str(ds, "StudyInstanceUID"),
        "series_instance_uid":        safe_str(ds, "SeriesInstanceUID"),
        # ── Patient
        "patient_id":                 safe_str(ds, "PatientID"),
        "patient_name":               safe_str(ds, "PatientName"),
        "patient_sex":                safe_str(ds, "PatientSex"),
        "patient_age":                safe_str(ds, "PatientAge"),
        "pregnancy_status":           safe_str(ds, "PregnancyStatus"),
        "patient_identity_removed":   safe_str(ds, "PatientIdentityRemoved"),
        "deidentification_method":    safe_str(ds, "DeidentificationMethod"),
        # ── Study
        "study_date":                 safe_str(ds, "StudyDate"),
        "study_time":                 safe_str(ds, "StudyTime"),
        "study_description":          safe_str(ds, "StudyDescription"),
        "referring_physician":        safe_str(ds, "ReferringPhysicianName"),
        "requested_procedure_description": safe_str(ds, "RequestedProcedureDescription"),
        "performed_procedure_step_start_date": safe_str(ds, "PerformedProcedureStepStartDate"),
        "performed_procedure_step_start_time": safe_str(ds, "PerformedProcedureStepStartTime"),
        "performed_procedure_step_description": safe_str(ds, "PerformedProcedureStepDescription"),
        # ── Series
        "series_date":                safe_str(ds, "SeriesDate"),
        "series_time":                safe_str(ds, "SeriesTime"),
        "series_description":         safe_str(ds, "SeriesDescription"),
        "series_number":              safe_str(ds, "SeriesNumber"),
        "acquisition_number":         safe_str(ds, "AcquisitionNumber"),
        "modality":                   safe_str(ds, "Modality"),
        "protocol_name":              safe_str(ds, "ProtocolName"),
        # ── Instance / acquisition
        "instance_number":            safe_str(ds, "InstanceNumber"),
        "image_type":                 safe_str(ds, "ImageType"),
        "acquisition_date":           safe_str(ds, "AcquisitionDate"),
        "acquisition_time":           safe_str(ds, "AcquisitionTime"),
        "content_date":               safe_str(ds, "ContentDate"),
        "content_time":               safe_str(ds, "ContentTime"),
        # ── Image geometry
        "rows":                       safe_str(ds, "Rows"),
        "columns":                    safe_str(ds, "Columns"),
        "bits_allocated":             safe_str(ds, "BitsAllocated"),
        "bits_stored":                safe_str(ds, "BitsStored"),
        "high_bit":                   safe_str(ds, "HighBit"),
        "samples_per_pixel":          safe_str(ds, "SamplesPerPixel"),
        "pixel_representation":       safe_str(ds, "PixelRepresentation"),
        "photometric_interpretation": safe_str(ds, "PhotometricInterpretation"),
        "number_of_frames":           safe_str(ds, "NumberOfFrames") or str(frame_count),
        "frame_count":                str(frame_count),
        "frame_time":                 safe_str(ds, "FrameTime"),
        "cine_rate":                  safe_str(ds, "CineRate"),
        "images_in_acquisition":      safe_str(ds, "ImagesInAcquisition"),
        "representative_frame_number":safe_str(ds, "RepresentativeFrameNumber"),
        "start_trim":                 safe_str(ds, "StartTrim"),
        "stop_trim":                  safe_str(ds, "StopTrim"),
        "recommended_display_frame_rate": safe_str(ds, "RecommendedDisplayFrameRate"),
        # ── Radiation / acquisition parameters
        "kvp":                        safe_str(ds, "KVP"),
        "exposure_time":              safe_str(ds, "ExposureTime"),
        "xray_tube_current":          safe_str(ds, "XRayTubeCurrent"),
        "avg_pulse_width":            safe_str(ds, "AveragePulseWidth"),
        "radiation_setting":          safe_str(ds, "RadiationSetting"),
        "radiation_mode":             safe_str(ds, "RadiationMode"),
        "dose_product":               safe_str(ds, "ImageAndFluoroscopyAreaDoseProduct"),
        # ── Geometry
        "distance_source_to_detector":  safe_str(ds, "DistanceSourceToDetector"),
        "distance_source_to_patient":   safe_str(ds, "DistanceSourceToPatient"),
        "est_magnification_factor":     safe_str(ds, "EstimatedRadiographicMagnificationFactor"),
        "intensifier_size":             safe_str(ds, "IntensifierSize"),
        "imager_pixel_spacing":         safe_str(ds, "ImagerPixelSpacing"),
        "focal_spots":                  safe_str(ds, "FocalSpots"),
        "positioner_motion":            safe_str(ds, "PositionerMotion"),
        "positioner_primary_angle":     safe_str(ds, "PositionerPrimaryAngle"),
        "positioner_secondary_angle":   safe_str(ds, "PositionerSecondaryAngle"),
        "patient_position":             safe_str(ds, "PatientPosition"),
        # ── Display / pixel
        "window_center":                safe_str(ds, "WindowCenter"),
        "window_width":                 safe_str(ds, "WindowWidth"),
        "voi_lut_function":             safe_str(ds, "VOILUTFunction"),
        "lossy_image_compression":      safe_str(ds, "LossyImageCompression"),
        "longitudinal_temporal_info_modified": safe_str(ds, "LongitudinalTemporalInformationModified"),
        "pixel_intensity_relationship": safe_str(ds, "PixelIntensityRelationship"),
        # ── Contrast
        "contrast_bolus_agent":         safe_str(ds, "ContrastBolusAgent"),
        "contrast_bolus_ingredient":    safe_str(ds, "ContrastBolusIngredient"),
        # ── Equipment
        "manufacturer":                 safe_str(ds, "Manufacturer"),
        "manufacturer_model_name":      safe_str(ds, "ManufacturerModelName"),
        "station_name":                 safe_str(ds, "StationName"),
        "software_versions":            safe_str(ds, "SoftwareVersions"),
        "device_serial_number":         safe_str(ds, "DeviceSerialNumber"),
        "detector_id":                  safe_str(ds, "DetectorID"),
        "detector_description":         safe_str(ds, "DetectorDescription"),
        "specific_character_set":       safe_str(ds, "SpecificCharacterSet"),
        # ── File location
        "source_file":                  dcm_path.name,
        "source_path":                  str(dcm_path),
    }

# ── File discovery ─────────────────────────────────────────────────────────────
def iter_dicom_files(root: Path):
    """Yield all .dcm files (case-insensitive) under root."""
    for dirpath, _, filenames in os.walk(str(root)):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                yield Path(dirpath) / fname

# ── Neo4j helpers ─────────────────────────────────────────────────────────────
def apply_schema(session):
    for q in SCHEMA_QUERIES:
        try:
            session.run(q)
        except Exception as e:
            log.warning(f"Schema skipped ({e}): {q[:70]}")
    log.info("Schema / constraints applied.")

def ingest_batch(session, batch: list[dict]):
    session.run(INGEST_QUERY, rows=batch)

FATAL_CODES = {
    "Neo.ClientError.Security.Unauthorized",
    "Neo.ClientError.Security.AuthenticationRateLimit",
    "Neo.ClientError.Security.AuthorizationExpired",
    "Neo.ClientError.Security.TokenExpired",
}

# ── Main ───────────────────────────────────────────────────────────────────────
def main(dicom_root: Path, uri: str, user: str, password: str,
         database: str, batch_size: int, dry_run: bool, limit: int):

    log.info(f"DICOM root: {dicom_root}")
    log.info(f"Neo4j:      {uri}  db={database}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Dry run:    {dry_run}  |  Limit: {limit if limit else 'none'}")

    if not dicom_root.exists():
        log.error(f"DICOM root does not exist: {dicom_root}")
        raise SystemExit(1)

    if dry_run:
        log.info("Dry run — scanning files only, no writes.")
        count = 0
        for p in iter_dicom_files(dicom_root):
            meta = extract_metadata(p)
            if meta:
                count += 1
                if limit and count >= limit:
                    break
        log.info(f"Dry run complete. Valid DICOM files found (up to limit): {count:,}")
        return

    # Verify Neo4j connection
    log.info("Verifying Neo4j connection...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        log.info("Neo4j connection OK.")
    except Exception as e:
        log.error(f"Cannot connect to Neo4j: {e}")
        raise SystemExit(1)

    with driver.session(database=database) as session:
        apply_schema(session)

        batch    = []
        ingested = 0
        skipped  = 0
        errors   = 0

        with tqdm(unit="file", dynamic_ncols=True, desc="Ingesting DICOM") as pbar:
            for dcm_path in iter_dicom_files(dicom_root):
                meta = extract_metadata(dcm_path)
                if meta is None:
                    skipped += 1
                    pbar.update(1)
                    continue

                batch.append(meta)

                if len(batch) >= batch_size:
                    try:
                        ingest_batch(session, batch)
                        ingested += len(batch)
                    except Exception as exc:
                        err_str = str(exc)
                        if any(code in err_str for code in FATAL_CODES):
                            log.error(f"FATAL auth error — stopping: {exc}")
                            driver.close()
                            raise SystemExit(1)
                        log.error(f"Batch failed ({len(batch)} rows): {exc}")
                        errors += len(batch)
                    batch = []

                pbar.update(1)
                pbar.set_postfix(ingested=ingested, skipped=skipped, errors=errors)

                if limit and (ingested + errors) >= limit:
                    log.info(f"Reached limit of {limit} — stopping.")
                    break

            # Flush remainder
            if batch:
                try:
                    ingest_batch(session, batch)
                    ingested += len(batch)
                except Exception as exc:
                    log.error(f"Final batch failed: {exc}")
                    errors += len(batch)

    driver.close()

    log.info("─" * 60)
    log.info(f"Done.  Ingested: {ingested:,}  |  Skipped: {skipped:,}  |  Errors: {errors:,}")
    log.info("")
    log.info("Sample Cypher queries to try in Neo4j Browser:")
    log.info("")
    log.info("  // All studies for a given AccessionNumber")
    log.info("  MATCH (acc:AccessionNumber {value:'0BrnGBKrkm'})-[:BELONGS_TO]->(st:Study)")
    log.info("        -[:HAS_SERIES]->(se:Series)-[:HAS_INSTANCE]->(i:Instance)")
    log.info("  RETURN acc, st, se, i LIMIT 25")
    log.info("")
    log.info("  // Instance count per modality")
    log.info("  MATCH (se:Series)-[:HAS_INSTANCE]->(i:Instance)")
    log.info("  RETURN se.modality, count(i) AS instances ORDER BY instances DESC")
    log.info("")
    log.info("  // DSA series with high frame counts")
    log.info("  MATCH (se:Series)-[:HAS_INSTANCE]->(i:Instance)")
    log.info("  WHERE se.series_description CONTAINS 'DSA' AND i.frame_count > 20")
    log.info("  RETURN se.series_description, i.frame_count, i.source_path")
    log.info("  ORDER BY i.frame_count DESC LIMIT 50")
    log.info("")
    log.info("  // Patient → Study → Series tree")
    log.info("  MATCH (p:Patient)-[:UNDERWENT]->(st:Study)-[:HAS_SERIES]->(se:Series)")
    log.info("  RETURN p.patient_id, st.study_description, collect(se.series_description)")
    log.info("  LIMIT 20")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DICOM metadata into Neo4j.")
    parser.add_argument("--root",     type=str, default=str(DICOM_ROOT),   help="Root DICOM directory.")
    parser.add_argument("--uri",      type=str, default=NEO4J_URI,         help="Neo4j bolt URI.")
    parser.add_argument("--user",     type=str, default=NEO4J_USER,        help="Neo4j username.")
    parser.add_argument("--password", type=str, default=NEO4J_PASSWORD,    help="Neo4j password.")
    parser.add_argument("--database", type=str, default=NEO4J_DATABASE,    help="Neo4j database name.")
    parser.add_argument("--batch",    type=int, default=BATCH_SIZE,        help="Instances per transaction.")
    parser.add_argument("--limit",    type=int, default=0,                 help="Stop after N files (0 = no limit).")
    parser.add_argument("--dry-run",  action="store_true",                 help="Scan & parse only, no Neo4j writes.")
    args = parser.parse_args()

    main(
        dicom_root = Path(args.root),
        uri        = args.uri,
        user       = args.user,
        password   = args.password,
        database   = args.database,
        batch_size = args.batch,
        dry_run    = args.dry_run,
        limit      = args.limit,
    )