"""SQLite schema, column list, and prepared INSERT statements."""

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
-- One row per (DICOM sequence, embedding model). One .dcm file = one sequence.
-- sequence_id     = the DICOM file stem (guaranteed unique per file).
-- embedding_model = the EMBEDDING_MODELS key used to build that collection, so a
--                   sequence can be independently ingested by several models
--                   (each into its own ChromaDB collection) without clobbering
--                   another model's resume state.
-- ChromaDB frame IDs are "{sequence_id}_f{frame_idx:06d}" within each model's
-- collection, so partial ingestion from a crashed run can be cleaned up
-- deterministically on retry.
CREATE TABLE IF NOT EXISTS image_ingestion_status (
    sequence_id      TEXT NOT NULL,
    embedding_model  TEXT NOT NULL DEFAULT 'rad-dino',
    accession_number TEXT,
    series_uid       TEXT,
    source_path      TEXT,
    frames_ingested  INTEGER DEFAULT 0,
    status           TEXT,      -- 'in_progress' | 'completed' | 'error'
    error_msg        TEXT,
    ingested_at      TEXT,
    PRIMARY KEY (sequence_id, embedding_model)
);

CREATE INDEX IF NOT EXISTS idx_img_status    ON image_ingestion_status (status);
CREATE INDEX IF NOT EXISTS idx_img_accession ON image_ingestion_status (accession_number);
CREATE INDEX IF NOT EXISTS idx_img_model     ON image_ingestion_status (embedding_model);
"""

# Migration applied to pre-existing databases whose image_ingestion_status table
# was created before per-embedding-model tracking existed (sequence_id was the
# sole PRIMARY KEY). Rebuilds the table with the composite key and tags all
# existing rows as 'rad-dino' (the only model used before this change). Idempotent:
# store.open_db() only runs it when the embedding_model column is absent.
MIGRATE_IMAGE_STATUS_ADD_MODEL = """
ALTER TABLE image_ingestion_status RENAME TO _img_status_old;
CREATE TABLE image_ingestion_status (
    sequence_id      TEXT NOT NULL,
    embedding_model  TEXT NOT NULL DEFAULT 'rad-dino',
    accession_number TEXT,
    series_uid       TEXT,
    source_path      TEXT,
    frames_ingested  INTEGER DEFAULT 0,
    status           TEXT,
    error_msg        TEXT,
    ingested_at      TEXT,
    PRIMARY KEY (sequence_id, embedding_model)
);
INSERT INTO image_ingestion_status
    (sequence_id, embedding_model, accession_number, series_uid, source_path,
     frames_ingested, status, error_msg, ingested_at)
SELECT sequence_id, 'rad-dino', accession_number, series_uid, source_path,
       frames_ingested, status, error_msg, ingested_at
FROM _img_status_old;
DROP TABLE _img_status_old;
CREATE INDEX IF NOT EXISTS idx_img_status    ON image_ingestion_status (status);
CREATE INDEX IF NOT EXISTS idx_img_accession ON image_ingestion_status (accession_number);
CREATE INDEX IF NOT EXISTS idx_img_model     ON image_ingestion_status (embedding_model);
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
