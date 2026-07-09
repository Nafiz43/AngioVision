#!/usr/bin/env python3
"""
Generate a tiny synthetic DICOM dataset to smoke-test the pipeline
end-to-end. Everything is written under sample_data/ (gitignored):

    sample_data/raw/           - 6 small DICOM files across 2 study dirs
    sample_data/reports.csv    - accession manifest for step 05

Deliberately includes cases that exercise every step:
    - 3 files that pass the strict filter (4 frames, 16x16, GR/DSA/STATIC)
    - 1 file failing SeriesDescription (passes only in relaxed mode)
    - 1 file failing RadiationSetting
    - 1 duplicate SOPInstanceUID + 1 file missing AccessionNumber
      (flagged by step 00)
    - reports.csv lists one accession with no DICOM data
      (flagged by step 05)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PIPELINE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PIPELINE_DIR / "sample_data" / "raw"
REPORTS_CSV = PIPELINE_DIR / "sample_data" / "reports.csv"

FRAMES = 4
SIZE = 16


def make_dicom(
    path: Path,
    accession: str,
    study_uid: str,
    sop_uid: str | None = None,
    radiation: str = "GR",
    series_desc: str = "DSA ABDOMEN",
    motion: str = "STATIC",
    series_date: str = "20240101",
    acq_time: str = "120000",
    frames: int = FRAMES,
) -> str:
    sop_uid = sop_uid or generate_uid()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    if accession:
        ds.AccessionNumber = accession
    ds.Modality = "XA"
    ds.RadiationSetting = radiation
    ds.SeriesDescription = series_desc
    ds.PositionerMotion = motion
    ds.SeriesDate = series_date
    ds.AcquisitionTime = acq_time
    ds.ContentDate = series_date
    ds.PatientName = "Dummy^Patient"
    ds.PatientID = "DUMMY001"

    rng = np.random.default_rng(abs(hash(sop_uid)) % (2**32))
    pixels = rng.integers(0, 255, size=(frames, SIZE, SIZE), dtype=np.uint8)
    ds.NumberOfFrames = frames
    ds.Rows = SIZE
    ds.Columns = SIZE
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return sop_uid


def main() -> int:
    study_a = generate_uid()
    study_b = generate_uid()

    # Study A: two clean strict-passing sequences (different times → ordering).
    # seq2 has 12 frames — inside the [10, 60] outlier range, so the
    # outlier post-check must copy every mosaic EXCEPT this one.
    make_dicom(RAW_DIR / "studyA" / "seq1.dcm", "ACC001", study_a,
               series_date="20240101", acq_time="090000")
    make_dicom(RAW_DIR / "studyA" / "seq2.dcm", "ACC001", study_a,
               series_date="20240101", acq_time="110000", frames=12)

    # Study B: one strict-passing sequence
    make_dicom(RAW_DIR / "studyB" / "seq1.dcm", "ACC002", study_b,
               series_desc="CO 2 RUNOFF")

    # Fails SeriesDescription — extracted only in relaxed mode
    make_dicom(RAW_DIR / "studyB" / "seq2_fluoro.dcm", "ACC002", study_b,
               series_desc="FLUORO SPOT")

    # Fails RadiationSetting — filtered in both modes
    make_dicom(RAW_DIR / "studyB" / "seq3_sc.dcm", "ACC002", study_b,
               radiation="SC")

    # Duplicate SOPInstanceUID (copy of an existing UID) + missing accession
    # → both flagged by step 00's consistency check
    dup_uid = make_dicom(RAW_DIR / "studyC" / "dup_orig.dcm", "ACC003",
                         generate_uid())
    make_dicom(RAW_DIR / "studyC" / "dup_copy.dcm", "ACC003",
               generate_uid(), sop_uid=dup_uid)
    make_dicom(RAW_DIR / "studyC" / "no_accession.dcm", "", generate_uid())

    # Reports manifest: ACC999 has no DICOM data → step 05 must flag it
    REPORTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Anon Acc #", "Report Text"])
        w.writerow(["ACC001", "Abdominal DSA, unremarkable."])
        w.writerow(["ACC002", "CO2 runoff study."])
        w.writerow(["ACC999", "Expected but never acquired."])

    n_files = sum(1 for _ in RAW_DIR.rglob("*.dcm"))
    print(f"Dummy dataset ready: {n_files} DICOM files under {RAW_DIR}")
    print(f"Reports manifest   : {REPORTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
