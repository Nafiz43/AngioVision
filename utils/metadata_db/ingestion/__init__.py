"""
AngioVision ingestion package.

Recursively parses DICOM metadata into SQLite, ingests radiology reports, and
embeds labeled DICOM sequences into ChromaDB with RAD-DINO. Use the
`run_ingestion` orchestrator (see `run_ingest.py` for the CLI).
"""

from . import config
from .pipeline import run_ingestion

__all__ = ["config", "run_ingestion"]
