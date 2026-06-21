"""SQLite access helpers, cached DB statistics, and SQL cleanup."""

import re
import sqlite3
import logging
from typing import Any, Dict, List

from .state import state

log = logging.getLogger(__name__)


def open_db() -> sqlite3.Connection:
    """Open a read-write connection to the DICOM SQLite database (Row factory)."""
    con = sqlite3.connect(str(state.db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def run_sql_query(sql: str) -> List[Dict[str, Any]]:
    """
    Execute a read-only SQL query against the database and return results.

    Raises:
        sqlite3.Error: On SQL syntax or execution errors
    """
    with state.lock:
        con = open_db()
        try:
            con.execute("PRAGMA query_only = ON")
            cur  = con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            con.close()


def clean_sql(raw_sql: str) -> str:
    """Remove markdown code fence markers from SQL text."""
    sql = re.sub(r"```(?:sql|sqlite)?\s*", "", raw_sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    return sql.strip()


def get_db_stats() -> Dict[str, Any]:
    """
    Compute and cache database statistics (instance, patient, study, series counts,
    error count, top modalities, and radiology-report linkage coverage).
    """
    if state.db_stats_cache:
        return state.db_stats_cache
    with state.lock:
        con = open_db()
        try:
            total = con.execute(
                "SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            patients = con.execute(
                "SELECT COUNT(DISTINCT patient_id) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            studies = con.execute(
                "SELECT COUNT(DISTINCT study_instance_uid) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            series = con.execute(
                "SELECT COUNT(DISTINCT series_instance_uid) FROM dicom_files WHERE parse_error IS NULL"
            ).fetchone()[0]
            errors = con.execute(
                "SELECT COUNT(*) FROM dicom_files WHERE parse_error IS NOT NULL"
            ).fetchone()[0]
            modalities = con.execute(
                "SELECT modality, COUNT(*) as cnt FROM dicom_files "
                "WHERE parse_error IS NULL GROUP BY modality ORDER BY cnt DESC LIMIT 5"
            ).fetchall()

            rpt_table_exists = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='radiology_reports'"
            ).fetchone()[0]
            if rpt_table_exists:
                rpt_total = con.execute("SELECT COUNT(*) FROM radiology_reports").fetchone()[0]
                rpt_linked = con.execute(
                    "SELECT COUNT(DISTINCT d.accession_number) "
                    "FROM dicom_files d "
                    "JOIN radiology_reports r USING (accession_number) "
                    "WHERE d.parse_error IS NULL"
                ).fetchone()[0]
                rpt_unlinked = con.execute(
                    "SELECT COUNT(DISTINCT d.accession_number) "
                    "FROM dicom_files d "
                    "LEFT JOIN radiology_reports r USING (accession_number) "
                    "WHERE d.parse_error IS NULL "
                    "  AND r.accession_number IS NULL "
                    "  AND d.accession_number NOT LIKE 'MISSING_%'"
                ).fetchone()[0]
            else:
                rpt_total = rpt_linked = rpt_unlinked = 0

            state.db_stats_cache = {
                "instances": total,
                "patients": patients,
                "studies": studies,
                "series": series,
                "errors": errors,
                "modalities": [{"modality": r[0] or "?", "count": r[1]} for r in modalities],
                "db_path": str(state.db_path),
                "rpt_total": rpt_total,
                "rpt_linked": rpt_linked,
                "rpt_unlinked": rpt_unlinked,
            }
            return state.db_stats_cache
        finally:
            con.close()
