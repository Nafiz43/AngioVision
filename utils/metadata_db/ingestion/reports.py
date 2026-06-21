"""Radiology report CSV ingestion into the radiology_reports table."""

import csv
import logging
import sqlite3
import datetime
from pathlib import Path

from .schema import INSERT_RPT_SQL

log = logging.getLogger(__name__)


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
