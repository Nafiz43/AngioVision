#!/usr/bin/env python3
"""
Neo4j DICOM Graph Wipe Utility

Removes ALL nodes and relationships created by the DICOM ingestion pipeline:
  (:Patient), (:Study), (:Series), (:Instance), (:AccessionNumber)
  and all relationships between them.

Also drops all indexes and constraints created by the ingestion schema.

Optionally resets the SQLite staging DB so Phase 2 can re-ingest from scratch.

Usage:
    python wipe_neo4j.py                        # dry run — shows counts only
    python wipe_neo4j.py --confirm              # actually wipes
    python wipe_neo4j.py --confirm --reset-db   # wipes Neo4j + resets SQLite
    python wipe_neo4j.py --confirm --labels Instance Series  # wipe specific labels only

WARNING: This is irreversible. The SQLite staging DB is your safety net.
         Never wipe SQLite without a backup unless you want to re-parse 800K files.
"""

import sys
import logging
import argparse

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j driver not installed.  Run: pip install neo4j")
    sys.exit(1)

try:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
except ImportError:
    NEO4J_URI      = "bolt://localhost:7687"
    NEO4J_USER     = "neo4j"
    NEO4J_PASSWORD = "neo4j-admin"
    NEO4J_DATABASE = "neo4j"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# All labels created by the ingestion pipeline
ALL_LABELS = ["Instance", "Series", "Study", "Patient", "AccessionNumber"]

# All constraints created by the ingestion schema (drop before indexes)
CONSTRAINTS = [
    "accession_unique",
    "patient_id_unique",
    "study_uid_unique",
    "series_uid_unique",
    "instance_uid_unique",
]

# All indexes created by the ingestion schema
INDEXES = [
    "study_accession_idx",
    "study_date_idx",
    "instance_acq_date",
    "series_modality_idx",
    "patient_sex_idx",
    "instance_frame_count_idx",
    "series_description_idx",
    "instance_source_path_idx",
]

# Delete in this order to respect relationship directions
# (children first avoids deleting a parent before its children are gone)
DELETE_ORDER = ["Instance", "Series", "Study", "AccessionNumber", "Patient"]


def count_nodes(session, label: str) -> int:
    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
    return result.single()["c"]


def count_relationships(session) -> int:
    result = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
    return result.single()["c"]


def wipe_label(session, label: str, batch_size: int = 10_000) -> int:
    """
    Delete all nodes of a given label in batches.
    Detaches relationships automatically (DETACH DELETE).
    Returns total nodes deleted.
    """
    deleted = 0
    while True:
        result = session.run(
            f"MATCH (n:{label}) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) AS d"
        )
        d = result.single()["d"]
        deleted += d
        log.info(f"  {label}: deleted {deleted:,} so far...")
        if d == 0:
            break
    return deleted


def drop_constraints(session):
    for name in CONSTRAINTS:
        try:
            session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            log.info(f"  Dropped constraint: {name}")
        except Exception as e:
            log.warning(f"  Could not drop constraint {name}: {e}")


def drop_indexes(session):
    for name in INDEXES:
        try:
            session.run(f"DROP INDEX {name} IF EXISTS")
            log.info(f"  Dropped index: {name}")
        except Exception as e:
            log.warning(f"  Could not drop index {name}: {e}")


def reset_sqlite(db_path: str):
    """
    Reset neo4j_ingested_at on all rows so Phase 2 will re-ingest everything.
    Does NOT delete SQLite rows — your parsed metadata is preserved.
    """
    import sqlite3
    try:
        con = sqlite3.connect(db_path)
        result = con.execute("UPDATE dicom_files SET neo4j_ingested_at = NULL")
        con.commit()
        log.info(f"SQLite reset: {result.rowcount:,} rows marked for re-ingestion")
        con.close()
    except Exception as e:
        log.error(f"SQLite reset failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Wipe DICOM nodes/relationships/indexes from Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wipe_neo4j.py                            # dry run — counts only
  python wipe_neo4j.py --confirm                  # full wipe
  python wipe_neo4j.py --confirm --labels Instance Series  # partial wipe
  python wipe_neo4j.py --confirm --reset-db       # wipe + reset SQLite for re-ingest
  python wipe_neo4j.py --confirm --keep-schema    # wipe data but keep indexes
        """,
    )
    parser.add_argument("--uri",         default=NEO4J_URI)
    parser.add_argument("--user",        default=NEO4J_USER)
    parser.add_argument("--password",    default=NEO4J_PASSWORD)
    parser.add_argument("--database",    default=NEO4J_DATABASE)
    parser.add_argument("--confirm",     action="store_true",
                        help="Actually perform the wipe (default is dry run)")
    parser.add_argument("--labels",      nargs="+", default=None,
                        choices=ALL_LABELS, metavar="LABEL",
                        help=f"Wipe only these labels (default: all). Choices: {ALL_LABELS}")
    parser.add_argument("--keep-schema", action="store_true",
                        help="Keep indexes and constraints (only delete nodes/rels)")
    parser.add_argument("--reset-db",    action="store_true",
                        help="Also reset SQLite neo4j_ingested_at so Phase 2 re-ingests")
    parser.add_argument("--db",          default=None,
                        help="SQLite DB path (required if --reset-db is used)")
    parser.add_argument("--batch-size",  type=int, default=10_000,
                        help="Nodes deleted per transaction (default: 10000)")
    args = parser.parse_args()

    labels_to_wipe = args.labels if args.labels else DELETE_ORDER

    log.info(f"Neo4j: {args.uri}  db={args.database}")
    log.info(f"Labels to wipe: {labels_to_wipe}")
    log.info(f"Keep schema:    {args.keep_schema}")
    log.info(f"Reset SQLite:   {args.reset_db}")
    log.info(f"Confirmed:      {args.confirm}")

    if args.reset_db and not args.db:
        log.error("--reset-db requires --db <path_to_sqlite.db>")
        raise SystemExit(1)

    # Connect
    try:
        driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
        driver.verify_connectivity()
        log.info("Neo4j connection OK.")
    except Exception as e:
        log.error(f"Cannot connect to Neo4j: {e}")
        raise SystemExit(1)

    with driver.session(database=args.database) as session:

        # ── Show current counts ───────────────────────────────────────
        log.info("─" * 60)
        log.info("Current node counts:")
        total_nodes = 0
        for label in ALL_LABELS:
            n = count_nodes(session, label)
            log.info(f"  {label:20s}: {n:>10,}")
            total_nodes += n
        total_rels = count_relationships(session)
        log.info(f"  {'Relationships':20s}: {total_rels:>10,}")
        log.info(f"  {'TOTAL nodes':20s}: {total_nodes:>10,}")
        log.info("─" * 60)

        if not args.confirm:
            log.info("DRY RUN — pass --confirm to actually wipe.")
            driver.close()
            return

        # ── Safety prompt ─────────────────────────────────────────────
        print(f"\n⚠️  About to permanently delete:")
        print(f"   Labels : {labels_to_wipe}")
        print(f"   Nodes  : ~{total_nodes:,}")
        print(f"   Rels   : ~{total_rels:,}")
        if not args.keep_schema:
            print(f"   Schema : {len(CONSTRAINTS)} constraints + {len(INDEXES)} indexes")
        print()
        answer = input("Type 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            log.info("Aborted.")
            driver.close()
            return

        # ── Delete nodes (batched DETACH DELETE) ──────────────────────
        log.info("Deleting nodes...")
        total_deleted = 0
        for label in labels_to_wipe:
            log.info(f"Wiping {label}...")
            n = wipe_label(session, label, args.batch_size)
            log.info(f"  {label}: {n:,} nodes deleted.")
            total_deleted += n

        # ── Drop schema ───────────────────────────────────────────────
        if not args.keep_schema:
            log.info("Dropping constraints...")
            drop_constraints(session)
            log.info("Dropping indexes...")
            drop_indexes(session)

        # ── Verify ───────────────────────────────────────────────────
        log.info("─" * 60)
        log.info("Post-wipe verification:")
        remaining = 0
        for label in ALL_LABELS:
            n = count_nodes(session, label)
            log.info(f"  {label:20s}: {n:>10,}")
            remaining += n
        rels_left = count_relationships(session)
        log.info(f"  {'Relationships':20s}: {rels_left:>10,}")
        if remaining == 0 and rels_left == 0:
            log.info("✓ Graph is clean.")
        else:
            log.warning(f"⚠ {remaining:,} nodes and {rels_left:,} relationships still present.")

    driver.close()

    # ── Reset SQLite ──────────────────────────────────────────────────
    if args.reset_db:
        log.info(f"Resetting SQLite DB: {args.db}")
        reset_sqlite(args.db)
        log.info("SQLite reset complete. Run dicom_ingest.py --phase 2 to re-ingest.")

    log.info("─" * 60)
    log.info(f"Wipe complete. Deleted {total_deleted:,} nodes.")


if __name__ == "__main__":
    main()