"""
Step 05 — Accession cross-check: reports list vs consolidated metadata.

Finds accession numbers present in the reports CSV (cfg.reports_csv) that
never appear in the consolidated metadata produced by step 04 — i.e.
expected studies whose DICOM data never made it through the pipeline.

Skips gracefully (with a warning) if no reports CSV is configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from vdp.common import write_csv


def _normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / "05_accession_check"

    if not cfg.reports_csv:
        summary = {"skipped": "no reports_csv configured"}
        print(f"[05] {summary}")
        return summary

    reports_path = Path(cfg.reports_csv)
    consolidated_csv = (run_dir / "04_consolidated_metadata"
                        / "consolidated_metadata_ALL_Sequences.csv")

    if not reports_path.exists():
        summary = {"error": f"reports_csv not found: {reports_path}"}
        print(f"[05] {summary}")
        return summary
    if not consolidated_csv.exists():
        summary = {"error": "consolidated metadata missing — run step 04 first"}
        print(f"[05] {summary}")
        return summary

    df_reports = pd.read_csv(reports_path, low_memory=False)
    acc_col = cfg.reports_accession_column
    if acc_col not in df_reports.columns:
        summary = {"error": f"column '{acc_col}' not in reports CSV "
                            f"(available: {list(df_reports.columns)})"}
        print(f"[05] {summary}")
        return summary

    df_reports["_acc"] = _normalize(df_reports[acc_col])
    report_accs = set(df_reports.loc[df_reports["_acc"] != "", "_acc"])

    df_meta = pd.read_csv(consolidated_csv, low_memory=False)
    # AccessionNumber may hold comma-joined lists per study — explode them.
    meta_accs = set()
    for cell in _normalize(df_meta["AccessionNumber"]):
        meta_accs.update(a.strip() for a in cell.split(",") if a.strip())

    in_both = report_accs & meta_accs
    only_reports = report_accs - meta_accs
    only_meta = meta_accs - report_accs

    step_dir.mkdir(parents=True, exist_ok=True)
    missing_rows = (
        df_reports[df_reports["_acc"].isin(only_reports)]
        .drop(columns=["_acc"])
    )
    missing_rows.to_csv(step_dir / "accessions_in_reports_not_in_metadata.csv",
                        index=False)
    write_csv(step_dir / "missing_accession_list_only.csv",
              ["AccessionNumber_missing"],
              [{"AccessionNumber_missing": a} for a in sorted(only_reports)])

    coverage = (len(in_both) / len(report_accs) * 100) if report_accs else 0.0
    summary = {
        "report_accessions": len(report_accs),
        "metadata_accessions": len(meta_accs),
        "in_both": len(in_both),
        "missing_from_metadata": len(only_reports),
        "extra_in_metadata": len(only_meta),
        "coverage_pct": round(coverage, 1),
    }
    print(f"[05] {summary}")
    return summary
