"""
VDP funnel report — a story-like accounting of what happened to every DICOM
instance across the pipeline's filtering stages.

Reads the per-step summaries already collected in the run manifest (no re-scan)
and renders:

    RAW INPUT (step 00)            counts + total frames, duplicates, no-accession
    SEQUENCE FILTER (step 01)      per-reason breakdown of what each gate dropped
    IMAGE-BASED FILTER (step 06)   DSA vs non-DSA (+ why things were non-DSA)
    WHERE IT LANDED                output paths for every artifact

``compose_rows`` is pure (takes plain dicts) so the funnel math is unit-tested
without running the pipeline. ``build`` wires it to the run dir and writes both
a human-readable .txt and a machine-readable .csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Filter reasons in the order the short-circuit gate applies them (step 01),
# each paired with a human label.
FILTER_REASONS = [
    ("bad_radiation", "RadiationSetting != GR"),
    ("bad_series", "SeriesDescription lacks DSA/CO 2  [strict only]"),
    ("bad_motion", "PositionerMotion != STATIC"),
    ("too_few_frames", "NumberOfFrames <= min_frames"),
    ("filter_error", "filter raised an exception"),
]


def _summ(steps: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    """Summary dict for a step, or {} if the step didn't run / has none."""
    info = steps.get(step_id, {})
    s = info.get("summary")
    return s if isinstance(s, dict) else {}


def compose_rows(steps: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the funnel as a flat list of rows (stage, metric, sequences, frames).

    Pure: `steps` is manifest["steps"] — {step_id: {"status", "summary": {...}}}.
    Missing keys degrade to blanks rather than raising.
    """
    s00, s01, s06 = _summ(steps, "00"), _summ(steps, "01"), _summ(steps, "06")
    rows: List[Dict[str, Any]] = []

    def add(stage: str, metric: str, seqs: Any = "", frames: Any = "") -> None:
        rows.append({"stage": stage, "metric": metric,
                     "sequences": seqs, "frames": frames})

    # ── RAW INPUT (step 00) ──────────────────────────────────────────────
    add("raw_input", "sequences (1 DICOM file = 1 sequence)",
        s00.get("dicom_files", ""), s00.get("total_frames", ""))
    add("raw_input", "unreadable", s00.get("unreadable", ""))
    add("raw_input", "missing accession number", s00.get("missing_accession", ""))
    add("raw_input", "duplicate SOP UIDs", s00.get("duplicate_sop_uids", ""))
    add("raw_input", "accession -> multi-study mismatches",
        s00.get("accession_study_mismatches", ""))

    # ── SEQUENCE ELIGIBILITY FILTER (step 01) ────────────────────────────
    add("sequence_filter", f"instances examined (mode={s01.get('mode', '?')})",
        s01.get("examined", ""))
    by_reason = s01.get("filtered_by_reason", {}) or {}
    by_reason_frames = s01.get("filtered_frames_by_reason", {}) or {}
    for reason, label in FILTER_REASONS:
        if reason in by_reason:
            add("sequence_filter", f"filtered: {reason} ({label})",
                by_reason[reason], by_reason_frames.get(reason, ""))
    # any reasons we didn't enumerate above
    for reason, cnt in by_reason.items():
        if reason not in {r for r, _ in FILTER_REASONS}:
            add("sequence_filter", f"filtered: {reason}", cnt,
                by_reason_frames.get(reason, ""))
    add("sequence_filter", "filtered TOTAL", s01.get("filtered", ""))
    add("sequence_filter", "errors (extraction)", s01.get("errors", ""))
    # Passed the filter = newly extracted + already-on-disk (skip-existing).
    processed, skipped = s01.get("processed"), s01.get("skipped_existing")
    passed_total = (processed + skipped
                    if isinstance(processed, int) and isinstance(skipped, int) else "")
    add("sequence_filter", "=> PASSED the filter (extracted + already on disk)",
        passed_total)
    add("sequence_filter", "   - extracted this run",
        processed if processed is not None else "", s01.get("extracted_frames", ""))
    add("sequence_filter", "   - already extracted (skipped)",
        skipped if skipped is not None else "")

    # ── IMAGE-BASED DSA FILTER (step 06) ─────────────────────────────────
    add("image_filter", "sequences classified", s06.get("sequences", ""),
        s06.get("frames", ""))
    add("image_filter", "potential DSA", s06.get("potential_dsas", ""),
        s06.get("potential_dsa_frames", ""))
    add("image_filter", "potential non-DSA", s06.get("potential_non_dsas", ""),
        s06.get("potential_non_dsa_frames", ""))
    for verdict, cnt in (s06.get("verdict_breakdown", {}) or {}).items():
        if verdict != "potential_dsa":
            add("image_filter", f"  non-DSA reason: {verdict}", cnt)

    return rows


def _fmt(v: Any) -> str:
    return f"{v:,}" if isinstance(v, int) else ("" if v == "" else str(v))


def _render_text(rows: List[Dict[str, Any]], paths: Dict[str, str],
                 header_lines: List[str] = None,
                 csv_files: List[str] = None) -> str:
    bar = "=" * 72
    stage_titles = {
        "raw_input": "RAW INPUT  (step 00 — header scan)",
        "sequence_filter": "SEQUENCE ELIGIBILITY FILTER  (step 01)",
        "image_filter": "IMAGE-BASED DSA FILTER  (step 06 — frame mask detection)",
    }
    out = [bar, "  VDP RUN — FILTERING FUNNEL / STORY", bar]
    for hl in (header_lines or []):
        out.append(f"  {hl}")
    out.append("  (counts are SEQUENCES; frame totals shown in parentheses)")
    seen_stage = None
    for r in rows:
        if r["stage"] != seen_stage:
            seen_stage = r["stage"]
            out.append("")
            out.append(stage_titles.get(seen_stage, seen_stage))
        seqs, frames = _fmt(r["sequences"]), _fmt(r["frames"])
        dots = "." * max(3, 48 - len(r["metric"]))
        line = f"  {r['metric']} {dots} {seqs:>12}"
        if frames:
            line += f"   ({frames:>12} frames)"
        out.append(line)

    out.append("")
    out.append("WHERE IT ALL LANDED")
    for label, p in paths.items():
        out.append(f"  {label:<26} {p}")
    if csv_files:
        out.append("")
        out.append("CSV FILES PRODUCED")
        for p in csv_files:
            out.append(f"  {p}")
    out.append(bar)
    return "\n".join(out) + "\n"


def build(steps: Dict[str, Any], cfg, run_dir: Path) -> Dict[str, Any]:
    """Compose the funnel, write .txt + .csv under run_dir, print it, return summary."""
    from vdp.common import write_csv  # lazy: keeps compose_rows import-light

    rows = compose_rows(steps)
    s06 = _summ(steps, "06")
    s04 = _summ(steps, "04")
    paths = {
        "extracted sequences": str(cfg.output_root),
        "potential DSAs": s06.get("dsa_dir", str(Path(cfg.dsa_split_root) / "00_potential_dsas")),
        "potential non-DSAs": s06.get("non_dsa_dir", str(Path(cfg.dsa_split_root) / "01_potential_non_dsas")),
        "consolidated metadata": s04.get("consolidated_csv", "(step 04 did not run)"),
        "per-stage reports": str(run_dir),
    }
    header = [
        f"data type   : {getattr(cfg, 'data_type', 'training').upper()}",
        f"input root  : {cfg.input_root}",
    ]

    report_txt = run_dir / "vdp_funnel_report.txt"
    write_csv(run_dir / "vdp_funnel_report.csv",
              ["stage", "metric", "sequences", "frames"], rows)
    paths["this report"] = str(report_txt)
    csv_files = _collect_csvs(steps, run_dir)
    text = _render_text(rows, paths, header, csv_files)
    report_txt.write_text(text, encoding="utf-8")
    print(text)
    return {"rows": len(rows), "report_txt": str(report_txt),
            "csv_files": csv_files}


def _collect_csvs(steps: Dict[str, Any], run_dir: Path) -> List[str]:
    """Every CSV the run produced: all under run_dir, plus any absolute CSV
    path a step summary mentions that lives outside it (e.g. sizes CSVs)."""
    found = {str(p.resolve()) for p in run_dir.rglob("*.csv")}
    for info in steps.values():
        s = info.get("summary")
        if isinstance(s, dict):
            for v in s.values():
                if isinstance(v, str) and v.endswith(".csv") and Path(v).exists():
                    found.add(str(Path(v).resolve()))
    return sorted(found)
