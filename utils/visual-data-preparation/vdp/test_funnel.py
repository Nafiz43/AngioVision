"""Self-check for the VDP funnel composition. Run: python3 vdp/test_funnel.py

No pytest / heavy deps — compose_rows and _render_text are pure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vdp.funnel import _render_text, compose_rows  # noqa: E402

STEPS = {
    "00": {"status": "ok", "summary": {
        "dicom_files": 1000, "total_frames": 50000, "unreadable": 2,
        "missing_accession": 5, "duplicate_sop_uids": 30,
        "accession_study_mismatches": 4}},
    "01": {"status": "ok", "summary": {
        "mode": "relaxed", "examined": 1000,
        "filtered_by_reason": {"bad_radiation": 400, "bad_motion": 100,
                               "too_few_frames": 50},
        "filtered_frames_by_reason": {"bad_radiation": 20000, "bad_motion": 5000,
                                      "too_few_frames": 100},
        "filtered": 550, "skipped_existing": 50, "errors": 0,
        "processed": 400, "extracted_frames": 24900}},
    "06": {"status": "ok", "summary": {
        "sequences": 450, "frames": 22500,
        "potential_dsas": 300, "potential_dsa_frames": 15000,
        "potential_non_dsas": 150, "potential_non_dsa_frames": 7500,
        "verdict_breakdown": {"potential_dsa": 300, "no_mask_detected": 140,
                              "skipped_all_black": 10},
        "dsa_dir": "/x/00_potential_dsas", "non_dsa_dir": "/x/01_potential_non_dsas"}},
}


def main() -> None:
    rows = compose_rows(STEPS)
    by = {(r["stage"], r["metric"]): r for r in rows}

    # raw input carried through (sequences + frames both present)
    seq_row = by[("raw_input", "sequences (1 DICOM file = 1 sequence)")]
    assert seq_row["sequences"] == 1000 and seq_row["frames"] == 50000
    assert by[("raw_input", "duplicate SOP UIDs")]["sequences"] == 30

    # every filter reason present with its frame count
    r = next(v for k, v in by.items() if "bad_radiation" in k[1])
    assert r["sequences"] == 400 and r["frames"] == 20000

    # funnel arithmetic: examined = filtered_total + skipped + errors + processed
    s01 = STEPS["01"]["summary"]
    assert s01["examined"] == s01["filtered"] + s01["skipped_existing"] + \
        s01["errors"] + s01["processed"]
    # filtered_by_reason sums to filtered total
    assert sum(s01["filtered_by_reason"].values()) == s01["filtered"]

    # passed-total line = processed + skipped (clear on skip-existing re-runs)
    passed = by[("sequence_filter", "=> PASSED the filter (extracted + already on disk)")]
    assert passed["sequences"] == s01["processed"] + s01["skipped_existing"] == 450

    # image filter: sequences AND frames shown; non-DSA reasons listed
    assert by[("image_filter", "potential DSA")]["sequences"] == 300
    assert by[("image_filter", "potential DSA")]["frames"] == 15000
    non_dsa_metrics = [k[1] for k in by if k[0] == "image_filter" and "reason" in k[1]]
    assert any("no_mask_detected" in m for m in non_dsa_metrics)
    assert not any("potential_dsa" in m for m in non_dsa_metrics)

    # missing step degrades to blanks, never raises
    partial = compose_rows({"01": STEPS["01"]})
    assert any(r["stage"] == "raw_input" for r in partial)

    # renders without error and mentions a landing path
    text = _render_text(rows, {"extracted sequences": "/x/out"})
    assert "FILTERING FUNNEL" in text and "20,000 frames" in text

    print("test_funnel: OK  (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
