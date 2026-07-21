#!/usr/bin/env python3
"""
DSA / non-DSA manual labeling app.

Serves the fixed stratified sample (sample.csv, built by select_sample.py)
one mosaic at a time, pre-selected to the algorithm's own verdict. A
submission appends to labels.csv and is never shown again - state lives
entirely in that file, so it's consistent across sessions, restarts, and
multiple labelers hitting the same server.

Usage:
    python3 app.py [--port 5051]
"""
from __future__ import annotations

import argparse
import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = APP_DIR / "sample.csv"
LABELS_CSV = APP_DIR / "labels.csv"
LABELERS = ["Goldman", "Nafiz"]
LABEL_FIELDS = ["id", "bucket", "algo_verdict", "human_label", "labeler", "labeled_at"]

app = Flask(__name__)
_lock = threading.Lock()  # guards labels.csv read-modify-write (low-concurrency: 2 labelers)


def load_sample() -> list[dict]:
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labeled_ids() -> set[str]:
    if not LABELS_CSV.exists():
        return set()
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def append_label(row: dict) -> None:
    is_new = not LABELS_CSV.exists()
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def sample_by_id() -> dict[str, dict]:
    return {row["id"]: row for row in load_sample()}


@app.route("/")
def index():
    sample = load_sample()
    with _lock:
        labeled_ids = load_labeled_ids()
    remaining = [r for r in sample if r["id"] not in labeled_ids]

    progress = {
        "done": len(labeled_ids & {r["id"] for r in sample}),
        "total": len(sample),
    }

    if not remaining:
        return render_template("done.html", progress=progress)

    current = remaining[0]
    return render_template(
        "index.html",
        item=current,
        labelers=LABELERS,
        progress=progress,
    )


@app.route("/mosaic/<item_id>")
def mosaic(item_id: str):
    row = sample_by_id().get(item_id)
    if not row:
        abort(404)
    path = Path(row["mosaic_path"])
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/label", methods=["POST"])
def label():
    item_id = request.form["id"]
    human_label = request.form["human_label"]
    labeler = request.form.get("labeler", LABELERS[0])
    if human_label not in ("dsa", "non_dsa") or labeler not in LABELERS:
        abort(400)

    row = sample_by_id().get(item_id)
    if not row:
        abort(404)

    with _lock:
        # Re-check under the lock: two near-simultaneous submits for the same
        # id (shouldn't happen with 2 labelers, but cheap to guard) only
        # record once.
        if item_id not in load_labeled_ids():
            append_label({
                "id": item_id,
                "bucket": row["bucket"],
                "algo_verdict": row["algo_verdict"],
                "human_label": human_label,
                "labeler": labeler,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            })

    return redirect(url_for("index"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    if not SAMPLE_CSV.exists():
        raise SystemExit(f"{SAMPLE_CSV} not found - run select_sample.py first.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
