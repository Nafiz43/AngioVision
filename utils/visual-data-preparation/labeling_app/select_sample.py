#!/usr/bin/env python3
"""
Build a stratified random sample for manual DSA / non-DSA validation.

Population = the latest step-06 classification (dsa_split_report.csv), split
into the two verdict buckets, restricted to sequences that actually have a
mosaic.png (a few thousand DSA-bucket sequences never got one — step 03 only
mosaics the potential-DSA subset, and a handful of those still failed/were
skipped). Sample size per bucket uses Cochran's formula for estimating a
proportion from a finite population (p=0.5, the conservative worst case),
with the finite-population correction applied.

Usage:
    python3 select_sample.py

Writes labeling_app/sample.csv (id, bucket, algo_verdict, mosaic_path,
accession, sop_uid) — the fixed manifest the labeling app serves against.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

DSA_SPLIT_REPORT = (
    "/data/Deep_Angiography/AngioVision/utils/visual-data-preparation/"
    "runs/training/run_20260721_060709/06_dsa_split/dsa_split_report.csv"
)
OUT_CSV = Path(__file__).resolve().parent / "sample.csv"

CONFIDENCE_Z = 1.96   # 95% confidence
MARGIN_E = 0.05        # +/- 5%
SEED = 42               # matches the project's existing --random_seed convention


def cochran_n(population: int, z: float = CONFIDENCE_Z, e: float = MARGIN_E) -> int:
    n0 = (z ** 2) * 0.25 / (e ** 2)
    n = n0 / (1 + (n0 - 1) / population)
    return min(population, round(n))


def load_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {"dsa": [], "non_dsa": []}
    seen_split_dirs: set[str] = set()  # dsa_split_report.csv has ~2.7k duplicate
    # rows per sequence (same source data quirk noted in step 00's "duplicate SOP
    # UIDs" count) - dedup by split_dir so each physical sequence is one candidate.
    with open(DSA_SPLIT_REPORT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_dir = Path(row["split_dir"])
            if str(split_dir) in seen_split_dirs:
                continue
            seen_split_dirs.add(str(split_dir))
            mosaic_path = split_dir / "mosaic.png"
            if not mosaic_path.exists():
                continue  # can't show it in the app - excluded from the population
            bucket = "dsa" if row["verdict"] == "potential_dsa" else "non_dsa"
            accession, sop_uid = split_dir.parts[-2], split_dir.parts[-1]
            pools[bucket].append({
                "bucket": bucket,
                "algo_verdict": bucket,
                "mosaic_path": str(mosaic_path),
                "accession": accession,
                "sop_uid": sop_uid,
            })
    return pools


def main() -> None:
    pools = load_pools()
    rng = random.Random(SEED)
    rows: list[dict] = []
    for bucket, items in pools.items():
        n = cochran_n(len(items))
        sample = rng.sample(items, n)
        print(f"{bucket}: population(with mosaic)={len(items)}  sample_n={n}")
        for item in sample:
            item["id"] = f'{item["accession"]}__{item["sop_uid"]}'
            rows.append(item)

    rng.shuffle(rows)  # interleave buckets so labelers don't see 379 DSA in a row
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "bucket", "algo_verdict", "mosaic_path", "accession", "sop_uid",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
