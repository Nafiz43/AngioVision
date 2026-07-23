#!/usr/bin/env python3
"""
Build a stratified random sample for manual article-screening validation.

Population = the latest Stage-1 LLM screening pass (stage1_results.csv),
stratified by llm_decision (INCLUDE / EXCLUDE / UNCERTAIN; the single ERROR
row is dropped). Sample size per bucket uses Cochran's formula for
estimating a proportion from a finite population (p=0.5, the conservative
worst case), with the finite-population correction applied - same
methodology as utils/visual-data-preparation/labeling_app/select_sample.py.

Usage:
    python3 select_sample.py

Writes labeling_app/sample.csv (id, bucket, algo_verdict, title, authors,
year, journal_venue, doi, url, abstract, llm_inclusion_reason,
llm_exclusion_reason) - the fixed manifest the labeling app serves against.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

STAGE1_RESULTS = "/data/Deep_Angiography/d_AngioVision/slr/results/stage1_results.csv"
OUT_CSV = Path(__file__).resolve().parent / "sample.csv"

CONFIDENCE_Z = 1.96   # 95% confidence
MARGIN_E = 0.05        # +/- 5%
SEED = 42               # matches the project's existing --random_seed convention

FIELDS = [
    "id", "bucket", "algo_verdict", "title", "authors", "year",
    "journal_venue", "doi", "url", "abstract",
    "llm_inclusion_reason", "llm_exclusion_reason",
]


def cochran_n(population: int, z: float = CONFIDENCE_Z, e: float = MARGIN_E) -> int:
    n0 = (z ** 2) * 0.25 / (e ** 2)
    n = n0 / (1 + (n0 - 1) / population)
    return min(population, round(n))


def load_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {"INCLUDE": [], "EXCLUDE": [], "UNCERTAIN": []}
    with open(STAGE1_RESULTS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            decision = row["llm_decision"]
            if decision not in pools:
                continue  # drops the single ERROR row
            pools[decision].append(row)
    return pools


def main() -> None:
    pools = load_pools()
    rng = random.Random(SEED)
    rows: list[dict] = []
    for bucket, items in pools.items():
        n = cochran_n(len(items))
        sample = rng.sample(items, n)
        print(f"{bucket}: population={len(items)}  sample_n={n}")
        for item in sample:
            rows.append({
                "id": item["record_id"],
                "bucket": bucket,
                "algo_verdict": "include" if bucket == "INCLUDE" else "exclude",
                "title": item["title"],
                "authors": item["authors"],
                "year": item["year"],
                "journal_venue": item["journal_venue"],
                "doi": item["doi"],
                "url": item["url"],
                "abstract": item["abstract"],
                "llm_inclusion_reason": item["llm_inclusion_reason"],
                "llm_exclusion_reason": item["llm_exclusion_reason"],
            })

    rng.shuffle(rows)  # interleave buckets so labelers don't see runs of one verdict
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
