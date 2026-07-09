"""
Validation Sample Expansion
===========================

Addresses a review comment that the original 200-record manual validation
sample (validation/sample_200.json) is too small relative to the full
stage-1 screening population (results/stage1_results.csv, N=4765 records
after LLM screening; 5064 before dedupe/screening).

Scope: INCLUDE vs. EXCLUDE only
--------------------------------
UNCERTAIN (195 records) and ERROR (1 record) pipeline decisions are
excluded from this audit's population entirely, not folded into the
EXCLUDE stratum. The primary pipeline already auto-flags every UNCERTAIN
record for human review (see Section "LLM-Assisted Screening Pipeline");
auditing them again here as if they were a confident machine decision
would conflate "the pipeline wasn't sure" with "the pipeline was
confidently wrong," which is a different failure mode and a different
question. The audited population is therefore the 4569 records with a
confident pipeline decision (276 INCLUDE + 4293 EXCLUDE).

Sample-size reasoning
----------------------
The quantity actually being validated is not "the pipeline's overall
accuracy" but two separate error rates that live on two very different
strata of the population:

  1. Precision (PPV) of the pipeline's INCLUDE calls
     -- stratum size N1 = records with llm_decision == INCLUDE (small, rare class)
  2. Miss-rate (false-negative rate) of the pipeline's EXCLUDE calls
     -- stratum size N2 = records with llm_decision == EXCLUDE (the bulk of the corpus)

A single simple-random sample of 200 drawn from N=4765 at the true INCLUDE
prevalence (~5.8%) would contain only ~12 INCLUDE-predicted records -- far
too few to bound precision. The original 200-sample partially corrected for
this by stratifying (70 INCLUDE / 70 EXCLUDE / 60 UNCERTAIN), but the
resulting per-stratum n's were picked by hand, not sized to a target
confidence interval, and mixed UNCERTAIN in with EXCLUDE.

This script sizes each stratum properly using Cochran's sample-size formula
for estimating a proportion, with the finite-population correction (FPC),
seeded by the error rates observed in the existing pilot-200 adjudication
(validation/screening_validation_raw.json), restricted to the pilot's
INCLUDE and EXCLUDE-only subsamples (the pilot's 60 UNCERTAIN records are
out of scope for this audit and are not counted toward any target below):

    n0 = Z^2 * p*(1-p) / e^2
    n  = n0 / (1 + (n0 - 1) / N)

  Z = 1.96 (95% CI), p = pilot-observed proportion, e = target absolute
  margin of error, N = stratum population size.

The pilot's EXCLUDE-only subsample showed zero false negatives (0/70).
Plugging p=0 directly into Cochran's formula would (nonsensically) imply
zero additional sampling is needed. Instead we use the standard "rule of
three" upper bound for a zero-event observation (Hanley & Lippman-Hand,
1983): with 0 events in n trials, the upper ~95% confidence bound on the
true rate is approximately 3/n. This is the conservative planning value
used for the EXCLUDE-stratum miss-rate below.

References:
  - Cochran, W.G. (1977). Sampling Techniques (3rd ed.). Wiley.
  - Buderer, N.M.F. (1996). "Statistical methodology: I. Incorporating the
    prevalence of disease into the sample size calculation for sensitivity
    and specificity." Acad Emerg Med, 3(9), 895-900.
    (Motivates stratifying by predicted/actual class when the positive
    class is rare, rather than one pooled SRS estimate.)
  - Hanley, J.A., Lippman-Hand, A. (1983). "If nothing goes wrong, is
    everything all right? Interpreting zero numerators." JAMA, 249(13),
    1743-1745. (The "rule of three" for a zero-event upper confidence bound.)

Strata and targets (see validation/sample_size_justification.md for the
full numeric derivation, written by this script):
  - INCLUDE stratum: required n (e=0.05) already exceeds half N1, so we
    recommend a full census of N1.
  - EXCLUDE stratum: required n sized at e=0.03 using the rule-of-three
    upper bound on the pilot's observed 0/70 miss-rate.

The pilot 200 records are NOT re-sampled -- their adjudicated labels
already count toward the totals above (except the 60 pilot UNCERTAIN
records, which are out of scope here). This script draws only the
additional records needed to reach each stratum's target, so the two
annotator CSVs it emits contain new work only. Once filled in, the
combined pilot + expansion pool feeds 10_merge_validation_results.py to
produce the final kappa / precision / recall / F1 for the paper.

Usage:
    python3 09_expand_validation_sample.py
"""

import csv
import json
import random
from pathlib import Path

STAGE1_PATH = Path("results/stage1_results.csv")
PILOT_SAMPLE_PATH = Path("validation/sample_200.json")
PILOT_RAW_PATH = Path("validation/screening_validation_raw.json")

JUSTIFICATION_PATH = Path("validation/sample_size_justification.md")
EXPANSION_MASTER_PATH = Path("validation/expansion_master.json")
FULL_MASTER_PATH = Path("validation/full_validation_master.json")
ANNOTATOR_A_PATH = Path("validation/annotator_A.csv")
ANNOTATOR_B_PATH = Path("validation/annotator_B.csv")

Z_95 = 1.96
MARGIN_INCLUDE_STRATUM = 0.05      # target absolute CI half-width on precision
MARGIN_EXCLUDE_STRATUM = 0.03      # tighter, since the miss-rate is a rare event

RANDOM_SEED = 42


def cochran_fpc(p, e, n_pop, z=Z_95):
    n0 = (z ** 2) * p * (1 - p) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / n_pop)
    return n0, n


def load_stage1():
    with open(STAGE1_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pilot():
    pilot = json.loads(PILOT_SAMPLE_PATH.read_text())
    raw = json.loads(PILOT_RAW_PATH.read_text())
    return pilot, raw


def pilot_error_rates(pilot, raw):
    """Precision from the pilot's INCLUDE subsample; miss-rate from the
    pilot's EXCLUDE-only subsample (UNCERTAIN pilot records excluded)."""
    sample_by_id = {s["sample_id"]: s for s in pilot}
    final_by_id = {d["sample_id"]: d["final_decision"] for d in raw["finalDecisions"]}

    tp = fp = fn = tn = 0
    for sid, final in final_by_id.items():
        s = sample_by_id[sid]
        decision = s["llm_decision"]
        if decision not in ("INCLUDE", "EXCLUDE"):
            continue  # UNCERTAIN pilot records are out of scope for this audit
        actual_include = final == "INCLUDE"
        if decision == "INCLUDE" and actual_include:
            tp += 1
        elif decision == "INCLUDE" and not actual_include:
            fp += 1
        elif decision == "EXCLUDE" and actual_include:
            fn += 1
        else:
            tn += 1

    precision_hat = tp / (tp + fp) if (tp + fp) else 0.5

    n_exclude_pilot = fn + tn
    if fn > 0:
        missrate_hat = fn / n_exclude_pilot
        missrate_note = f"{fn}/{n_exclude_pilot} = {missrate_hat:.4f}"
    else:
        # Rule of three: upper ~95% bound on a zero-event rate is ~3/n.
        missrate_hat = 3 / n_exclude_pilot
        missrate_note = f"0/{n_exclude_pilot} observed -> rule-of-three upper bound 3/{n_exclude_pilot} = {missrate_hat:.4f}"

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_hat": precision_hat,
        "missrate_hat": missrate_hat,
        "missrate_note": missrate_note,
        "n_include_stratum_pilot": tp + fp,
        "n_exclude_stratum_pilot": n_exclude_pilot,
    }


def main():
    population = load_stage1()
    pilot, raw = load_pilot()
    rates = pilot_error_rates(pilot, raw)

    include_pop = [r for r in population if r["llm_decision"] == "INCLUDE"]
    exclude_pop = [r for r in population if r["llm_decision"] == "EXCLUDE"]
    out_of_scope_pop = [r for r in population if r["llm_decision"] not in ("INCLUDE", "EXCLUDE")]

    n1_pop = len(include_pop)
    n2_pop = len(exclude_pop)
    n_scope = n1_pop + n2_pop

    n0_incl, n_incl_req = cochran_fpc(rates["precision_hat"], MARGIN_INCLUDE_STRATUM, n1_pop)
    n0_exc, n_exc_req = cochran_fpc(rates["missrate_hat"], MARGIN_EXCLUDE_STRATUM, n2_pop)

    # If the required sample already exceeds half the stratum, take a full census instead.
    include_target = n1_pop if n_incl_req > n1_pop / 2 else int(round(n_incl_req))
    exclude_target = int(round(n_exc_req)) if n_exc_req <= n2_pop / 2 else n2_pop

    pilot_include_ids = {s["record_id"] for s in pilot if s["llm_decision"] == "INCLUDE"}
    pilot_exclude_ids = {s["record_id"] for s in pilot if s["llm_decision"] == "EXCLUDE"}

    include_needed = max(0, include_target - len(pilot_include_ids))
    exclude_needed = max(0, exclude_target - len(pilot_exclude_ids))

    include_pool = [r for r in include_pop if r["record_id"] not in pilot_include_ids]
    exclude_pool = [r for r in exclude_pop if r["record_id"] not in pilot_exclude_ids]

    rng = random.Random(RANDOM_SEED)

    # Include stratum target == full census -> take the entire remaining pool.
    include_draw = include_pool if include_target >= n1_pop else rng.sample(
        sorted(include_pool, key=lambda r: r["record_id"]), include_needed
    )
    exclude_draw = rng.sample(
        sorted(exclude_pool, key=lambda r: r["record_id"]), min(exclude_needed, len(exclude_pool))
    )

    expansion_records = []
    sid = 1000  # keep expansion ids clear of the pilot's 1-200 sample_id space
    for r, stratum in [(rec, "INCLUDE") for rec in include_draw] + [(rec, "EXCLUDE") for rec in exclude_draw]:
        expansion_records.append({
            "sample_id": sid,
            "record_id": r["record_id"],
            "title": r["title"],
            "abstract": r["abstract"],
            "year": r["year"],
            "llm_decision": r["llm_decision"],
            "stratum": stratum,
        })
        sid += 1

    EXPANSION_MASTER_PATH.write_text(json.dumps(expansion_records, indent=2))

    # Build the FULL validation set (pilot INCLUDE/EXCLUDE records + new
    # expansion). The pilot's 60 UNCERTAIN records are dropped -- out of
    # scope for this audit -- and the pilot's earlier LLM-vs-LLM secondary
    # check is superseded: every in-scope record, pilot and newly drawn
    # alike, gets a fresh, independent decision from two HUMAN annotators,
    # so the reported kappa/precision/recall reflect one consistent
    # human-annotation design.
    pilot_records = []
    for s in pilot:
        if s["llm_decision"] not in ("INCLUDE", "EXCLUDE"):
            continue
        pilot_records.append({
            "sample_id": s["sample_id"],
            "record_id": s["record_id"],
            "title": s["title"],
            "abstract": s["abstract"],
            "year": s["year"],
            "llm_decision": s["llm_decision"],
            "stratum": s["llm_decision"],
        })

    full_records = pilot_records + expansion_records
    rng.shuffle(full_records)  # blind stratum/batch order for annotators

    FULL_MASTER_PATH.write_text(json.dumps(full_records, indent=2))

    fieldnames = ["sample_id", "record_id", "title", "abstract", "year", "decision", "notes"]
    for path in (ANNOTATOR_A_PATH, ANNOTATOR_B_PATH):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in full_records:
                writer.writerow({
                    "sample_id": rec["sample_id"],
                    "record_id": rec["record_id"],
                    "title": rec["title"],
                    "abstract": rec["abstract"],
                    "year": rec["year"],
                    "decision": "",  # annotator fills: INCLUDE / EXCLUDE
                    "notes": "",
                })

    combined_include_n = len(pilot_include_ids) + len(include_draw)
    combined_exclude_n = len(pilot_exclude_ids) + len(exclude_draw)
    combined_total = combined_include_n + combined_exclude_n
    n_new = len(include_draw) + len(exclude_draw)
    n_pilot_in_scope = len(pilot_include_ids) + len(pilot_exclude_ids)

    justification = f"""# Validation Sample Size Justification

## Population and scope
- Stage-1 screened corpus: **N = {len(population)}** records
  (`results/stage1_results.csv`; {n1_pop} pipeline-INCLUDE / {n2_pop} pipeline-EXCLUDE
  / {len(out_of_scope_pop)} pipeline-UNCERTAIN or ERROR).
- Note: this is the actual screened population size. It is **not** ~6000 --
  6000 is closer to the pre-dedupe/pre-screening search yield
  (`results/slr_stage1_screening.csv`, 5064 rows); the number relevant to
  validating the *screening decision* is the {len(population)} records that
  received an `llm_decision`.
- **UNCERTAIN and ERROR pipeline decisions ({len(out_of_scope_pop)} records) are
  excluded from this audit's population entirely.** The primary pipeline
  already auto-flags every UNCERTAIN record for human review, so treating it
  as a confident decision to audit here would conflate "the pipeline wasn't
  sure" with "the pipeline was confidently wrong" -- a different failure
  mode. The audited population is the **{n_scope}** records where the
  pipeline made a confident INCLUDE/EXCLUDE call.

## Why a flat 200-record SRS is the wrong benchmark
A simple random sample of 200 out of {n_scope} in-scope records, drawn at the
pipeline's ~{n1_pop/n_scope*100:.1f}% INCLUDE rate, would contain only
~{200*n1_pop/n_scope:.0f} INCLUDE-predicted records -- not enough to
bound precision on the class that matters most for a systematic review
(missing a true include, or admitting a false one, is the costly error).
The pilot sample already corrected for this informally by stratifying
70/70/60 (INCLUDE/EXCLUDE/UNCERTAIN); this script formalizes that intuition
with a target confidence interval per in-scope stratum instead of round
numbers, and drops the UNCERTAIN third of the pilot from the audited scope.

## Method
Cochran's sample-size-for-a-proportion formula with finite-population
correction (FPC), applied **per stratum**, stratified on the pipeline's own
predicted decision (INCLUDE vs. EXCLUDE) -- the standard approach for
auditing a classifier when the population's true labels are what you're
trying to estimate (Cochran, 1977; prevalence-aware sizing per Buderer, 1996):

    n0 = Z^2 * p(1-p) / e^2
    n  = n0 / (1 + (n0-1)/N)

Z = {Z_95} (95% CI). `p` is seeded from the pilot's own adjudicated error
rates, restricted to its INCLUDE/EXCLUDE subsample (the pilot's 60 UNCERTAIN
records are out of scope and excluded from these rates):

- Pilot INCLUDE-stratum precision (TP / (TP+FP)): {rates['tp']}/{rates['tp']+rates['fp']} = {rates['precision_hat']:.4f}
- Pilot EXCLUDE-stratum miss-rate (FN / (FN+TN)): {rates['missrate_note']}
  (the pilot observed zero false negatives among its 70 EXCLUDE-only records;
  per the rule of three (Hanley & Lippman-Hand, 1983) we use the upper ~95%
  bound on a zero-event rate, 3/n, as the conservative planning value rather
  than treating the true miss-rate as exactly zero)

## Stratum 1 -- pipeline-INCLUDE (precision), margin e = {MARGIN_INCLUDE_STRATUM}
- N1 = {n1_pop}
- n0 = {n0_incl:.1f}, FPC-adjusted n = {n_incl_req:.1f}
- Required n already exceeds half of N1 -> **recommend a full census: {include_target} of {n1_pop}.**

## Stratum 2 -- pipeline-EXCLUDE (miss-rate / recall), margin e = {MARGIN_EXCLUDE_STRATUM}
- N2 = {n2_pop}
- n0 = {n0_exc:.1f}, FPC-adjusted n = {n_exc_req:.1f}
- Target n = **{exclude_target}**

## Combined target
- Target: {include_target} (Stratum 1) + {exclude_target} (Stratum 2) = **{include_target + exclude_target}** records
- Already collected in-scope from pilot-200: {len(pilot_include_ids)} (Stratum 1) + {len(pilot_exclude_ids)} (Stratum 2) = {n_pilot_in_scope}
  (the pilot's 60 UNCERTAIN records are excluded from scope and not counted here)
- New records drawn by this script: {len(include_draw)} (Stratum 1) + {len(exclude_draw)} (Stratum 2) = **{n_new}**
- **Combined validation pool once the new records are annotated: {combined_total}**
  ({combined_include_n} of {n1_pop} INCLUDE stratum, {combined_exclude_n} of {n2_pop} EXCLUDE stratum)

This is a ~{combined_total/n_pilot_in_scope:.1f}x increase over the {n_pilot_in_scope}
in-scope pilot records, sized to hit an explicit 95%-CI margin per stratum
rather than picked as a round number.

The pilot's earlier LLM-vs-LLM secondary check is superseded here: rather
than mixing that automated pass with new manual review, all {combined_total}
in-scope records (the {n_pilot_in_scope} in-scope pilot records plus the
{n_new} newly drawn) are sent fresh to two independent HUMAN annotators, so
the reported inter-rater kappa and pipeline precision/recall reflect one
consistent human-annotation design.

## Outputs
- `validation/expansion_master.json` -- the {n_new} newly drawn
  candidate records (internal bookkeeping copy only).
- `validation/full_validation_master.json` -- the full {combined_total}-record
  validation set (in-scope pilot + expansion), includes `llm_decision` and
  `stratum` for later joining -- not for annotator eyes.
- `validation/annotator_A.csv`, `validation/annotator_B.csv` -- identical,
  blinded (no `llm_decision` column) lists of all {combined_total} candidate
  records to send to the two human annotators. Each fills in `decision`
  (INCLUDE/EXCLUDE) and optional `notes` independently.
"""
    JUSTIFICATION_PATH.write_text(justification)

    print(justification)
    print(f"Wrote {combined_total} candidate records ({n_new} newly drawn "
          f"+ {n_pilot_in_scope} in-scope from the pilot) to:")
    print(f"  {EXPANSION_MASTER_PATH}")
    print(f"  {FULL_MASTER_PATH}")
    print(f"  {ANNOTATOR_A_PATH}")
    print(f"  {ANNOTATOR_B_PATH}")
    print(f"  {JUSTIFICATION_PATH}")


if __name__ == "__main__":
    main()
