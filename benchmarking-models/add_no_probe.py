"""Insert a `no_probe` column into a probe leaderboard CSV.

no_probe = the fine-tuned checkpoint's OWN yes/no readout (no LR), scored
per-question with the SAME averaging as the probe columns, over the SAME rows.

Reads each checkpoint's readout predictions (AccessionNumber,SOPInstanceUID,
Question,Answer=prediction), dedups by (SOP,Question), joins ground truth from
the validation CSV, computes per-question F1, and writes the value into the
`no_probe` column (inserted right after `yes_rate`) for that checkpoint's rows.

Usage:
  python3 add_no_probe.py <gt_csv> <probe_csv> <out_csv> <macro|weighted> \
          <label> <readout_csv> [<label> <readout_csv> ...]
"""
import csv, sys
from collections import defaultdict
from sklearn.metrics import f1_score

def canon(q):
    s = str(q).strip().lower()
    suf = "please state yes or no."
    if s.endswith(suf):
        s = s[:-len(suf)].strip()
    return s

def binr(v):
    s = str(v).strip().lower()
    if s in ("1", "yes", "y", "true"): return 1
    if s in ("0", "no", "n", "false"): return 0
    return None

def load_gt(gt_csv):
    gt = {}
    for r in csv.DictReader(open(gt_csv)):
        sop = str(r.get("SOPInstanceUID", "")).strip()
        b = binr(r.get("Answer", ""))
        if sop and b is not None:
            gt[(sop, canon(r.get("Question", "")))] = b
    return gt

def perq_readout_f1(readout_csv, gt, average):
    # dedup by (SOP, canon question), join GT -> per-question y_true / y_pred
    seen = {}
    for r in csv.DictReader(open(readout_csv)):
        sop = str(r.get("SOPInstanceUID", "")).strip()
        cq = canon(r.get("Question", ""))
        pred = binr(r.get("Answer", ""))
        if sop and pred is not None and (sop, cq) not in seen:
            seen[(sop, cq)] = pred
    by_q = defaultdict(lambda: ([], []))
    matched = 0
    for (sop, cq), pred in seen.items():
        if (sop, cq) in gt:
            yt, yp = by_q[cq]; yt.append(gt[(sop, cq)]); yp.append(pred); matched += 1
    f1 = {cq: round(float(f1_score(yt, yp, average=average, zero_division=0)), 3)
          for cq, (yt, yp) in by_q.items()}
    n = {cq: len(yt) for cq, (yt, yp) in by_q.items()}
    # overall (sanity vs known epoch readout)
    allt = [v for cq in by_q for v in by_q[cq][0]]
    allp = [v for cq in by_q for v in by_q[cq][1]]
    overall = round(float(f1_score(allt, allp, average="macro", zero_division=0)), 4)
    acc = round(sum(int(a == b) for a, b in zip(allt, allp)) / len(allt), 4)
    return f1, n, matched, overall, acc

def main():
    gt_csv, probe_csv, out_csv, average = sys.argv[1:5]
    pairs = list(zip(sys.argv[5::2], sys.argv[6::2]))
    gt = load_gt(gt_csv)
    print(f"[GT] {len(gt)} (SOP,question) ground-truth entries")

    # label -> {canon_q: f1}
    perq = {}
    for label, rc in pairs:
        f1, n, matched, overall, acc = perq_readout_f1(rc, gt, average)
        perq[label] = f1
        print(f"[{label:14s}] matched {matched} rows, {len(f1)} questions | "
              f"overall readout: acc={acc} macroF1={overall}")

    rows = list(csv.DictReader(open(probe_csv)))
    cols = list(rows[0].keys())
    if "no_probe" not in cols:
        i = cols.index("yes_rate") + 1
        cols.insert(i, "no_probe")
    filled = mism = 0
    for r in rows:
        f1map = perq.get(r["checkpoint"], {})
        v = f1map.get(canon(r["question"]))
        r["no_probe"] = "" if v is None else v
        if v is not None:
            filled += 1
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[{average}] wrote {out_csv}: filled no_probe for {filled}/{len(rows)} rows")

if __name__ == "__main__":
    main()
