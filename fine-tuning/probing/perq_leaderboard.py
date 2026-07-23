"""Per-question leaderboard across checkpoints x feature-sets.
One row per (checkpoint, question): group, n, yes_rate, then macro-F1 for
BASELINE / A1 / A2 / A3 / A2+A3 (+ B3margins where the checkpoint's npz has
captured yes/no hypothesis embeddings). Probe = ONE LogisticRegression per
fold trained on ALL questions (OOF GroupKFold by accession); per-question F1
= F1 on that question's OOF slice.

Usage: python3 perq_leaderboard.py [macro|weighted]  (default macro)

Writes /tmp/perq_leaderboard_<average>.csv plus a confusion-count sidecar
/tmp/perq_leaderboard_<average>_conf.csv ("TP=.. TN=.. FP=.. FN=.. (n=..)"
per feature column) -- the sidecar is averaging-independent (same raw
predictions), matching bmk.s06_per_question's expectation of a
"<probe_perq_csv>_conf.csv" file alongside the leaderboard CSV.
"""
import csv, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

CKPTS = [
    ("SigLIP2 ep2",   "/tmp/rich_siglip2.npz"),
    ("X-CLIP ep7",    "/tmp/rich_xclip.npz"),
    ("MedSigLIP ep3", "/tmp/rich_medsiglip.npz"),
    ("CLIP ep2",      "/tmp/rich_clip_ep2.npz"),
    ("MedSigLIP ep7", "/tmp/rich_medsiglip_ep7.npz"),
]
BASE_FEATS = ["BASELINE", "A1_pre", "A2_attn", "A3_xmodal", "A2+A3"]

def group_of(q):
    s = q.lower()
    if "catheter tip" in s or "sheath tip" in s:                                   return "LOCATION"
    if any(k in s for k in ("embolic", "coil", "plug", "stent", "microcatheter")): return "DEVICE"
    if any(k in s for k in ("abnormal", "aberran", "tumor", "extravasation",
                            "bleeding", "atherosclerot", "shunt", "stenos", "aneurysm")): return "PATHOLOGY"
    if "approach" in s:                                                            return "ACCESS"
    if any(k in s for k in ("opacif", "perfused", "patent")):                      return "OPACIFICATION"
    return "OTHER"

def nrm(a):  return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
def cosf(a, b): return (nrm(a) * nrm(b)).sum(1, keepdims=True)

def build_feats(d):
    """BASE_FEATS always; B3margins too when the npz carries yes/no hypothesis
    embeddings (yes_emb/no_emb/family) -- the round-4 winner: [final; attn;
    question; cross-modal products/diffs/cosines; zero-shot yes/no similarity
    margin, raw + per-family debiased]."""
    f, p, at, qv = d["final"], d["pre"], d["attn"], d["qv"]
    feats = {
        "BASELINE":  np.concatenate([f, qv, f * qv], 1),
        "A1_pre":    p,
        "A2_attn":   np.concatenate([at, qv, at * qv], 1),
        "A3_xmodal": np.concatenate([f, qv, f * qv, np.abs(f - qv), cosf(f, qv)], 1),
        "A2+A3":     np.concatenate([at, qv, at * qv, np.abs(at - qv)], 1),
    }
    if "yes_emb" in d.files:
        yes_e, no_e, family = d["yes_emb"], d["no_emb"], d["family"].astype(str)
        margin_final = (f * yes_e).sum(1, keepdims=True) - (f * no_e).sum(1, keepdims=True)
        margin_attn = (at * yes_e).sum(1, keepdims=True) - (at * no_e).sum(1, keepdims=True)

        def debias(m):
            out = m.copy()
            for fam in set(family.tolist()):
                mask = family == fam
                out[mask] -= np.median(m[mask])
            return out

        margins = np.concatenate(
            [margin_final, margin_attn, debias(margin_final), debias(margin_attn)], 1)
        feats["B3margins"] = np.concatenate(
            [f, at, qv, f * qv, at * qv, np.abs(f - qv), np.abs(at - qv),
             cosf(f, qv), cosf(at, qv), margins], 1)
    return feats

def oof_pred(X, y, g, nsp=5):
    ns = min(nsp, len(set(g)))
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(ns).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return pred

def confusion(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    return tp, tn, fp, fn

def conf_tip(tp, tn, fp, fn):
    return f"TP={tp} TN={tn} FP={fp} FN={fn} (n={tp+tn+fp+fn})"

average = sys.argv[1] if len(sys.argv) > 1 else "macro"
assert average in ("macro", "weighted"), f"average must be macro|weighted, got {average!r}"

rows, conf_rows = [], []
for cname, path in CKPTS:
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"].astype(str); q = d["q"].astype(str)
    fs = build_feats(d)
    feat_names = list(fs.keys())  # BASE_FEATS + B3margins if present
    preds = {fn: oof_pred(X, y, g) for fn, X in fs.items()}
    for qu in sorted(set(q.tolist())):
        m = q == qu
        yq = y[m]; n = int(m.sum()); yr = float(yq.mean())
        row = {"checkpoint": cname, "group": group_of(qu), "question": qu.replace("Please state yes or no.", "").strip(),
               "n": n, "yes_rate": round(yr, 2)}
        conf_row = {"checkpoint": cname, "question": row["question"]}
        for fn in BASE_FEATS:
            pq = preds[fn][m]
            f1 = f1_score(yq, pq, average=average, zero_division=0)
            row[f"F1_{fn}"] = round(float(f1), 3)
            conf_row[fn] = conf_tip(*confusion(yq, pq))
        if "B3margins" in feat_names:
            pq = preds["B3margins"][m]
            f1 = f1_score(yq, pq, average=average, zero_division=0)
            row["F1_B3margins"] = round(float(f1), 3)
            conf_row["B3margins"] = conf_tip(*confusion(yq, pq))
        else:
            row["F1_B3margins"] = ""
        # best-A among A1..A2+A3 (+B3margins, if present) vs baseline
        avals = {fn: row[f"F1_{fn}"] for fn in BASE_FEATS if fn != "BASELINE"}
        if "B3margins" in feat_names:
            avals["B3margins"] = row["F1_B3margins"]
        bestA = max(avals, key=avals.get)
        row["best_A"] = bestA; row["best_A_F1"] = avals[bestA]
        row["delta_vs_base"] = round(avals[bestA] - row["F1_BASELINE"], 3)
        rows.append(row)
        conf_rows.append(conf_row)

cols = ["checkpoint", "group", "question", "n", "yes_rate",
        "F1_BASELINE", "F1_A1_pre", "F1_A2_attn", "F1_A3_xmodal", "F1_A2+A3", "F1_B3margins",
        "best_A", "best_A_F1", "delta_vs_base"]
out_csv = f"/tmp/perq_leaderboard_{average}.csv"
with open(out_csv, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)

conf_cols = ["checkpoint", "question"] + BASE_FEATS + ["B3margins"]
conf_csv = f"/tmp/perq_leaderboard_{average}_conf.csv"
with open(conf_csv, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=conf_cols, restval="")
    w.writeheader(); w.writerows(conf_rows)

print("CSV rows:", len(rows), "->", out_csv)
print("confusion sidecar ->", conf_csv)

# stdout: readable n>=8 subset, grouped by checkpoint
print("\n== n>=8 questions (interpretable) ==")
hdr = f"{'ckpt':13s} {'group':13s} {'n':>3s} {'yr':>4s} {'base':>5s} {'A1':>5s} {'A2':>5s} {'A3':>5s} {'A2A3':>5s} {'B3m':>5s}  question"
for cname, _ in CKPTS:
    print("\n-- " + cname + " --")
    print(hdr)
    for r in [r for r in rows if r["checkpoint"] == cname and r["n"] >= 8]:
        b3 = f"{r['F1_B3margins']:5.3f}" if r["F1_B3margins"] != "" else "    ."
        print(f"{r['checkpoint'][:13]:13s} {r['group'][:13]:13s} {r['n']:3d} {r['yes_rate']:4.2f} "
              f"{r['F1_BASELINE']:5.3f} {r['F1_A1_pre']:5.3f} {r['F1_A2_attn']:5.3f} "
              f"{r['F1_A3_xmodal']:5.3f} {r['F1_A2+A3']:5.3f} {b3}  {r['question'][:44]}")
# emit full CSV to stdout for offline capture, sentinel-delimited
print("\n===CSVSTART===")
print(open(out_csv).read())
print("===CSVEND===")
