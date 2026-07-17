"""Per-question leaderboard across checkpoints x feature-sets.
One row per (checkpoint, question): group, n, yes_rate, then macro-F1 for
BASELINE / A1 / A2 / A3 / A2+A3. Probe = ONE LogisticRegression per fold trained
on ALL questions (OOF GroupKFold by accession); per-question F1 = F1 on that
question's OOF slice. Also emits acc and confusion for the CSV. Writes CSV."""
import csv, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

CKPTS = [
    ("SigLIP2 ep2",   "/tmp/rich_siglip2.npz"),
    ("X-CLIP ep7",    "/tmp/rich_xclip.npz"),
    ("MedSigLIP ep3", "/tmp/rich_medsiglip.npz"),
]
FEATS = ["BASELINE", "A1_pre", "A2_attn", "A3_xmodal", "A2+A3"]

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
    f, p, at, qv = d["final"], d["pre"], d["attn"], d["qv"]
    return {
        "BASELINE":  np.concatenate([f, qv, f * qv], 1),
        "A1_pre":    p,
        "A2_attn":   np.concatenate([at, qv, at * qv], 1),
        "A3_xmodal": np.concatenate([f, qv, f * qv, np.abs(f - qv), cosf(f, qv)], 1),
        "A2+A3":     np.concatenate([at, qv, at * qv, np.abs(at - qv)], 1),
    }

def oof_pred(X, y, g, nsp=5):
    ns = min(nsp, len(set(g)))
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(ns).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return pred

rows = []
for cname, path in CKPTS:
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"].astype(str); q = d["q"].astype(str)
    fs = build_feats(d)
    preds = {fn: oof_pred(X, y, g) for fn, X in fs.items()}
    for qu in sorted(set(q.tolist())):
        m = q == qu
        yq = y[m]; n = int(m.sum()); yr = float(yq.mean())
        row = {"checkpoint": cname, "group": group_of(qu), "question": qu.replace("Please state yes or no.", "").strip(),
               "n": n, "yes_rate": round(yr, 2)}
        for fn in FEATS:
            pq = preds[fn][m]
            f1 = f1_score(yq, pq, average="macro", zero_division=0)
            row[f"F1_{fn}"] = round(float(f1), 3)
        # best-A among A1..A2+A3 vs baseline
        avals = {fn: row[f"F1_{fn}"] for fn in FEATS if fn != "BASELINE"}
        bestA = max(avals, key=avals.get)
        row["best_A"] = bestA; row["best_A_F1"] = avals[bestA]
        row["delta_vs_base"] = round(avals[bestA] - row["F1_BASELINE"], 3)
        rows.append(row)

cols = ["checkpoint", "group", "question", "n", "yes_rate",
        "F1_BASELINE", "F1_A1_pre", "F1_A2_attn", "F1_A3_xmodal", "F1_A2+A3",
        "best_A", "best_A_F1", "delta_vs_base"]
with open("/tmp/perq_leaderboard.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)

# stdout: readable n>=8 subset, grouped by checkpoint
print("CSV rows:", len(rows), "-> /tmp/perq_leaderboard.csv")
print("\n== n>=8 questions (interpretable) ==")
hdr = f"{'ckpt':13s} {'group':13s} {'n':>3s} {'yr':>4s} {'base':>5s} {'A1':>5s} {'A2':>5s} {'A3':>5s} {'A2A3':>5s}  question"
for cname, _ in CKPTS:
    print("\n-- " + cname + " --")
    print(hdr)
    for r in [r for r in rows if r["checkpoint"] == cname and r["n"] >= 8]:
        print(f"{r['checkpoint'][:13]:13s} {r['group'][:13]:13s} {r['n']:3d} {r['yes_rate']:4.2f} "
              f"{r['F1_BASELINE']:5.3f} {r['F1_A1_pre']:5.3f} {r['F1_A2_attn']:5.3f} "
              f"{r['F1_A3_xmodal']:5.3f} {r['F1_A2+A3']:5.3f}  {r['question'][:44]}")
# emit full CSV to stdout for offline capture, sentinel-delimited
print("\n===CSVSTART===")
print(open("/tmp/perq_leaderboard.csv").read())
print("===CSVEND===")
