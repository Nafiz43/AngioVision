"""Approach-A interpretability report per checkpoint.
For each checkpoint x feature-set, train ONE probe per fold on ALL questions
(GroupKFold by accession), collect out-of-fold predictions, then slice by
clinical question-group. Reports n, acc, F1(macro), TP, TN, FP, FN
(positive class = 'yes'). Overall + per-group. Baseline + all A-variants."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

FILES = [
    ("SigLIP2 ep2   (cos 0.557)",   "/tmp/rich_siglip2.npz"),
    ("X-CLIP ep7    (cos 0.513)",   "/tmp/rich_xclip.npz"),
    ("MedSigLIP ep3 (cos 0.483 inv)", "/tmp/rich_medsiglip.npz"),
]

def group_of(q):
    s = q.lower()
    if "catheter tip" in s or "sheath tip" in s:                                   return "LOCATION"
    if any(k in s for k in ("embolic", "coil", "plug", "stent", "microcatheter")): return "DEVICE"
    if any(k in s for k in ("abnormal", "aberran", "tumor", "extravasation",
                            "bleeding", "atherosclerot", "shunt", "stenos", "aneurysm")): return "PATHOLOGY"
    if "approach" in s:                                                            return "ACCESS"
    if any(k in s for k in ("opacif", "perfused", "patent")):                      return "OPACIFICATION"
    return "OTHER"

GROUP_ORDER = ["OPACIFICATION", "LOCATION", "PATHOLOGY", "DEVICE", "ACCESS", "OTHER"]

def nrm(a):  return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
def cosf(a, b): return (nrm(a) * nrm(b)).sum(1, keepdims=True)

def featsets(d):
    f, p, at, qv = d["final"], d["pre"], d["attn"], d["qv"]
    return {
        "BASELINE [final;qv;f*qv]":  np.concatenate([f, qv, f * qv], 1),
        "A1 pre-only":               p,
        "A2 attn [attn;qv;a*qv]":    np.concatenate([at, qv, at * qv], 1),
        "A3 rich-xmodal":            np.concatenate([f, qv, f * qv, np.abs(f - qv), cosf(f, qv)], 1),
        "A2+A3 attn+xmodal":         np.concatenate([at, qv, at * qv, np.abs(at - qv)], 1),
    }

def oof_pred(X, y, g, C=1.0, nsp=5):
    ns = min(nsp, len(set(g)))
    pred = np.full_like(y, -1)
    for tr, te in GroupKFold(ns).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=42))
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return pred

def confusion(y, p):  # positive = yes(1)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    f1 = f1_score(y, p, average="macro", zero_division=0)
    return acc, f1, tp, tn, fp, fn

def rowfmt(scope, n, acc, f1, tp, tn, fp, fn):
    return f"  {scope:16s} {n:4d}  {acc:5.3f} {f1:5.3f}   {tp:3d} {tn:3d} {fp:3d} {fn:3d}"

out = []
def w(s=""): out.append(s)

for name, path in FILES:
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(int); g = d["groups"].astype(str); q = d["q"].astype(str)
    grp = np.array([group_of(x) for x in q])
    w("#" * 74)
    w(f"# CHECKPOINT: {name}")
    w(f"# n={len(y)}  yes={int(y.sum())} no={int((1-y).sum())}  "
      f"groups: " + " ".join(f"{gg}={int((grp==gg).sum())}" for gg in GROUP_ORDER if (grp==gg).any()))
    w("#" * 74)
    fs = featsets(d)
    for fname, X in fs.items():
        p = oof_pred(X, y, g)
        w(f"\n  -- feature set: {fname} --")
        w(f"  {'scope':16s} {'n':>4s}  {'acc':>5s} {'F1':>5s}   {'TP':>3s} {'TN':>3s} {'FP':>3s} {'FN':>3s}")
        w(rowfmt("OVERALL", len(y), *confusion(y, p)))
        for gg in GROUP_ORDER:
            m = grp == gg
            if m.sum() == 0:
                continue
            w(rowfmt(gg, int(m.sum()), *confusion(y[m], p[m])))
    w("")

txt = "\n".join(out)
open("/tmp/approachA_leaderboard.txt", "w").write(txt)
print(txt)
print("\n[saved -> /tmp/approachA_leaderboard.txt]")
