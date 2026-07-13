#!/usr/bin/env python3
"""Linear probe: is the yes/no answer linearly decodable from the embeddings?

Freezes a checkpoint, embeds every validation sequence and question, then
cross-validates a logistic regression on (sequence-emb, question-emb) features
against the ground-truth answers. Folds are grouped by accession so no study
leaks across train/test.

Interpretation:
  probe accuracy >> majority baseline  -> the information is in the embeddings;
                                          the zero-shot readout is the bottleneck.
  probe accuracy ~= majority baseline  -> the embeddings don't encode the answers;
                                          the training signal must change.

    python3 eval_linear_probe.py --checkpoint checkpoints/<run>/epoch_5.pt \
        --validation_csv <gt.csv> --data_dir <DICOM_Sequence_Processed>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from angio_ft.common import get_vit_processor  # noqa: E402
from angio_ft.models import PooledCLIP  # noqa: E402
from angio_ft.qa_eval import (  # noqa: E402
    find_frames_dir,
    find_validation_sequence_dirs,
    list_images_in_dir,
    make_yes_no_hypotheses_tagged,
    normalize_str,
    read_metadata_key_value_csv,
)
from transformers import AutoTokenizer  # noqa: E402


# ── data ────────────────────────────────────────────────────────────────────

def load_gt_rows(validation_csv: str) -> pd.DataFrame:
    """Yes/no GT rows keyed by normalized SOPInstanceUID."""
    df = pd.read_csv(validation_csv)
    df = df[["Accession", "SOPInstanceUID", "Question", "Answer"]].dropna()
    df["Answer"] = df["Answer"].astype(str).str.strip().str.lower()
    df = df[df["Answer"].isin(["yes", "no"])].copy()
    df["sop_norm"] = df["SOPInstanceUID"].map(normalize_str)
    return df


def build_model(ckpt_path: Path, device: torch.device) -> Tuple[PooledCLIP, dict]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = PooledCLIP(
        vit_name=cfg["vit_name"], text_model_name=cfg["bert_name"],
        embed_dim=cfg["embed_dim"], arch=cfg["arch"],
        frame_pooling=cfg["frame_pooling"], sequence_pooling=cfg["sequence_pooling"],
        temporal_mode=cfg["temporal_mode"], temporal_on_frames=cfg["temporal_on_frames"],
        temporal_on_sequences=cfg["temporal_on_sequences"],
        frame_temporal_scale=cfg["frame_temporal_scale"],
        sequence_temporal_scale=cfg["sequence_temporal_scale"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, cfg


def embed_sequences(
    model: PooledCLIP, processor, device: torch.device, data_dir: Path,
    wanted_sops: set, frame_chunk_size: int, vit_image_size: Optional[int],
) -> Dict[str, np.ndarray]:
    """sop_norm -> L2-normalized pooled sequence embedding."""
    out: Dict[str, np.ndarray] = {}
    for seq_dir in tqdm(find_validation_sequence_dirs(data_dir), desc="Embedding sequences"):
        acc, sop, status = read_metadata_key_value_csv(seq_dir)
        if status != "ok":
            continue
        sop_norm = normalize_str(sop)
        if sop_norm not in wanted_sops or sop_norm in out:
            continue
        frames_dir = find_frames_dir(seq_dir)
        if frames_dir is None:
            continue
        frame_paths = list_images_in_dir(frames_dir)
        if not frame_paths:
            continue
        emb, emb_status = model.encode_sequence_from_frames(
            processor=processor, frame_paths=frame_paths, device=device,
            frame_chunk_size=frame_chunk_size, max_frames=None,
            vit_image_size=vit_image_size,
        )
        if emb_status == "ok" and emb is not None:
            out[sop_norm] = emb.squeeze(0).float().cpu().numpy()
    return out


def embed_questions(
    model: PooledCLIP, tokenizer, device: torch.device, questions: List[str],
) -> Dict[str, np.ndarray]:
    """question -> [q_emb ; yes_hyp_emb ; no_hyp_emb] (each L2-normalized)."""
    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for q in tqdm(sorted(set(questions)), desc="Embedding questions"):
            yes_h, no_h, _ = make_yes_no_hypotheses_tagged(q)
            embs = model.encode_text(tokenizer, [q, yes_h, no_h], device=device)
            out[q] = embs.float().cpu().numpy().reshape(-1)
    return out


# ── features ────────────────────────────────────────────────────────────────

def build_features(
    gt: pd.DataFrame, seq_embs: Dict[str, np.ndarray], q_embs: Dict[str, np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns X, y (1=yes), groups (accession), and the kept GT rows."""
    keep = gt[gt["sop_norm"].isin(seq_embs)].reset_index(drop=True)
    feats, dim = [], None
    for _, r in keep.iterrows():
        s = seq_embs[r["sop_norm"]]
        qcat = q_embs[r["Question"]]
        d = s.shape[0]
        q, yes_h, no_h = qcat[:d], qcat[d:2 * d], qcat[2 * d:]
        if mode == "concat":
            f = np.concatenate([s, q])
        elif mode == "interact":
            f = np.concatenate([s, q, s * q])
        elif mode == "hyp":
            f = np.concatenate([s, s * yes_h, s * no_h,
                                [s @ yes_h, s @ no_h, s @ yes_h - s @ no_h]])
        else:
            raise ValueError(f"unknown feature mode: {mode}")
        feats.append(f)
    X = np.stack(feats)
    y = (keep["Answer"] == "yes").to_numpy().astype(int)
    groups = keep["Accession"].astype(str).to_numpy()
    return X, y, groups, keep


# ── probe ───────────────────────────────────────────────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5, C: float = 1.0, seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=5000, random_state=seed))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "accs": [round(a, 4) for a in accs]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--validation_csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--frame_chunk_size", type=int, default=64)
    ap.add_argument("--feature_modes", default="concat,interact,hyp")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(Path(a.checkpoint), device)
    print(f"[INFO] checkpoint={a.checkpoint} arch={cfg['arch']} device={device}")
    processor = get_vit_processor(cfg["vit_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_name"])

    gt = load_gt_rows(a.validation_csv)
    print(f"[INFO] GT yes/no rows: {len(gt)} (yes={int((gt['Answer']=='yes').sum())})")

    seq_embs = embed_sequences(model, processor, device, Path(a.data_dir),
                               set(gt["sop_norm"]), a.frame_chunk_size,
                               cfg.get("vit_image_size"))
    q_embs = embed_questions(model, tokenizer, device, list(gt["Question"]))
    print(f"[INFO] embedded sequences: {len(seq_embs)}")

    for mode in a.feature_modes.split(","):
        X, y, groups, keep = build_features(gt, seq_embs, q_embs, mode)
        maj = max(y.mean(), 1 - y.mean())
        res = cross_validate(X, y, groups, C=a.C)
        print(f"[RESULT] mode={mode:<9} n={len(y)} majority={maj:.3f} "
              f"probe_acc={res['acc_mean']:.3f}±{res['acc_std']:.3f} "
              f"F1={res['f1_mean']:.3f} folds={res['accs']}")


if __name__ == "__main__":
    main()
