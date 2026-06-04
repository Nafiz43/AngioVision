#!/usr/bin/env python3
"""
K@1 Sequence Retrieval Evaluation — AngioVision DSA
====================================================

Pipeline
────────
  1. Load labeled_DSA_2023_10_24.csv; skip rows where angio_run contains
     'other'; group sequences by angio_run category.
     The Best_Image column (1-indexed frame number) is parsed for each row.

  2. Per-category ephemeral split (nothing stored anywhere):
       TRAIN = --scale-down fraction  (default 0.80)
       TEST  = 1 - scale-down         (default 0.20)
     Categories with < --min-seqs sequences are skipped.

  3. For each TRAIN sequence, extract ONLY its Best_Image frame and embed
     with microsoft/rad-dino (model loaded ONCE) → 1 embedding per sequence.
     Ingest into a single in-memory ChromaDB EphemeralClient — no disk I/O.

  4. For each TEST sequence (K=1 retrieval):
       • Extract ONLY its Best_Image frame and embed (1 embedding)
       • Query ChromaDB K=1 with that single embedding
       • Hit ↔ retrieved sequence's angio_run == query sequence's angio_run
     No majority voting — comparison is best-frame ↔ best-frame.

  5. Print per-category accuracy table + save a results bar-chart PNG.

Usage
─────
  python eval_k1_retrieval.py
  python eval_k1_retrieval.py --scale-down 0.8 --seed 42
  python eval_k1_retrieval.py --limit-cats 5      # smoke-test: first N cats
  python eval_k1_retrieval.py --device cpu        # force CPU

Arguments
─────────
  --scale-down   FLOAT  Train fraction per category   [default: 0.80]
  --embed-batch  INT    Frames per RAD-DINO fwd pass  [default: 16]
  --min-seqs     INT    Min seqs/category to include  [default: 3]
  --seed         INT    RNG seed                      [default: 42]
  --device       STR    'cuda' | 'cpu' (auto if omit)
  --limit-cats   INT    Process only first N categories (smoke-test)
  --labeled-csv  PATH
  --dicom-root   PATH
  --sqlite-db    PATH   SQLite staging DB for fast DICOM index
  --out-plot     PATH   Output bar-chart PNG          [default: k1_results.png]

Note: --max-frames is removed. Each sequence contributes exactly ONE frame
      (its Best_Image) to both ingestion and querying.
"""

import os
import sys
import csv
import random
import logging
import argparse
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print("ERROR: pip install pydicom")
    sys.exit(1)

try:
    import chromadb
except ImportError:
    print("ERROR: pip install chromadb")
    sys.exit(1)

try:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    print("ERROR: pip install transformers torch pillow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    log.warning("matplotlib not found — bar chart will be skipped (pip install matplotlib)")

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_LABELED_CSV = "/data/Deep_Angiography/labeled_DSA_2023_10_24.csv"
DEFAULT_DICOM_ROOT  = "/data/Deep_Angiography/DICOM"
DEFAULT_SQLITE_DB   = "/data/Deep_Angiography/AngioVision/dicom_staging.db"
DEFAULT_OUT_PLOT    = "k1_results.png"

RAD_DINO_MODEL_ID = "microsoft/rad-dino"
CHROMA_COLLECTION = "eval_retrieval"

# ── Category code map (from your EDA script) ──────────────────────────────────
CODE_MAP: dict[str, str] = {
    "intrahepatic artery":                             "IHA",
    "external iliac artery, right":                    "EIA-R",
    "celiac trunk":                                    "CT",
    "nondiagnostic":                                   "ND",
    "common hepatic artery":                           "CHA",
    "hepatic artery, right":                           "HA-R",
    "superior mesenteric artery (SMA)":                "SMA",
    "proper hepatic artery":                           "PHA",
    "lower abdominal aorta and aortic bifurcation":    "LAA",
    "hepatic artery, left":                            "HA-L",
    "splenic artery":                                  "SA",
    "internal iliac artery, left":                     "IIA-L",
    "inferior mesenteric artery (IMA)":                "IMA",
    "external iliac artery, left":                     "EIA-L",
    "internal iliac artery, right":                    "IIA-R",
    "renal artery, right":                             "RA-R",
    "unstable":                                        "UNS",
    "upper abdominal aorta":                           "UAA",
    "renal artery, left":                              "RA-L",
    "TRAS, extrenal iliac artery, right":              "TRAS-R",
    "common iliac artery, right":                      "CIA-R",
    "common iliac artery, left":                       "CIA-L",
    "common femoral artery, left":                     "CFA-L",
    "TRAS, extrenal iliac artery, left":               "TRAS-L",
}


def code(label: str) -> str:
    return CODE_MAP.get(label, label[:8])


# ═══════════════════════════════════════════════════════════════════════════════
# 1 ── Labeled CSV
# ═══════════════════════════════════════════════════════════════════════════════

def load_labeled_csv_grouped(csv_path: Path) -> dict[str, list[dict]]:
    """
    Load CSV, exclude rows where angio_run contains 'other', group by label.
    Returns { angio_run: [seq_dict, ...] }
    Each seq_dict: accession, series_uid, run_type, angio_run, file_path,
                   best_frame_idx (0-indexed; -1 = missing/invalid → middle frame)

    Best_Image is stored as a 1-indexed float (e.g. 9.0 = frame 9).
    We convert to 0-indexed by subtracting 1.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    skip_other = skip_path = skip_dup = 0
    n_missing_best = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {csv_path}")

        norm = {h.strip().lower(): h for h in reader.fieldnames}

        def col(k: str) -> Optional[str]:
            return norm.get(k.lower())

        path_key  = col("file_path")
        angio_key = col("angio_run")
        acc_key   = col("accession")
        ser_key   = col("seriesuid")
        run_key   = col("run_type")
        best_key  = col("best_image")      # ← NEW

        if not path_key:
            raise ValueError("'file_path' column not found in labeled CSV")

        for row in reader:
            fp    = (row.get(path_key)  or "").strip()
            angio = (row.get(angio_key) or "").strip() if angio_key else ""

            if not fp:
                skip_path += 1; continue
            if "other" in angio.lower():
                skip_other += 1; continue
            if fp in seen:
                skip_dup += 1; continue
            seen.add(fp)

            # ── Parse Best_Image → 0-indexed frame index ──────────────────
            best_frame_idx = -1   # sentinel: fall back to middle frame
            if best_key:
                raw_best = (row.get(best_key) or "").strip()
                try:
                    val = float(raw_best)
                    if val >= 1:
                        best_frame_idx = int(val) - 1   # 1-indexed → 0-indexed
                    # val == 0 or negative → treat as missing
                except (ValueError, TypeError):
                    pass
            if best_frame_idx == -1:
                n_missing_best += 1

            groups[angio].append({
                "accession":     (row.get(acc_key)  or "").strip() if acc_key else "",
                "series_uid":    (row.get(ser_key)  or "").strip() if ser_key else "",
                "run_type":      (row.get(run_key)  or "").strip() if run_key else "",
                "angio_run":     angio,
                "file_path":     fp,
                "best_frame_idx": best_frame_idx,   # ← NEW
            })

    total = sum(len(v) for v in groups.values())
    log.info(
        f"CSV loaded — {len(groups)} categories, {total:,} sequences "
        f"(skipped other={skip_other:,}, dup={skip_dup:,}, no-path={skip_path:,})"
    )
    if n_missing_best:
        log.warning(
            f"  {n_missing_best:,} sequences have no valid Best_Image "
            f"— will use middle frame as fallback"
        )
    return dict(groups)


# ═══════════════════════════════════════════════════════════════════════════════
# 2 ── DICOM index
# ═══════════════════════════════════════════════════════════════════════════════

def _stem(file_path_str: str) -> Optional[str]:
    """Extract .dcm filename stem from a CSV file_path value."""
    p = (file_path_str or "").strip()
    if not p:
        return None
    fname = p.rsplit("/", 1)[-1] if "/" in p else p
    return fname[:-4] if fname.lower().endswith(".dcm") else (fname or None)


def build_dicom_index(
    dicom_root: Path,
    sqlite_db: Optional[Path] = None,
) -> dict[str, str]:
    """
    Build { stem → absolute_path } for all .dcm files.
    Tries SQLite staging DB first (fast O(rows)), then falls back to os.walk.
    """
    if sqlite_db and sqlite_db.exists():
        try:
            con = sqlite3.connect(str(sqlite_db))
            n = con.execute(
                "SELECT COUNT(*) FROM dicom_files WHERE source_file IS NOT NULL"
            ).fetchone()[0]
            if n > 0:
                log.info(f"Building DICOM index from SQLite ({n:,} rows) …")
                idx: dict[str, str] = {}
                for src_file, src_path in con.execute(
                    "SELECT source_file, source_path FROM dicom_files "
                    "WHERE source_file IS NOT NULL AND source_path IS NOT NULL"
                ):
                    s = Path(src_file).stem
                    if s and s not in idx:
                        idx[s] = src_path
                con.close()
                log.info(f"Index ready: {len(idx):,} unique stems (SQLite)")
                return idx
            con.close()
        except Exception as e:
            log.warning(f"SQLite index failed ({e}) — falling back to filesystem walk …")

    log.info(f"Walking {dicom_root} to build DICOM index …")
    idx = {}
    dups = 0
    for dp, _, fnames in os.walk(str(dicom_root)):
        for fname in fnames:
            if fname.lower().endswith(".dcm"):
                s = Path(fname).stem
                if s in idx:
                    dups += 1
                else:
                    idx[s] = str(Path(dp) / fname)
    log.info(f"Index ready: {len(idx):,} unique stems ({dups} dups ignored)")
    return idx


def resolve_paths(
    groups: dict[str, list[dict]],
    dicom_index: dict[str, str],
) -> dict[str, list[dict]]:
    """
    Attach 'dicom_path' and 'stem' to each seq_dict.
    Sequences whose file is not on disk are dropped.
    """
    resolved: dict[str, list[dict]] = {}
    n_missing = 0

    for label, seqs in groups.items():
        kept = []
        for seq in seqs:
            s = _stem(seq["file_path"])
            if s and s in dicom_index:
                kept.append({**seq, "dicom_path": dicom_index[s], "stem": s})
            else:
                n_missing += 1
        if kept:
            resolved[label] = kept

    n_found = sum(len(v) for v in resolved.values())
    log.info(f"Path resolution — found: {n_found:,}, missing (dropped): {n_missing:,}")
    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
# 3 ── Train / test split  (ephemeral — nothing written anywhere)
# ═══════════════════════════════════════════════════════════════════════════════

def split_categories(
    groups: dict[str, list[dict]],
    scale_down: float,
    min_seqs: int,
    seed: int,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """
    Returns { label: (train_list, test_list) }.
    Categories with fewer than min_seqs sequences are excluded.
    The scale_down argument is the TRAIN fraction (e.g. 0.80 → 80% train).
    """
    rng    = random.Random(seed)
    splits = {}
    skipped: list[tuple[str, int]] = []

    log.info(f"Splitting categories (scale_down={scale_down:.0%} train) …")
    for label, seqs in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(seqs)
        if n < min_seqs:
            skipped.append((label, n))
            continue

        shuffled = list(seqs)
        rng.shuffle(shuffled)

        n_train = max(1, round(n * scale_down))
        n_test  = n - n_train
        if n_test == 0:          # guarantee ≥ 1 test sequence
            n_train -= 1
            n_test   = 1

        splits[label] = (shuffled[:n_train], shuffled[n_train:])
        log.info(
            f"  {code(label):8s}  total={n:4d}  "
            f"train={n_train:4d}  test={n_test:4d}"
        )

    if skipped:
        log.info(
            f"  Skipped (< {min_seqs} seqs): "
            + ", ".join(f"{code(l)}({n})" for l, n in skipped)
        )
    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# 4 ── RAD-DINO model
# ═══════════════════════════════════════════════════════════════════════════════

def load_rad_dino_model(device: Optional[str] = None):
    """Load microsoft/rad-dino once; return (model, processor, device)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {RAD_DINO_MODEL_ID} on {device} …")
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_ID)
    model     = AutoModel.from_pretrained(RAD_DINO_MODEL_ID).eval().to(device)
    log.info("RAD-DINO ready (768-dim CLS embeddings)")
    return model, processor, device


def embed_frames(
    frames: list[np.ndarray],
    model,
    processor,
    device: str,
    batch_size: int = 16,
) -> list[list[float]]:
    """
    Embed a list of HxWx3 uint8 numpy frames.
    Returns L2-normalised 768-dim float vectors.
    """
    all_embs: list[list[float]] = []
    for i in range(0, len(frames), batch_size):
        batch  = frames[i : i + batch_size]
        pil    = [PILImage.fromarray(f) for f in batch]
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]           # (B, 768)
            cls = cls / cls.norm(dim=-1, keepdim=True)     # L2-normalise
        all_embs.extend(cls.cpu().float().numpy().tolist())
    return all_embs


# ═══════════════════════════════════════════════════════════════════════════════
# 5 ── DICOM frame extraction  (Best_Image only)
# ═══════════════════════════════════════════════════════════════════════════════

def _to_uint8_rgb(frame: np.ndarray, photometric: str) -> np.ndarray:
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    f = ((f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f)).astype(np.uint8)
    if "MONOCHROME1" in photometric.upper():
        f = 255 - f
    return np.stack([f, f, f], axis=-1)


def extract_single_frame(
    path_str: str,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """
    Extract exactly ONE frame from a DICOM file and return it as uint8 RGB.

    frame_idx is 0-indexed (already converted from Best_Image's 1-indexed value
    during CSV loading).

    Fallback rules:
      • frame_idx == -1  → Best_Image was missing/invalid; use middle frame
      • frame_idx out of range for this file → clamp to middle frame + warn

    Returns None on any read error.
    """
    try:
        ds          = pydicom.dcmread(path_str, force=False)
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        pixels      = ds.pixel_array
    except Exception as e:
        log.debug(f"Frame read failed [{path_str}]: {e}")
        return None

    # Normalise to (T, H, W) or (T, H, W, C)
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]
    elif pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        pixels = pixels[np.newaxis]

    total = pixels.shape[0]

    if frame_idx < 0 or frame_idx >= total:
        if frame_idx != -1:
            log.debug(
                f"Best_Image index {frame_idx} out of range [0,{total-1}] "
                f"for {Path(path_str).name} — using middle frame"
            )
        idx = total // 2
    else:
        idx = frame_idx

    raw = pixels[idx]
    if raw.ndim == 3:
        f  = raw.astype(np.float32)
        lo, hi = f.min(), f.max()
        f  = ((f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f)).astype(np.uint8)
        return f[:, :, :3]
    else:
        return _to_uint8_rgb(raw, photometric)


# ═══════════════════════════════════════════════════════════════════════════════
# 6 ── ChromaDB  (in-memory — no disk writes)
# ═══════════════════════════════════════════════════════════════════════════════

def create_ephemeral_collection():
    """Return a fresh in-memory ChromaDB collection (EphemeralClient)."""
    try:
        client = chromadb.EphemeralClient()
    except AttributeError:
        # Older chromadb versions — use Settings(chroma_db_impl="duckdb+parquet")
        # workaround: create with no path
        client = chromadb.Client()   # deprecated but functional

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("In-memory ChromaDB collection created (no disk I/O)")
    return collection


def ingest_train_sequences(
    splits: dict[str, tuple[list, list]],
    collection,
    model, processor, device: str,
    embed_batch: int,
) -> int:
    """
    For each TRAIN sequence, extract its Best_Image frame, embed it with
    RAD-DINO, and add to ChromaDB.  One frame → one embedding per sequence.
    Returns total embeddings ingested.
    """
    train_seqs = [seq for _, (train, _) in splits.items() for seq in train]
    log.info(
        f"Ingesting {len(train_seqs):,} TRAIN sequences "
        f"(1 Best_Image frame each) …"
    )

    total_ingested = 0
    skipped_seqs   = 0
    fallback_count = 0

    with tqdm(total=len(train_seqs), unit="seq", desc="  Ingesting TRAIN") as pbar:
        for seq in train_seqs:
            frame_idx = seq["best_frame_idx"]
            if frame_idx == -1:
                fallback_count += 1

            frame = extract_single_frame(seq["dicom_path"], frame_idx)
            if frame is None:
                skipped_seqs += 1
                pbar.update(1)
                continue

            emb  = embed_frames([frame], model, processor, device, embed_batch)
            stem = seq["stem"]

            collection.add(
                embeddings=emb,
                ids=[stem],                   # one ID per sequence (the stem)
                metadatas=[{
                    "sequence_id":    stem,
                    "angio_run":      seq["angio_run"],
                    "accession":      seq["accession"],
                    "run_type":       seq["run_type"],
                    "best_frame_idx": frame_idx,
                }],
            )
            total_ingested += 1
            pbar.update(1)
            pbar.set_postfix(ingested=total_ingested, skip=skipped_seqs)

    log.info(
        f"Ingestion done — {total_ingested:,} sequences stored | "
        f"{skipped_seqs} skipped (unreadable) | "
        f"{fallback_count} used middle-frame fallback"
    )
    return total_ingested


# ═══════════════════════════════════════════════════════════════════════════════
# 7 ── K=1 sequence-level evaluation  (Best_Image frame only)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_one_sequence(
    seq: dict,
    collection,
    model, processor, device: str,
    embed_batch: int,
) -> Optional[tuple[bool, str, str]]:
    """
    Evaluate a single test sequence using its Best_Image frame only.

    Query strategy
    ──────────────
    Extract the single Best_Image frame → embed with RAD-DINO (1 vector) →
    query ChromaDB K=1 → the returned entry is a TRAIN sequence's best frame.
    The prediction is that train sequence's angio_run label.

    Hit ↔ predicted label == true label.

    This is a clean 1-to-1 best-frame comparison: the most representative
    frame of the test sequence is matched against the most representative
    frame of every train sequence.  No aggregation or voting needed.

    Returns (hit, true_label, predicted_label) or None if the frame
    could not be extracted.
    """
    frame_idx = seq["best_frame_idx"]
    frame     = extract_single_frame(seq["dicom_path"], frame_idx)
    if frame is None:
        return None

    emb = embed_frames([frame], model, processor, device, embed_batch)

    result = collection.query(
        query_embeddings=emb,
        n_results=1,
        include=["metadatas"],
    )

    if not result["metadatas"] or not result["metadatas"][0]:
        return None

    meta            = result["metadatas"][0][0]
    predicted_label = meta.get("angio_run", "")
    true_label      = seq["angio_run"]
    hit             = (predicted_label == true_label)

    return hit, true_label, predicted_label


def run_evaluation(
    splits: dict[str, tuple[list, list]],
    collection,
    model, processor, device: str,
    embed_batch: int,
) -> dict[str, dict]:
    """
    Evaluate all TEST sequences across all categories.
    Returns per-category result dict.
    """
    results: dict[str, dict] = {
        label: {
            "n_train":      len(train),
            "n_test":       len(test),
            "correct":      0,
            "total_evaled": 0,
            "skipped":      0,
            "predictions":  [],    # list of (true_label, predicted_label)
        }
        for label, (train, test) in splits.items()
    }

    all_test = [seq for _, (_, test) in splits.items() for seq in test]
    log.info(f"Evaluating {len(all_test):,} TEST sequences (Best_Image frame each) …")

    with tqdm(total=len(all_test), unit="seq", desc="  Evaluating TEST") as pbar:
        for seq in all_test:
            label   = seq["angio_run"]
            outcome = evaluate_one_sequence(
                seq, collection, model, processor, device, embed_batch,
            )

            r = results[label]
            if outcome is None:
                r["skipped"] += 1
            else:
                hit, true_lbl, pred_lbl = outcome
                r["total_evaled"] += 1
                r["correct"]      += int(hit)
                r["predictions"].append((true_lbl, pred_lbl))

            pbar.update(1)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8 ── Results reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: dict[str, dict], scale_down: float) -> None:
    """Print a formatted per-category K@1 accuracy table to stdout."""

    rows: list[tuple] = []
    for label, r in results.items():
        acc = r["correct"] / r["total_evaled"] if r["total_evaled"] > 0 else None
        rows.append((label, code(label), r, acc))
    rows.sort(key=lambda x: (x[3] is None, -(x[3] or 0)))

    W = max(len(lbl) for lbl, *_ in rows) + 2

    hdr = (
        f"  {'Category':<{W}}  {'Code':<8}  "
        f"{'Train':>6}  {'Test':>5}  "
        f"{'Evaled':>6}  {'Skip':>5}  "
        f"{'Correct':>7}  {'K@1 Acc':>8}"
    )
    sep = "─" * (len(hdr))

    print()
    print(f"  AngioVision  K@1 Retrieval Evaluation"
          f"  (scale_down={scale_down:.0%} train / {1-scale_down:.0%} test)")
    print(sep)
    print(hdr)
    print(sep)

    macro_accs:   list[float] = []
    total_correct = total_evaled = 0

    for label, cd, r, acc in rows:
        acc_str = f"{acc:>8.1%}" if acc is not None else "     N/A"
        print(
            f"  {label:<{W}}  {cd:<8}  "
            f"{r['n_train']:>6,}  {r['n_test']:>5,}  "
            f"{r['total_evaled']:>6,}  {r['skipped']:>5,}  "
            f"{r['correct']:>7,}  {acc_str}"
        )
        if acc is not None:
            macro_accs.append(acc)
            total_correct += r["correct"]
            total_evaled  += r["total_evaled"]

    macro  = sum(macro_accs) / len(macro_accs) if macro_accs else 0.0
    micro  = total_correct / total_evaled       if total_evaled > 0 else 0.0

    print(sep)
    pad = " " * (W + 10)
    print(f"  {'MACRO AVERAGE':<{W}}  {'':8}  {pad}  {'':>7}  {macro:>8.1%}")
    print(f"  {'OVERALL (micro)':<{W}}  {'':8}  {pad}  "
          f"{total_correct:>7,}  {micro:>8.1%}")
    print(sep)

    # Confusion breakdown (top predicted labels per true category)
    print()
    print("  Predicted-label distribution per true category (top-3 predictions):")
    print()
    for label, cd, r, acc in rows:
        preds = r["predictions"]
        if not preds:
            continue
        counts = Counter(pred for _, pred in preds)
        top    = counts.most_common(3)
        parts  = [
            f"{code(p)}({'✓' if p == label else '✗'})={n}"
            for p, n in top
        ]
        marker = "✓" if (acc is not None and acc >= 0.5) else "✗"
        print(f"  {cd:<8}  [{marker}]  " + "   ".join(parts))
    print()


def save_bar_chart(
    results: dict[str, dict],
    scale_down: float,
    out_path: Path,
) -> None:
    """Save a per-category K@1 accuracy bar chart + reference legend."""
    if not MATPLOTLIB_OK:
        log.warning("matplotlib unavailable — chart not saved")
        return

    rows = []
    for label, r in results.items():
        if r["total_evaled"] > 0:
            acc = r["correct"] / r["total_evaled"]
            rows.append((code(label), label, acc, r["n_train"], r["n_test"]))
    rows.sort(key=lambda x: -x[2])

    if not rows:
        log.warning("No evaluated categories — chart skipped")
        return

    codes  = [r[0] for r in rows]
    accs   = [r[2] for r in rows]
    labels = [r[1] for r in rows]
    n_seqs = [r[3] + r[4] for r in rows]

    # Colour: green-ish for high accuracy, red-ish for low
    colours = [
        "#2ecc71" if a >= 0.80
        else "#f39c12" if a >= 0.50
        else "#e74c3c"
        for a in accs
    ]

    fig, (ax_bar, ax_leg) = plt.subplots(
        1, 2,
        figsize=(16, max(6, len(rows) * 0.45 + 2)),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    y_pos = range(len(rows))
    bars  = ax_bar.barh(
        y_pos, accs,
        color=colours, edgecolor="black", linewidth=0.5, height=0.7,
    )

    for i, (bar, acc_val, n) in enumerate(zip(bars, accs, n_seqs)):
        ax_bar.text(
            min(acc_val + 0.012, 0.99), i,
            f"{acc_val:.1%}  (n={n})",
            va="center", fontsize=8, fontweight="bold",
        )

    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(codes, fontsize=9)
    ax_bar.set_xlim(0, 1.15)
    ax_bar.axvline(x=0.5,  color="gray",  linewidth=1, linestyle="--", alpha=0.5)
    ax_bar.axvline(x=0.80, color="green", linewidth=1, linestyle="--", alpha=0.5)
    ax_bar.set_xlabel("K@1 Retrieval Accuracy", fontsize=10)
    ax_bar.set_title(
        f"AngioVision K@1 Retrieval  "
        f"(train={scale_down:.0%} / test={1-scale_down:.0%})",
        fontsize=11, fontweight="bold",
    )
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="≥ 80% accuracy"),
        mpatches.Patch(color="#f39c12", label="50–79% accuracy"),
        mpatches.Patch(color="#e74c3c", label="< 50% accuracy"),
    ]
    ax_bar.legend(handles=legend_patches, fontsize=8, loc="lower right")

    # ── Reference legend table ────────────────────────────────────────────────
    ax_leg.axis("off")
    table_data  = [[cd, lbl[:45]] for cd, lbl, *_ in rows]
    col_headers = ["Code", "Full Label"]

    tbl = ax_leg.table(
        cellText=table_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.12)

    for col in range(2):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")
    for row_i in range(1, len(rows) + 1):
        bg = "#f0f4f8" if row_i % 2 == 0 else "white"
        for col in range(2):
            tbl[(row_i, col)].set_facecolor(bg)

    tbl.auto_set_column_width([0, 1])
    ax_leg.set_title("Code Reference", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Bar chart saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9 ── Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K@1 sequence retrieval evaluation — AngioVision DSA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labeled-csv",  default=DEFAULT_LABELED_CSV)
    parser.add_argument("--dicom-root",   default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--sqlite-db",    default=DEFAULT_SQLITE_DB)
    parser.add_argument("--out-plot",     default=DEFAULT_OUT_PLOT,
                        help="Path for the output bar-chart PNG")
    parser.add_argument(
        "--scale-down", type=float, default=0.80,
        help=(
            "Fraction of each category's sequences used for TRAIN. "
            "The remaining (1 - scale_down) fraction becomes TEST. "
            "E.g. 0.80 = 80%% train, 20%% test  [default: 0.80]"
        ),
    )
    parser.add_argument(
        "--embed-batch", type=int, default=16,
        help="Frames per RAD-DINO forward pass  [default: 16]",
    )
    parser.add_argument(
        "--min-seqs", type=int, default=3,
        help="Minimum sequences per category to include  [default: 3]",
    )
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument(
        "--limit-cats", type=int, default=0,
        help="Evaluate only the first N categories (smoke-test mode)",
    )

    args = parser.parse_args()

    log.info("═" * 60)
    log.info("  AngioVision K@1 Retrieval Evaluation  [Best_Image mode]")
    log.info("═" * 60)
    log.info(f"  scale-down    : {args.scale_down:.0%} train / "
             f"{1 - args.scale_down:.0%} test")
    log.info(f"  frame mode    : Best_Image only (1 frame per sequence)")
    log.info(f"  min-seqs      : {args.min_seqs}")
    log.info(f"  seed          : {args.seed}")
    log.info(f"  embed-batch   : {args.embed_batch}")
    log.info(f"  limit-cats    : {args.limit_cats or 'all'}")
    log.info("═" * 60)

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    groups = load_labeled_csv_grouped(Path(args.labeled_csv))

    # ── 2. Build DICOM index ──────────────────────────────────────────────────
    dicom_index = build_dicom_index(Path(args.dicom_root), Path(args.sqlite_db))

    # ── 3. Resolve file paths ─────────────────────────────────────────────────
    groups = resolve_paths(groups, dicom_index)
    if not groups:
        log.error("No sequences with resolvable DICOM paths — aborting")
        sys.exit(1)

    # ── 4. Train / test split ─────────────────────────────────────────────────
    splits = split_categories(groups, args.scale_down, args.min_seqs, args.seed)
    if not splits:
        log.error("No categories remain after filtering — aborting")
        sys.exit(1)

    if args.limit_cats > 0:
        splits = dict(list(splits.items())[: args.limit_cats])
        log.info(f"--limit-cats={args.limit_cats}: restricted to {list(splits.keys())}")

    # ── 5. Load RAD-DINO ──────────────────────────────────────────────────────
    model, processor, device = load_rad_dino_model(args.device)

    # ── 6. Create in-memory ChromaDB (no disk writes) ─────────────────────────
    collection = create_ephemeral_collection()

    # ── 7. Ingest ALL train sequences ─────────────────────────────────────────
    n_ingested = ingest_train_sequences(
        splits, collection, model, processor, device,
        args.embed_batch,
    )
    log.info(f"ChromaDB total items: {collection.count():,}")

    if collection.count() == 0:
        log.error("ChromaDB is empty after ingestion — check DICOM paths")
        sys.exit(1)

    # ── 8. Evaluate all test sequences ───────────────────────────────────────
    results = run_evaluation(
        splits, collection, model, processor, device,
        args.embed_batch,
    )

    # ── 9. Report ─────────────────────────────────────────────────────────────
    print_results_table(results, args.scale_down)
    save_bar_chart(results, args.scale_down, Path(args.out_plot))


if __name__ == "__main__":
    main()