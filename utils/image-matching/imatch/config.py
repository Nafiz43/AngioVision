"""Constants, model aliases, label codes and output naming for imatch.

All values are ported verbatim from ``utils/metadata_db/eval_kn_retrieval.py``
so results stay bit-for-bit comparable with prior runs.
"""

import os

# ── Data defaults (lab-server paths, overridable via CLI) ─────────────────────
DEFAULT_LABELED_CSV = "/data/Deep_Angiography/labeled_DSA_2023_10_24.csv"
DEFAULT_DICOM_ROOT  = "/data/Deep_Angiography/DICOM"
DEFAULT_SQLITE_DB   = "/data/Deep_Angiography/AngioVision/dicom_staging.db"

DEFAULT_K_VALUES    = [1, 3, 5, 7, 9, 11, 13, 15]
DEFAULT_WORKERS     = max(1, (os.cpu_count() or 4) // 2)
CHROMA_COLLECTION   = "eval_retrieval"
FRAME_MODES         = ("best", "fl", "all")

# ── Embedding-model aliases ───────────────────────────────────────────────────
HF_ALIASES: dict[str, str] = {
    "rad-dino": "microsoft/rad-dino",
    "vit-b16":  "google/vit-base-patch16-224",
    "vit-l16":  "google/vit-large-patch16-224",
}
OPENCLIP_ALIASES: dict[str, tuple[str, str]] = {
    "openclip-b32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "openclip-l14": ("ViT-L-14", "laion2b_s32b_b82k"),
}

# ── Anatomy label → short display code ────────────────────────────────────────
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


def auto_output_paths(
    model_alias: str, frame_mode: str, temporal: bool, split_tag: str,
) -> tuple[str, str, str, str]:
    """Build self-documenting filenames: kn_results_{model}_{mode}_{split_tag}[_temporal].ext

    Returns (markdown, png, docx, csv) filenames.
    """
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in model_alias)
    while "--" in safe:
        safe = safe.replace("--", "-")
    safe = safe.strip("-") or "model"
    parts = ["kn_results", safe, frame_mode, split_tag]
    if temporal:
        parts.append("temporal")
    stem = "_".join(parts)
    return f"{stem}.md", f"{stem}.png", f"{stem}.docx", f"{stem}.csv"
