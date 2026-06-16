#!/usr/bin/env python3
"""
AngioVision — System Architecture Diagram
==========================================
Renders a publication-quality architecture figure that accurately reflects the
AngioVision tool: the three storage layers (SQLite metadata, ChromaDB vectors,
on-disk pixel arrays), the two ingestion pipelines, and the Flask web interface
with its NL->SQL and image-similarity query paths, all running locally.

Output: fig1_architecture.pdf  and  fig1_architecture.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Palette (mirrors the dark UI accent scheme described in the tool)
# ----------------------------------------------------------------------------
C = {
    "bg":        "#ffffff",
    "ink":       "#1f2630",
    "subink":    "#55617a",
    "ingest":    "#1f6feb",   # blue   - ingestion pipelines
    "ingest_bg": "#e9f1fe",
    "store":     "#0e9e8e",   # teal   - storage layers
    "store_bg":  "#e3f6f3",
    "disk":      "#b4690e",   # amber  - on-disk pixels
    "disk_bg":   "#fbeede",
    "serve":     "#7c3aed",   # violet - web/serving + models
    "serve_bg":  "#f0e9fd",
    "model":     "#c2410c",   # orange - ML / LLM models
    "model_bg":  "#fbe9df",
    "edge":      "#c4ccd8",
    "user":      "#111827",
}

FS_TITLE = 18
FS_GROUP = 14
FS_BOX   = 12.5
FS_SUB   = 10.3
FS_EDGE  = 9.5


def box(ax, x, y, w, h, title, subtitle=None, fc="#fff", ec="#000",
        tc="#000", fs=FS_BOX, bold=True, radius=0.022):
    """Draw a rounded box with a title and optional subtitle."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(p)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
                fontsize=FS_SUB, color=tc, zorder=4, style="italic", alpha=0.9)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=tc, zorder=4)
    return (x, y, w, h)


def group(ax, x, y, w, h, label, ec, fc):
    """Draw a translucent container that groups related boxes."""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.03",
        linewidth=1.4, edgecolor=ec, facecolor=fc, alpha=0.5,
        linestyle=(0, (5, 3)), zorder=1,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.17, label, ha="center", va="top",
            fontsize=FS_GROUP, fontweight="bold", color=ec, zorder=2)


def arrow(ax, p0, p1, color=C["edge"], label=None, lw=1.8, style="-|>",
          rad=0.0, ls="-", label_pos=0.5, label_dx=0.0, label_dy=0.12):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=15,
        linewidth=lw, color=color, zorder=5,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * label_pos + label_dx
        my = p0[1] + (p1[1] - p0[1]) * label_pos + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=FS_EDGE,
                color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.9))


def edge_anchor(b, side):
    x, y, w, h = b
    return {
        "top":    (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "left":   (x, y + h / 2),
        "right":  (x + w, y + h / 2),
    }[side]


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.6, 9.4))
ax.set_xlim(0, 13.6)
ax.set_ylim(0, 9.4)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

# Title + privacy banner
# ax.text(6.8, 9.08, "AngioVision — System Architecture", ha="center",
#         va="center", fontsize=FS_TITLE, fontweight="bold", color=C["ink"])
# ax.text(6.8, 8.74,
#         "Locally hosted on a single GPU Linux server  ·  no PHI leaves the premises",
#         ha="center", va="center", fontsize=11, color=C["subink"], style="italic")

# ===========================================================================
# DATA SOURCES (left column)
# ===========================================================================
group(ax, 0.25, 4.55, 2.55, 3.55, "Data Sources", C["subink"], "#f4f6f9")
b_dcm = box(ax, 0.45, 6.75, 2.15, 0.85, "DICOM Archive",
            ".dcm files on disk", C["disk_bg"], C["disk"], C["disk"])
b_csv = box(ax, 0.45, 5.70, 2.15, 0.80, "Labeled CSV",
            "angio-run labels", C["disk_bg"], C["disk"], C["disk"])
b_rpt = box(ax, 0.45, 4.78, 2.15, 0.70, "Radiology Reports",
            "free-text radrpt", C["disk_bg"], C["disk"], C["disk"])

# ===========================================================================
# INGESTION PIPELINES (center-left)
# ===========================================================================
group(ax, 3.05, 4.05, 3.30, 4.05, "Ingestion Pipelines", C["ingest"], C["ingest_bg"])

b_meta = box(ax, 3.22, 6.65, 2.95, 0.95,
             "DICOM Metadata Pipeline",
             "pydicom · SOPInstanceUID skip", "#ffffff", C["ingest"], C["ingest"])
b_emb = box(ax, 3.22, 5.30, 2.95, 0.95,
            "Image Embedding Pipeline",
            "multi-frame · single-process", "#ffffff", C["ingest"], C["ingest"])
b_rad = box(ax, 3.22, 4.45, 2.95, 0.68,
            "RAD-DINO encoder (ViT-B)",
            "CLS 768-d · L2-norm", C["model_bg"], C["model"], C["model"])

# ===========================================================================
# STORAGE LAYER (center column)
# ===========================================================================
group(ax, 6.70, 4.05, 3.0, 4.05, "Storage Layer", C["store"], C["store_bg"])

b_sql = box(ax, 6.90, 6.65, 2.6, 0.95, "SQLite",
            "dicom_files · radiology_reports", "#ffffff", C["store"], C["store"])
b_chr = box(ax, 6.90, 5.40, 2.6, 0.95, "ChromaDB",
            "cosine HNSW · multi-collection", "#ffffff", C["store"], C["store"])
b_dsk = box(ax, 6.90, 4.18, 2.6, 0.95, "Pixel Arrays on Disk",
            "pixel_path pointer only", C["disk_bg"], C["disk"], C["disk"])

# ===========================================================================
# SERVING + MODELS (right column)
# ===========================================================================
group(ax, 10.05, 2.35, 3.20, 5.75, "Web Interface  (Flask)", C["serve"], C["serve_bg"])

b_api = box(ax, 10.25, 6.65, 2.80, 0.90, "Flask REST API",
            "SSE streaming", "#ffffff", C["serve"], C["serve"])
b_nl  = box(ax, 10.25, 5.55, 2.80, 0.88, "NL → SQL pipeline",
            "generate · repair×2 · synth", "#ffffff", C["serve"], C["serve"])
b_img = box(ax, 10.25, 4.45, 2.80, 0.88, "Image-Similarity query",
            "/api/image-query · top-k", "#ffffff", C["serve"], C["serve"])
b_qwen = box(ax, 10.25, 3.60, 2.80, 0.66, "Qwen3-8B  (Ollama)",
             "local LLM inference", C["model_bg"], C["model"], C["model"])
b_fe  = box(ax, 10.25, 2.55, 2.80, 0.85, "Vanilla-JS Frontend",
            "lightbox · thumbnails · dedup", C["serve_bg"], C["serve"], C["serve"])

# ===========================================================================
# USER
# ===========================================================================
b_user = box(ax, 5.35, 0.80, 2.9, 0.95, "Researcher / Clinician",
             "natural-language & image queries", "#ffffff", C["user"], C["user"])

# ----------------------------------------------------------------------------
# ARROWS — Data sources -> ingestion
# ----------------------------------------------------------------------------
arrow(ax, edge_anchor(b_dcm, "right"), edge_anchor(b_meta, "left"),
      C["ingest"], "tags")
arrow(ax, edge_anchor(b_csv, "right"), edge_anchor(b_emb, "left"),
      C["ingest"], "frames", label_dy=0.15)
arrow(ax, edge_anchor(b_rpt, "right"), (3.35, 6.85),
      C["ingest"], None, rad=-0.18)

# embedding pipeline uses RAD-DINO (internal, ingestion side)
arrow(ax, edge_anchor(b_emb, "bottom"), edge_anchor(b_rad, "top"),
      C["model"], "embed", lw=1.5, label_dx=0.42, label_dy=0.0)

# ----------------------------------------------------------------------------
# ARROWS — ingestion -> storage
# ----------------------------------------------------------------------------
arrow(ax, edge_anchor(b_meta, "right"), edge_anchor(b_sql, "left"),
      C["store"], "write rows")
arrow(ax, edge_anchor(b_rad, "right"), edge_anchor(b_chr, "left"),
      C["store"], "vectors", rad=-0.10, label_dy=0.15)
# pixels written to disk by embedding pipeline
arrow(ax, edge_anchor(b_emb, "right"), edge_anchor(b_dsk, "left"),
      C["disk"], "pixels", rad=-0.28, label_pos=0.5, label_dx=0.0, label_dy=-0.20)

# ----------------------------------------------------------------------------
# ARROWS — storage <-> serving  (query paths)
# ----------------------------------------------------------------------------
# SQL round-trip
arrow(ax, edge_anchor(b_nl, "left"), edge_anchor(b_sql, "right"),
      C["store"], "SQL", rad=-0.16, style="<|-|>", label_dy=0.15)
# vector search round-trip
arrow(ax, edge_anchor(b_img, "left"), edge_anchor(b_chr, "right"),
      C["store"], "ANN top-k", rad=0.16, style="<|-|>", label_dy=-0.16)
# enrichment of image hits from SQLite (dashed)
arrow(ax, (10.25, 4.70), (9.50, 6.65), C["store"], "enrich",
      rad=0.30, lw=1.4, ls=(0, (4, 2)), label_pos=0.30, label_dx=-0.22)
# frame extraction reads pixels on disk (dashed)
arrow(ax, edge_anchor(b_fe, "left"), edge_anchor(b_dsk, "bottom"),
      C["disk"], "frames", rad=0.30, lw=1.4, ls=(0, (4, 2)),
      label_pos=0.5, label_dy=-0.20)

# ----------------------------------------------------------------------------
# ARROWS — API internal flow
# ----------------------------------------------------------------------------
arrow(ax, edge_anchor(b_api, "bottom"), edge_anchor(b_nl, "top"),
      C["serve"], None, lw=1.4)
arrow(ax, edge_anchor(b_nl, "bottom"), edge_anchor(b_img, "top"),
      C["serve"], None, lw=1.4)
# both NL and image pipelines call the local LLM
arrow(ax, edge_anchor(b_img, "bottom"), edge_anchor(b_qwen, "top"),
      C["model"], "synthesize", lw=1.4, style="<|-|>", label_dx=0.50, label_dy=0.0)

# ----------------------------------------------------------------------------
# ARROWS — user <-> interface
# ----------------------------------------------------------------------------
arrow(ax, edge_anchor(b_user, "right"), (10.25, 2.85),
      C["user"], "query (NL / image)", rad=-0.16, lw=2.0,
      label_pos=0.5, label_dy=0.20)
arrow(ax, edge_anchor(b_fe, "bottom"), edge_anchor(b_user, "top"),
      C["serve"], "streamed answer (SSE)", rad=-0.30, lw=1.8,
      label_pos=0.5, label_dx=0.45, label_dy=-0.08)

# ----------------------------------------------------------------------------
# LEGEND
# ----------------------------------------------------------------------------
legend_items = [
    ("Ingestion pipeline", C["ingest"]),
    ("Storage layer",      C["store"]),
    ("On-disk pixels",     C["disk"]),
    ("Web / serving",      C["serve"]),
    ("ML model",           C["model"]),
]
handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c,
                  markeredgecolor=c, markersize=13, label=l)
           for l, c in legend_items]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.012),
          ncol=1, frameon=True, fontsize=10.5, handletextpad=0.5,
          borderpad=0.7, labelspacing=0.6,
          edgecolor=C["edge"], facecolor="#ffffff")

plt.tight_layout()
fig.savefig("fig1_architecture.png", dpi=200, bbox_inches="tight",
            facecolor=C["bg"])
fig.savefig("fig1_architecture.pdf", bbox_inches="tight", facecolor=C["bg"])
print("Saved fig1_architecture.png and fig1_architecture.pdf")