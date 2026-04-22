#!/usr/bin/env python3
"""
Generate a vis-network HTML file with three selectable graph modes:
  A. Task <-> Architecture
  B. Task <-> Limitations
  C. Task <-> Future Work

Each right-side node stores all originating articles for display in the detail panel.

Usage:
    python generate_network.py \
        --input  /data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl \
        --output /data/Deep_Angiography/AngioVision/slr/results/task_network.html
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def safe_str(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] line {i+1} skipped - {exc}")
    return records


# ---------------------------------------------------------------------------
# colour palette
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "architecture": {
        "background": "#3b82f6", "border": "#1d4ed8",
        "highlight": {"background": "#60a5fa", "border": "#1e40af"},
        "hover":     {"background": "#60a5fa", "border": "#1e40af"},
    },
    "limitation": {
        "background": "#a855f7", "border": "#7e22ce",
        "highlight": {"background": "#c084fc", "border": "#6b21a8"},
        "hover":     {"background": "#c084fc", "border": "#6b21a8"},
    },
    "future_work": {
        "background": "#10b981", "border": "#065f46",
        "highlight": {"background": "#34d399", "border": "#064e3b"},
        "hover":     {"background": "#34d399", "border": "#064e3b"},
    },
}

TASK_COLOR = {
    "background": "#f97316", "border": "#c2410c",
    "highlight": {"background": "#fb923c", "border": "#9a3412"},
    "hover":     {"background": "#fb923c", "border": "#9a3412"},
}


def _make_task_node(tid: int, pt: str, meta: dict, degree: int) -> dict:
    return {
        "id":               tid,
        "label":            pt if len(pt) <= 30 else pt[:28] + "\u2026",
        "node_type":        "task",
        "size":             max(14, min(36, 12 + degree * 3)),
        "primary_task":     meta["primary_task"],
        "secondary_tasks":  meta["secondary_tasks"],
        "task_description": meta["task_description"],
        "example_journal":  meta["example_journal"],
        "color":            TASK_COLOR,
        "font":             {"color": "#fff", "size": 14, "face": "DM Sans, sans-serif"},
        "shape":            "dot",
        "borderWidth":      2,
        "shadow":           True,
    }


# ---------------------------------------------------------------------------
# build all three graph datasets
# ---------------------------------------------------------------------------

def build_graph(records: list[dict]) -> dict:
    task_meta:   dict[str, dict] = {}
    arch_meta:   dict[str, dict] = {}   # key -> {fields, sources: [{title, year, journal, doi, pub_type}]}
    lim_meta:    dict[str, dict] = {}   # key -> {full_text, sources: [...]}
    future_meta: dict[str, dict] = {}   # key -> {full_text, sources: [...]}

    arch_edges:   set[tuple[str, str]] = set()
    lim_edges:    set[tuple[str, str]] = set()
    future_edges: set[tuple[str, str]] = set()

    for rec in records:
        task_block   = rec.get("task",   {}) or {}
        method_block = rec.get("method", {}) or {}
        identity     = rec.get("study_identity", {}) or {}
        eval_block   = rec.get("evaluation", {}) or {}

        pt = (task_block.get("primary_task") or "").strip()
        if not pt:
            continue

        # Source article info attached to every right-side node
        source_entry = {
            "title":       safe_str(rec.get("title")),
            "year":        safe_str(rec.get("year")),
            "journal":     safe_str(identity.get("journal_or_venue")),
            "doi":         safe_str(identity.get("doi")),
            "pub_type":    safe_str(identity.get("publication_type")),
            "authors":     safe_str(identity.get("authors")),
        }

        if pt not in task_meta:
            task_meta[pt] = {
                "primary_task":     pt,
                "secondary_tasks":  safe_str(task_block.get("secondary_tasks")),
                "task_description": safe_str(task_block.get("task_description")),
                "example_journal":  safe_str(identity.get("journal_or_venue")),
            }

        # ── A: architecture ──────────────────────────────────────────────
        an = (method_block.get("architecture_name") or "").strip()
        if an:
            if an not in arch_meta:
                arch_meta[an] = {
                    "architecture_name":    an,
                    "architecture_family":  safe_str(method_block.get("architecture_family")),
                    "input_type":           safe_str(method_block.get("input_type")),
                    "temporal_modelling":   safe_str(method_block.get("temporal_modelling")),
                    "pretrained_backbone":  safe_str(method_block.get("pretrained_backbone")),
                    "training_supervision": safe_str(method_block.get("training_supervision")),
                    "best_metric":          safe_str(eval_block.get("best_metric_value")),
                    "metrics":              safe_str(eval_block.get("metrics")),
                    "validation_design":    safe_str(eval_block.get("validation_design")),
                    "comparators":          safe_str(eval_block.get("comparators")),
                    "sources": [],
                }
            # accumulate sources (avoid exact duplicates)
            existing_titles = {s["title"] for s in arch_meta[an]["sources"]}
            if source_entry["title"] not in existing_titles:
                arch_meta[an]["sources"].append(source_entry)
            arch_edges.add((pt, an))

        # ── B: limitations ───────────────────────────────────────────────
        lim_raw = safe_str(rec.get("limitations") or "")
        if lim_raw and lim_raw != "N/A":
            lk = lim_raw[:80]
            if lk not in lim_meta:
                lim_meta[lk] = {"full_text": lim_raw, "sources": []}
            existing_titles = {s["title"] for s in lim_meta[lk]["sources"]}
            if source_entry["title"] not in existing_titles:
                lim_meta[lk]["sources"].append(source_entry)
            lim_edges.add((pt, lk))

        # ── C: future work ───────────────────────────────────────────────
        fw_raw = safe_str(rec.get("future_work_stated") or "")
        if fw_raw and fw_raw != "N/A":
            fk = fw_raw[:80]
            if fk not in future_meta:
                future_meta[fk] = {"full_text": fw_raw, "sources": []}
            existing_titles = {s["title"] for s in future_meta[fk]["sources"]}
            if source_entry["title"] not in existing_titles:
                future_meta[fk]["sources"].append(source_entry)
            future_edges.add((pt, fk))

    # ── stable IDs ───────────────────────────────────────────────────────
    task_id   = {pt: i          for i, pt in enumerate(sorted(task_meta))}
    arch_id   = {an: i + 10_000 for i, an in enumerate(sorted(arch_meta))}
    lim_id    = {lk: i + 20_000 for i, lk in enumerate(sorted(lim_meta))}
    future_id = {fk: i + 30_000 for i, fk in enumerate(sorted(future_meta))}

    def deg(pt, edge_set):
        return sum(1 for (t, _) in edge_set if t == pt)

    # ── mode A nodes/edges ────────────────────────────────────────────────
    a_nodes, a_edges = [], []
    for pt, meta in task_meta.items():
        d = deg(pt, arch_edges)
        if d:
            a_nodes.append(_make_task_node(task_id[pt], pt, meta, d))
    for an, meta in arch_meta.items():
        d = sum(1 for (_, a) in arch_edges if a == an)
        a_nodes.append({
            "id":    arch_id[an],
            "label": an if len(an) <= 28 else an[:26] + "\u2026",
            "node_type": "architecture",
            "size":  max(12, min(34, 10 + d * 3)),
            "architecture_name":    meta["architecture_name"],
            "architecture_family":  meta["architecture_family"],
            "input_type":           meta["input_type"],
            "temporal_modelling":   meta["temporal_modelling"],
            "pretrained_backbone":  meta["pretrained_backbone"],
            "training_supervision": meta["training_supervision"],
            "best_metric":          meta["best_metric"],
            "metrics":              meta["metrics"],
            "validation_design":    meta["validation_design"],
            "comparators":          meta["comparators"],
            "sources":              meta["sources"],
            "color": NODE_COLORS["architecture"],
            "font":  {"color": "#fff", "size": 13, "face": "DM Sans, sans-serif"},
            "shape": "dot", "borderWidth": 2, "shadow": True,
        })
    for eid, (pt, an) in enumerate(arch_edges):
        a_edges.append({
            "id": eid, "from": task_id[pt], "to": arch_id[an],
            "color": {"color": "#94a3b8", "highlight": "#334155", "hover": "#334155", "opacity": 0.7},
            "width": 1.5, "smooth": {"enabled": True, "type": "dynamic"},
        })

    # ── mode B nodes/edges ────────────────────────────────────────────────
    b_nodes, b_edges = [], []
    for pt, meta in task_meta.items():
        d = deg(pt, lim_edges)
        if d:
            b_nodes.append(_make_task_node(task_id[pt], pt, meta, d))
    for lk, meta in lim_meta.items():
        short = lk[:40] + ("\u2026" if len(lk) >= 40 else "")
        b_nodes.append({
            "id": lim_id[lk], "label": short, "node_type": "limitation", "size": 14,
            "full_text": meta["full_text"],
            "sources":   meta["sources"],
            "color": NODE_COLORS["limitation"],
            "font":  {"color": "#fff", "size": 12, "face": "DM Sans, sans-serif"},
            "shape": "dot", "borderWidth": 2, "shadow": True,
        })
    for eid, (pt, lk) in enumerate(lim_edges):
        b_edges.append({
            "id": eid, "from": task_id[pt], "to": lim_id[lk],
            "color": {"color": "#c084fc", "highlight": "#a855f7", "hover": "#a855f7", "opacity": 0.65},
            "width": 1.5, "smooth": {"enabled": True, "type": "dynamic"},
        })

    # ── mode C nodes/edges ────────────────────────────────────────────────
    c_nodes, c_edges = [], []
    for pt, meta in task_meta.items():
        d = deg(pt, future_edges)
        if d:
            c_nodes.append(_make_task_node(task_id[pt], pt, meta, d))
    for fk, meta in future_meta.items():
        short = fk[:40] + ("\u2026" if len(fk) >= 40 else "")
        c_nodes.append({
            "id": future_id[fk], "label": short, "node_type": "future_work", "size": 14,
            "full_text": meta["full_text"],
            "sources":   meta["sources"],
            "color": NODE_COLORS["future_work"],
            "font":  {"color": "#fff", "size": 12, "face": "DM Sans, sans-serif"},
            "shape": "dot", "borderWidth": 2, "shadow": True,
        })
    for eid, (pt, fk) in enumerate(future_edges):
        c_edges.append({
            "id": eid, "from": task_id[pt], "to": future_id[fk],
            "color": {"color": "#34d399", "highlight": "#10b981", "hover": "#10b981", "opacity": 0.65},
            "width": 1.5, "smooth": {"enabled": True, "type": "dynamic"},
        })

    stats = {
        "total_records":      len(records),
        "task_count":         len(task_meta),
        "architecture_count": len(arch_meta),
        "limitation_count":   len(lim_meta),
        "future_work_count":  len(future_meta),
        "arch_edges":         len(a_edges),
        "lim_edges":          len(b_edges),
        "future_edges":       len(c_edges),
    }

    return {
        "arch":        {"nodes": a_nodes, "edges": a_edges},
        "limitations": {"nodes": b_nodes, "edges": b_edges},
        "future_work": {"nodes": c_nodes, "edges": c_edges},
        "stats":       stats,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SLR Network Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,600;0,800;1,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    [data-theme="dark"] {
      --bg:#0d1117; --surface:#161b22; --surface2:#1c2128; --border:#30363d;
      --text:#e6edf3; --muted:#7d8590; --accent:#58a6ff;
      --shadow:0 8px 32px rgba(0,0,0,.55);
      --net-bg:radial-gradient(ellipse at 50% 40%,#1a2035 0%,#0d1117 100%);
      --status-fg:#3fb950; --status-bg:rgba(63,185,80,.08); --status-br:rgba(63,185,80,.2);
      --badge-bg:rgba(59,130,246,.18); --badge-fg:#58a6ff;
      --btn-sec-bg:#1c2128; --btn-sec-fg:#e6edf3;
      --source-bg:#0d1117; --source-border:#30363d;
    }
    [data-theme="light"] {
      --bg:#f4f7fb; --surface:#fff; --surface2:#f0f4f9; --border:#d0d7e3;
      --text:#0f172a; --muted:#64748b; --accent:#2563eb;
      --shadow:0 8px 32px rgba(15,23,42,.10);
      --net-bg:radial-gradient(ellipse at 50% 40%,#e8f0fe 0%,#f4f7fb 100%);
      --status-fg:#16a34a; --status-bg:rgba(22,163,74,.07); --status-br:rgba(22,163,74,.22);
      --badge-bg:rgba(37,99,235,.10); --badge-fg:#2563eb;
      --btn-sec-bg:#e8eef6; --btn-sec-fg:#0f172a;
      --source-bg:#f8faff; --source-border:#d0d7e3;
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    html,body{height:100%;font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);transition:background .25s,color .25s}

    div.vis-tooltip{
      position:absolute;visibility:hidden;padding:9px 13px;max-width:340px;
      white-space:normal;word-break:break-word;font-size:13px;font-family:'DM Sans',sans-serif;
      color:var(--text);background:var(--surface);border:1px solid var(--border);
      border-radius:10px;box-shadow:var(--shadow);z-index:10;pointer-events:none;line-height:1.5
    }

    /* layout */
    .app{display:flex;flex-direction:row;height:100vh;overflow:hidden}
    .main{display:flex;flex-direction:column;flex:1 1 0;min-width:0;height:100vh;overflow:hidden}

    /* hero */
    .hero{background:var(--surface);border-bottom:1px solid var(--border);padding:10px 18px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
    .hero-text{flex:1;min-width:160px}
    .eyebrow{font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:2px}
    .hero h1{font-size:16px;font-weight:800;line-height:1.2}
    .status-bar{font-size:11px;font-family:'DM Mono',monospace;color:var(--status-fg);background:var(--status-bg);border:1px solid var(--status-br);border-radius:7px;padding:4px 10px;white-space:nowrap;flex-shrink:0}
    .theme-toggle{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:5px 11px;font-size:12px;font-weight:700;font-family:'DM Sans',sans-serif;color:var(--text);cursor:pointer;display:flex;align-items:center;gap:6px;flex-shrink:0;transition:border-color .15s}
    .theme-toggle:hover{border-color:var(--accent)}

    /* mode bar */
    .mode-bar{background:var(--surface2);border-bottom:1px solid var(--border);padding:9px 18px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .mode-label{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-right:4px}
    .mode-btn{padding:7px 15px;border-radius:20px;font-size:12px;font-weight:700;border:2px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .15s;display:flex;align-items:center;gap:7px}
    .mode-btn:hover{border-color:var(--accent);color:var(--text)}
    .mode-btn.active{color:#fff;border-color:transparent}
    .mode-btn.active.arch-mode{background:#3b82f6}
    .mode-btn.active.lim-mode{background:#a855f7}
    .mode-btn.active.fw-mode{background:#10b981}
    .mode-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
    .dot-arch{background:#3b82f6} .dot-lim{background:#a855f7} .dot-fw{background:#10b981}

    /* net card */
    .net-card{background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex:1 1 0;min-height:0;overflow:hidden}
    .net-toolbar{display:flex;justify-content:space-between;align-items:center;padding:8px 16px;border-bottom:1px solid var(--border);background:var(--surface2);gap:12px;flex-wrap:wrap}
    .net-toolbar .title{font-size:13px;font-weight:700}
    .net-summary{font-size:11px;color:var(--muted)}
    .legend{display:flex;gap:12px;font-size:11px;color:var(--muted);flex-wrap:wrap}
    .legend-item{display:flex;align-items:center;gap:5px}
    .leg-dot{width:9px;height:9px;border-radius:50%}

    /* placeholder */
    .net-placeholder{flex:1 1 0;min-height:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;color:var(--muted);background:var(--net-bg)}
    .net-placeholder .big-icon{font-size:44px;opacity:.35}
    .net-placeholder p{font-size:13px;text-align:center;max-width:300px;line-height:1.6}

    #mynetwork{width:100%;background:var(--net-bg);transition:background .25s}

    /* side panel */
    .side{flex:0 0 330px;width:330px;height:100vh;overflow-y:auto;background:var(--surface);border-left:1px solid var(--border);display:flex;flex-direction:column}
    .side-section{padding:12px 14px;border-bottom:1px solid var(--border)}
    .side-section:last-child{border-bottom:none;flex:1}
    .panel-title{font-size:14px;font-weight:800;margin-bottom:2px}
    .panel-sub{font-size:11px;color:var(--muted);line-height:1.4}

    .search-input{width:100%;padding:8px 11px;border:1px solid var(--border);border-radius:8px;background:var(--surface2);color:var(--text);font-size:12px;font-family:'DM Sans',sans-serif;outline:none;transition:border-color .15s}
    .search-input::placeholder{color:var(--muted)}
    .search-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(88,166,255,.12)}

    .btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}
    .btn{padding:6px 11px;border-radius:7px;font-size:11px;font-weight:700;border:1px solid transparent;cursor:pointer;font-family:'DM Sans',sans-serif;transition:opacity .15s,transform .1s}
    .btn:hover{opacity:.85;transform:translateY(-1px)}
    .btn-primary{background:var(--accent);color:#fff}
    .btn-secondary{background:var(--btn-sec-bg);color:var(--btn-sec-fg);border-color:var(--border)}

    .section-head{display:flex;justify-content:space-between;align-items:center;font-weight:700;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:6px}
    .badge{background:var(--badge-bg);color:var(--badge-fg);border-radius:99px;padding:2px 7px;font-size:10px;font-weight:800}

    .stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
    .stat-box{background:var(--surface2);border:1px solid var(--border);border-radius:9px;padding:9px}
    .stat-label{font-size:10px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.05em}
    .stat-val{font-size:20px;font-weight:800;font-family:'DM Mono',monospace}

    .node-list{display:flex;flex-direction:column;gap:4px;max-height:150px;overflow-y:auto}
    .node-btn{width:100%;text-align:left;border:1px solid var(--border);background:var(--surface2);color:var(--text);padding:7px 10px;border-radius:8px;font-size:11px;font-weight:600;cursor:pointer;font-family:'DM Sans',sans-serif;display:flex;align-items:center;gap:6px;transition:border-color .12s,background .12s}
    .node-btn:hover{border-color:var(--accent)}
    .node-btn.selected{background:rgba(88,166,255,.12);border-color:var(--accent);color:var(--accent)}
    .type-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
    .empty-note{border:1px dashed var(--border);border-radius:8px;padding:9px;color:var(--muted);font-size:11px;text-align:center}

    /* detail panel */
    .detail-card{background:var(--surface2);border:1px solid var(--border);border-radius:9px;padding:11px;font-size:11px;color:var(--muted);line-height:1.6}
    .detail-title{font-size:13px;font-weight:800;color:var(--text);margin-bottom:8px;line-height:1.3}
    .detail-section-label{font-size:9px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin-top:10px;margin-bottom:3px}
    .detail-value{color:var(--text);font-size:11px;white-space:pre-wrap;line-height:1.55}
    .tag{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;margin:1px}
    .tag-task{background:rgba(249,115,22,.15);color:#f97316}
    .tag-arch{background:rgba(59,130,246,.15);color:#3b82f6}
    .tag-lim{background:rgba(168,85,247,.15);color:#a855f7}
    .tag-fw{background:rgba(16,185,129,.15);color:#10b981}

    /* source article cards inside detail panel */
    .source-list{display:flex;flex-direction:column;gap:6px;margin-top:4px}
    .source-card{background:var(--source-bg);border:1px solid var(--source-border);border-radius:7px;padding:8px 10px}
    .source-title{font-size:11px;font-weight:700;color:var(--text);margin-bottom:4px;line-height:1.35}
    .source-meta{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:3px}
    .source-chip{font-size:10px;font-weight:600;padding:2px 7px;border-radius:4px;background:var(--surface2);color:var(--muted);border:1px solid var(--border)}
    .source-chip.year{color:#f97316;border-color:rgba(249,115,22,.3);background:rgba(249,115,22,.08)}
    .source-chip.journal{color:var(--accent);border-color:rgba(88,166,255,.3);background:rgba(88,166,255,.08)}
    .source-chip.pub{color:var(--muted)}
    .source-authors{font-size:10px;color:var(--muted);font-style:italic;margin-top:2px;line-height:1.3}
    .source-doi{font-size:10px;color:var(--accent);margin-top:3px;word-break:break-all}
    .source-doi a{color:var(--accent);text-decoration:none}
    .source-doi a:hover{text-decoration:underline}

    @media(max-width:960px){
      .app{flex-direction:column;height:auto}
      .main{height:auto;flex:none}
      #mynetwork{flex:none}
      .side{height:auto;max-height:none;width:100%;flex:none}
    }
  </style>
</head>
<body>
<div class="app">
  <main class="main">

    <div class="hero">
      <div class="hero-text">
        <div class="eyebrow">SLR Stage-2 &middot; AngioVision</div>
        <h1>Task Network Explorer</h1>
      </div>
      <div class="status-bar" id="statusBar">Loading&hellip;</div>
      <button class="theme-toggle" id="themeBtn" onclick="toggleTheme()">
        <span>&#9728;&#65039;</span> Light mode
      </button>
    </div>

    <div class="mode-bar">
      <span class="mode-label">View:</span>
      <button class="mode-btn arch-mode" id="btnModeArch" onclick="setMode('arch')">
        <span class="mode-dot dot-arch"></span> Task &harr; Architecture
      </button>
      <button class="mode-btn lim-mode"  id="btnModeLim"  onclick="setMode('limitations')">
        <span class="mode-dot dot-lim"></span> Task &harr; Limitations
      </button>
      <button class="mode-btn fw-mode"   id="btnModeFw"   onclick="setMode('future_work')">
        <span class="mode-dot dot-fw"></span> Task &harr; Future Work
      </button>
    </div>

    <div class="net-card">
      <div class="net-toolbar">
        <div>
          <div class="title" id="netTitle">No graph selected</div>
          <div class="net-summary" id="netSummary">Select a view above to display the network</div>
        </div>
        <div class="legend" id="netLegend"></div>
      </div>
      <div class="net-placeholder" id="netPlaceholder">
        <div class="big-icon">&#128300;</div>
        <p>Select a graph type above to explore relationships between tasks and other study attributes.</p>
      </div>
      <div id="mynetwork" style="display:none;"></div>
    </div>

  </main>

  <aside class="side">
    <div class="side-section">
      <div class="panel-title">Network Panel</div>
      <div class="panel-sub">Search nodes, click to inspect details.</div>
      <input id="searchBox" class="search-input" type="text" placeholder="Search&hellip;" style="margin-top:7px" />
      <div class="btn-row">
        <button id="btnClearSel" class="btn btn-secondary">Clear</button>
        <button id="btnShowAll"  class="btn btn-primary">Show all</button>
      </div>
    </div>

    <div class="side-section">
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Tasks</div>
          <div class="stat-val" style="color:#f97316">__TASK_COUNT__</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Arch.</div>
          <div class="stat-val" style="color:#3b82f6">__ARCH_COUNT__</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Limits</div>
          <div class="stat-val" style="color:#a855f7">__LIM_COUNT__</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Fut. Work</div>
          <div class="stat-val" style="color:#10b981">__FW_COUNT__</div>
        </div>
      </div>
    </div>

    <div class="side-section">
      <div class="section-head">
        <span>Task nodes</span><span class="badge" id="taskListCount">0</span>
      </div>
      <div id="taskList" class="node-list"></div>
    </div>

    <div class="side-section">
      <div class="section-head">
        <span id="rightListLabel">Right-side nodes</span>
        <span class="badge" id="rightListCount">0</span>
      </div>
      <div id="rightList" class="node-list"></div>
    </div>

    <div class="side-section">
      <div class="section-head" style="margin-bottom:7px">Details</div>
      <div id="detailPanel" class="detail-card">
        Select a graph type, then click any node to inspect its details.
      </div>
    </div>
  </aside>
</div>

<script>
// ── injected data ──────────────────────────────────────────────────────────
const GRAPH_DATA = __GRAPH_DATA__;
const STATS      = __STATS__;
// ──────────────────────────────────────────────────────────────────────────

const nodeMaps = {
  arch:        new Map(GRAPH_DATA.arch.nodes.map(n        => [n.id, n])),
  limitations: new Map(GRAPH_DATA.limitations.nodes.map(n => [n.id, n])),
  future_work: new Map(GRAPH_DATA.future_work.nodes.map(n  => [n.id, n])),
};

let network    = null;
let nodesDS    = null;
let edgesDS    = null;
let activeMode = null;
let selectedId = null;
let searchText = "";

const MODE_META = {
  arch:        { title:"Task \u2194 Architecture", rightLabel:"Architecture nodes", rightType:"architecture", legendColor:"#3b82f6", legendLabel:"Architecture" },
  limitations: { title:"Task \u2194 Limitations",  rightLabel:"Limitation nodes",   rightType:"limitation",   legendColor:"#a855f7", legendLabel:"Limitation"   },
  future_work: { title:"Task \u2194 Future Work",   rightLabel:"Future Work nodes",  rightType:"future_work",  legendColor:"#10b981", legendLabel:"Future Work"  },
};

function esc(v){
  return String(v??"N/A")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

// ── source article card HTML ───────────────────────────────────────────────
function renderSourceCards(sources){
  if(!sources||!sources.length) return '<div class="empty-note" style="margin-top:4px">No source articles recorded.</div>';
  return '<div class="source-list">' + sources.map(s => {
    const yearChip  = s.year  && s.year  !== "N/A" ? `<span class="source-chip year">${esc(s.year)}</span>` : "";
    const jChip     = s.journal && s.journal !== "N/A" ? `<span class="source-chip journal">${esc(s.journal)}</span>` : "";
    const ptChip    = s.pub_type && s.pub_type !== "N/A" ? `<span class="source-chip pub">${esc(s.pub_type)}</span>` : "";
    const authLine  = s.authors && s.authors !== "N/A"
      ? `<div class="source-authors">${esc(s.authors.length > 80 ? s.authors.slice(0,77)+"..." : s.authors)}</div>` : "";
    const doiLine   = s.doi && s.doi !== "N/A"
      ? `<div class="source-doi"><a href="${esc(s.doi)}" target="_blank" rel="noopener">${esc(s.doi)}</a></div>` : "";
    return `<div class="source-card">
      <div class="source-title">${esc(s.title)}</div>
      <div class="source-meta">${yearChip}${jChip}${ptChip}</div>
      ${authLine}${doiLine}
    </div>`;
  }).join("") + "</div>";
}

// ── mode selector ──────────────────────────────────────────────────────────
function setMode(mode){
  if(activeMode===mode) return;
  activeMode=mode; selectedId=null; searchText="";
  document.getElementById("searchBox").value="";

  document.getElementById("btnModeArch").classList.toggle("active", mode==="arch");
  document.getElementById("btnModeLim").classList.toggle("active",  mode==="limitations");
  document.getElementById("btnModeFw").classList.toggle("active",   mode==="future_work");

  document.getElementById("netPlaceholder").style.display="none";
  document.getElementById("mynetwork").style.display="block";

  const meta=MODE_META[mode];
  document.getElementById("netTitle").textContent=meta.title;
  document.getElementById("rightListLabel").textContent=meta.rightLabel;
  document.getElementById("netLegend").innerHTML=
    `<span class="legend-item"><span class="leg-dot" style="background:#f97316"></span>Task</span>`+
    `<span class="legend-item"><span class="leg-dot" style="background:${meta.legendColor}"></span>${meta.legendLabel}</span>`;

  rebuildNetwork();
  renderPanelLists();
  resetDetail();
}

// ── container height ───────────────────────────────────────────────────────
function ensureContainerHeight(){
  const el=document.getElementById("mynetwork");
  if(el.style.display==="none") return;
  const heroH   =(document.querySelector(".hero")      ||{getBoundingClientRect:()=>({height:52})}).getBoundingClientRect().height;
  const modeH   =(document.querySelector(".mode-bar")  ||{getBoundingClientRect:()=>({height:44})}).getBoundingClientRect().height;
  const toolbarH=(document.querySelector(".net-toolbar")||{getBoundingClientRect:()=>({height:38})}).getBoundingClientRect().height;
  const h=window.innerHeight-heroH-modeH-toolbarH;
  el.style.height=Math.max(h,280)+"px";
  el.style.flex="none";
}

// ── full network rebuild ───────────────────────────────────────────────────
function rebuildNetwork(){
  if(!activeMode) return;
  const {visNodes,visEdges}=getVisibleState();
  const container=document.getElementById("mynetwork");
  ensureContainerHeight();

  if(network){ network.destroy(); network=null; }

  nodesDS=new vis.DataSet(visNodes.map(n=>{ const c=Object.assign({},n); c.title=buildTooltip(n); return c; }));
  edgesDS=new vis.DataSet(visEdges.map(e=>Object.assign({},e)));

  const isDark=document.documentElement.getAttribute("data-theme")!=="light";
  const nodeFontColor=isDark?"#ffffff":"#0f172a";

  network=new vis.Network(container,{nodes:nodesDS,edges:edgesDS},{
    autoResize:true,
    physics:{enabled:false},
    layout:{randomSeed:42,improvedLayout:true},
    interaction:{hover:true,tooltipDelay:120,navigationButtons:true},
    nodes:{font:{face:"DM Sans, sans-serif", color:nodeFontColor}},
    edges:{selectionWidth:3,hoverWidth:2,smooth:{enabled:true,type:"dynamic"}},
  });

  network.fit();

  setTimeout(()=>{
    if(!network) return;
    network.setOptions({physics:{enabled:true,stabilization:{enabled:true,iterations:200,fit:true},barnesHut:{gravitationalConstant:-8000,centralGravity:0.25,springLength:180,springConstant:0.04,damping:0.2,avoidOverlap:0.3}}});
    network.once("stabilizationIterationsDone",()=>{
      network.setOptions({physics:{enabled:false}});
      network.fit({animation:{duration:500,easingFunction:"easeInOutQuad"}});
    });
  },100);

  network.on("click",params=>{
    selectedId=(params.nodes&&params.nodes.length)
      ? (selectedId===params.nodes[0]?null:params.nodes[0])
      : null;
    updateNodeStyles();
    renderPanelLists();
    if(selectedId) showDetail(selectedId); else resetDetail();
  });

  updateSummary(visNodes,visEdges);
}

// ── get visible state ──────────────────────────────────────────────────────
function getVisibleState(){
  if(!activeMode) return {visNodes:[],visEdges:[]};
  const allNodes=GRAPH_DATA[activeMode].nodes;
  const allEdges=GRAPH_DATA[activeMode].edges;
  const term=searchText.trim().toLowerCase();

  if(!term){
    return {visNodes:allNodes.map(n=>Object.assign({},n)), visEdges:allEdges.map(e=>Object.assign({},e))};
  }

  const direct=new Set();
  for(const n of allNodes){
    const blob=[n.label,n.primary_task,n.secondary_tasks,n.task_description,
                n.example_journal,n.architecture_name,n.architecture_family,
                n.input_type,n.pretrained_backbone,n.training_supervision,
                n.full_text, (n.sources||[]).map(s=>s.title+" "+s.journal+" "+s.authors).join(" "),
               ].join(" ").toLowerCase();
    if(blob.includes(term)) direct.add(n.id);
  }

  const visNodeIds=new Set(direct);
  const visEdgeIds=new Set();
  for(const e of allEdges){
    if(direct.has(e.from)||direct.has(e.to)){
      visNodeIds.add(e.from); visNodeIds.add(e.to); visEdgeIds.add(e.id);
    }
  }

  return {
    visNodes:allNodes.filter(n=>visNodeIds.has(n.id)).map(n=>Object.assign({},n)),
    visEdges:allEdges.filter(e=>visEdgeIds.has(e.id)).map(e=>Object.assign({},e)),
    directIds:direct,
  };
}

function buildTooltip(n){
  if(n.node_type==="task")
    return "<b>"+esc(n.primary_task)+"</b> (task)<br><b>Desc:</b> "+esc((n.task_description||"").slice(0,120));
  if(n.node_type==="architecture")
    return "<b>"+esc(n.architecture_name)+"</b><br><b>Family:</b> "+esc(n.architecture_family)+"<br><b>Input:</b> "+esc(n.input_type)+"<br><b>Backbone:</b> "+esc(n.pretrained_backbone);
  const src=(n.sources&&n.sources[0])?n.sources[0].title:"";
  const short=(n.full_text||"").slice(0,160)+((n.full_text||"").length>160?"…":"");
  return (src?"<b>"+esc(src)+"</b><br>":"")+esc(short);
}

// ── style-only update (no DataSet rebuild) ────────────────────────────────
function updateNodeStyles(){
  if(!nodesDS||!edgesDS||!activeMode) return;
  const allNodes=GRAPH_DATA[activeMode].nodes;
  const allEdges=GRAPH_DATA[activeMode].edges;
  const term=searchText.trim().toLowerCase();

  let directIds=null;
  if(term){
    directIds=new Set();
    for(const n of allNodes){
      if(nodesDS.get(n.id)===null) continue;
      const blob=[n.label,n.primary_task,n.secondary_tasks,n.task_description,
                  n.architecture_name,n.architecture_family,n.full_text,
                  (n.sources||[]).map(s=>s.title).join(" ")].join(" ").toLowerCase();
      if(blob.includes(term)) directIds.add(n.id);
    }
  }

  nodesDS.update(allNodes.filter(n=>nodesDS.get(n.id)!==null).map(n=>{
    const isSelected=n.id===selectedId;
    const isDirect=directIds?directIds.has(n.id):true;
    if(isSelected){
      const bg=n.node_type==="task"?"#fb923c":n.node_type==="architecture"?"#60a5fa":n.node_type==="limitation"?"#c084fc":"#34d399";
      return {id:n.id,borderWidth:5,size:Math.round((n.size||14)*1.4),color:{background:bg,border:"#fff",highlight:{background:bg,border:"#fff"}},opacity:1};
    }
    if(!isDirect) return {id:n.id,borderWidth:n.borderWidth||2,size:n.size||14,color:n.color,opacity:0.3};
    return {id:n.id,borderWidth:n.borderWidth||2,size:n.size||14,color:n.color,opacity:1};
  }));

  edgesDS.update(allEdges.filter(e=>edgesDS.get(e.id)!==null).map(e=>{
    const connected=(e.from===selectedId||e.to===selectedId);
    if(connected) return {id:e.id,color:{color:"#f97316",highlight:"#fb923c",hover:"#fb923c",opacity:1},width:3};
    return {id:e.id,color:e.color,width:e.width||1.5};
  }));
}

// ── panel lists ────────────────────────────────────────────────────────────
function renderPanelLists(){
  if(!activeMode){
    ["taskListCount","rightListCount"].forEach(id=>document.getElementById(id).textContent="0");
    ["taskList","rightList"].forEach(id=>{document.getElementById(id).innerHTML='<div class="empty-note">No graph selected.</div>'});
    return;
  }
  const {visNodes}=getVisibleState();
  const visIds=new Set(visNodes.map(n=>n.id));
  const rightType=MODE_META[activeMode].rightType;
  const rightColor=MODE_META[activeMode].legendColor;
  const allNodes=GRAPH_DATA[activeMode].nodes;

  const visTasks =allNodes.filter(n=>n.node_type==="task"    &&visIds.has(n.id));
  const visRight =allNodes.filter(n=>n.node_type===rightType &&visIds.has(n.id));

  document.getElementById("taskListCount").textContent =visTasks.length;
  document.getElementById("rightListCount").textContent=visRight.length;
  renderList("taskList",  visTasks,  "#f97316");
  renderList("rightList", visRight,  rightColor);
}

function renderList(elId,items,color){
  const el=document.getElementById(elId);
  if(!items.length){ el.innerHTML='<div class="empty-note">No items.</div>'; return; }
  el.innerHTML=items.map(n=>{
    const isSel=n.id===selectedId;
    const label=esc(n.label||n.primary_task||String(n.id));
    return `<button class="node-btn${isSel?" selected":""}" onclick="clickNodeBtn(${n.id})"><span class="type-dot" style="background:${color}"></span>${label}</button>`;
  }).join("");
}

function clickNodeBtn(id){
  selectedId=(selectedId===id)?null:id;
  updateNodeStyles(); renderPanelLists();
  if(selectedId) showDetail(selectedId); else resetDetail();
  if(network&&selectedId) network.focus(selectedId,{scale:1.2,animation:{duration:400,easingFunction:"easeInOutQuad"}});
}

// ── detail panel ───────────────────────────────────────────────────────────
function showDetail(id){
  if(!activeMode) return;
  const n=nodeMaps[activeMode].get(id);
  if(!n) return;
  const el=document.getElementById("detailPanel");

  if(n.node_type==="task"){
    el.innerHTML=
      `<div class="detail-title">${esc(n.primary_task)} <span class="tag tag-task">task</span></div>`+
      `<div class="detail-section-label">Task description</div><div class="detail-value">${esc(n.task_description)}</div>`+
      `<div class="detail-section-label">Secondary tasks</div><div class="detail-value">${esc(n.secondary_tasks)}</div>`+
      `<div class="detail-section-label">Example journal</div><div class="detail-value">${esc(n.example_journal)}</div>`;
    return;
  }

  if(n.node_type==="architecture"){
    el.innerHTML=
      `<div class="detail-title">${esc(n.architecture_name)} <span class="tag tag-arch">architecture</span></div>`+
      `<div class="detail-section-label">Family</div><div class="detail-value">${esc(n.architecture_family)}</div>`+
      `<div class="detail-section-label">Input type</div><div class="detail-value">${esc(n.input_type)}</div>`+
      `<div class="detail-section-label">Temporal modelling</div><div class="detail-value">${esc(n.temporal_modelling)}</div>`+
      `<div class="detail-section-label">Pretrained backbone</div><div class="detail-value">${esc(n.pretrained_backbone)}</div>`+
      `<div class="detail-section-label">Training supervision</div><div class="detail-value">${esc(n.training_supervision)}</div>`+
      `<div class="detail-section-label">Best metric</div><div class="detail-value">${esc(n.best_metric)}</div>`+
      `<div class="detail-section-label">Evaluation metrics</div><div class="detail-value">${esc(n.metrics)}</div>`+
      `<div class="detail-section-label">Validation design</div><div class="detail-value">${esc(n.validation_design)}</div>`+
      `<div class="detail-section-label">Comparators</div><div class="detail-value">${esc(n.comparators)}</div>`+
      `<div class="detail-section-label">Source article${(n.sources||[]).length!==1?"s":""} (${(n.sources||[]).length})</div>`+
      renderSourceCards(n.sources);
    return;
  }

  if(n.node_type==="limitation"){
    const srcCount=(n.sources||[]).length;
    el.innerHTML=
      `<div class="detail-title">Limitation <span class="tag tag-lim">limitation</span></div>`+
      `<div class="detail-section-label">Full limitation text</div><div class="detail-value">${esc(n.full_text)}</div>`+
      `<div class="detail-section-label">Reported in ${srcCount} article${srcCount!==1?"s":""}</div>`+
      renderSourceCards(n.sources);
    return;
  }

  if(n.node_type==="future_work"){
    const srcCount=(n.sources||[]).length;
    el.innerHTML=
      `<div class="detail-title">Future Work <span class="tag tag-fw">future work</span></div>`+
      `<div class="detail-section-label">Full future work text</div><div class="detail-value">${esc(n.full_text)}</div>`+
      `<div class="detail-section-label">Proposed in ${srcCount} article${srcCount!==1?"s":""}</div>`+
      renderSourceCards(n.sources);
    return;
  }
}

function resetDetail(){
  document.getElementById("detailPanel").textContent="Click any node in the graph or lists above to inspect its details.";
}

function updateSummary(visNodes,visEdges){
  if(!activeMode) return;
  const rightType=MODE_META[activeMode].rightType;
  const tCount=visNodes.filter(n=>n.node_type==="task").length;
  const rCount=visNodes.filter(n=>n.node_type===rightType).length;
  document.getElementById("netSummary").textContent=
    `${tCount} task${tCount!==1?"s":""}, ${rCount} ${rightType.replace("_"," ")} node${rCount!==1?"s":""}, ${visEdges.length} edge${visEdges.length!==1?"s":""}`;
}

// ── controls ───────────────────────────────────────────────────────────────
document.getElementById("searchBox").addEventListener("input",e=>{
  searchText=e.target.value; selectedId=null;
  if(activeMode){ rebuildNetwork(); renderPanelLists(); }
  resetDetail();
});
document.getElementById("btnClearSel").addEventListener("click",()=>{
  selectedId=null; updateNodeStyles(); renderPanelLists(); resetDetail();
  if(network) network.unselectAll();
});
document.getElementById("btnShowAll").addEventListener("click",()=>{
  selectedId=null; searchText=""; document.getElementById("searchBox").value="";
  if(activeMode){ rebuildNetwork(); renderPanelLists(); }
  resetDetail();
  if(network){ network.unselectAll(); network.fit({animation:{duration:400}}); }
});

function toggleTheme(){
  const html=document.documentElement;
  const next=html.getAttribute("data-theme")==="dark"?"light":"dark";
  html.setAttribute("data-theme",next);
  document.getElementById("themeBtn").innerHTML=next==="light"?"<span>\uD83C\uDF19</span> Dark mode":"<span>\u2600\uFE0F</span> Light mode";
  if(network) network.setOptions({nodes:{font:{color:next==="light"?"#0f172a":"#ffffff"}}});
  document.getElementById("mynetwork").style.background=getComputedStyle(document.documentElement).getPropertyValue("--net-bg").trim();
}

window.addEventListener("resize",()=>{
  if(activeMode) ensureContainerHeight();
  if(network) network.fit();
});

requestAnimationFrame(()=>{
  document.getElementById("statusBar").textContent=
    `${STATS.total_records} paper(s) \u00B7 ${STATS.task_count} tasks \u00B7 ${STATS.architecture_count} arch \u00B7 ${STATS.limitation_count} limits \u00B7 ${STATS.future_work_count} future`;
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# generate HTML
# ---------------------------------------------------------------------------

def generate_html(graph_data: dict) -> str:
    stats = graph_data["stats"]
    payload = {
        "arch":        graph_data["arch"],
        "limitations": graph_data["limitations"],
        "future_work": graph_data["future_work"],
    }
    html = HTML_TEMPLATE
    html = html.replace("__TASK_COUNT__",  str(stats["task_count"]))
    html = html.replace("__ARCH_COUNT__",  str(stats["architecture_count"]))
    html = html.replace("__LIM_COUNT__",   str(stats["limitation_count"]))
    html = html.replace("__FW_COUNT__",    str(stats["future_work_count"]))
    html = html.replace("__GRAPH_DATA__",  json.dumps(payload, ensure_ascii=False))
    html = html.replace("__STATS__",       json.dumps(stats,   ensure_ascii=False))
    return html


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multi-mode Task network HTML.")
    parser.add_argument("--input",  "-i", default="/data/Deep_Angiography/AngioVision/slr/results/stage2_results.jsonl")
    parser.add_argument("--output", "-o", default="/data/Deep_Angiography/AngioVision/slr/results/task_network.html")
    args = parser.parse_args()

    ip, op = Path(args.input), Path(args.output)

    print(f"[1/4] Reading  {ip}")
    records = load_records(str(ip))
    print(f"      {len(records)} record(s) loaded")

    print("[2/4] Building graph data ...")
    gd = build_graph(records)
    s  = gd["stats"]
    print(f"      tasks={s['task_count']} | arch={s['architecture_count']}({s['arch_edges']}e) | "
          f"lim={s['limitation_count']}({s['lim_edges']}e) | fw={s['future_work_count']}({s['future_edges']}e)")

    print("[3/4] Rendering HTML ...")
    html = generate_html(gd)

    print(f"[4/4] Writing  {op}")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(html, encoding="utf-8")
    print(f"\n  Done!  Open: {op}")


if __name__ == "__main__":
    main()