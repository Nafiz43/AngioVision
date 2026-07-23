"""CSV -> self-contained HTML leaderboard.

render(csv_path, html_path, title, heat_from, freeze_cols, filter_cols,
       weight_col, qchart_cols):
  * top **filter chips** for each column in `filter_cols` (single-select, "All"
    or one value) — e.g. checkpoint and question-group selectors,
  * **frozen columns** (`freeze_cols`): sticky to the left, stay visible on
    horizontal scroll (checkpoint / group / question / n),
  * Google-Sheets-style **collapsible columns** (hide/show any column),
  * a pinned **weighted-average** footer row (weighted by `weight_col`, the n
    column) recomputed over whatever rows the current filters select,
  * a **Question type** button opening an overlay bar chart of question counts,
    toggleable "By question" / "By group" (needs `qchart_cols=[group,q,n]`),
  * click-to-sort headers, a row search box,
  * F1 heat-shade (red .40 -> green .90) on values in [0,1]; best-in-row ringed,
  * an **All / UCD / Non-UCD** toggle (top right, next to the theme button):
    for any column pair `<col>_UCD` / `<col>_NONUCD` alongside a bare `<col>`,
    shows only the selected institution's twin and hides the other two —
    columns with no institution twins are unaffected. No-op (toggle hidden)
    if the CSV has no institution-split columns.
  * a **Bottom row** toggle (Weighted avg / Pooled TP/TN/FP/FN): the pooled
    mode sums confusion counts over the shown rows then derives macro-F1 from
    the totals, rather than averaging per-row F1. Needs "_cm:<col>:TP|TN|FP|FN"
    sidecar columns (never rendered as their own <th>/<td>) alongside a bare
    `<col>` — columns without a sidecar fall back to the weighted average so
    the footer is never blank.

Self-contained, no external assets. Kept inside the eval suite so the CSV->HTML
transform ships with the pipeline:  python3 -m bmk.html_report in.csv out.html
"""

from __future__ import annotations

import csv
import html
import json
import os
import sys


def _rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r]


def render(csv_path: str, html_path: str, title: str = "Leaderboard",
           heat_from: int = 0, freeze_cols=None, filter_cols=None,
           weight_col=None, qchart_cols=None,
           titles_csv=None, title_key_cols=None) -> str:
    header, rows = _rows(csv_path)
    # Optional per-cell tooltips (e.g. TP/TN/FP/FN behind each F1). A parallel
    # CSV with the SAME header/row order; each non-empty cell becomes the
    # `title` attribute on the matching data cell. Rows are matched by the
    # values in `title_key_cols` (default the checkpoint + question columns),
    # so sorting/filtering in the browser keeps tooltips aligned.
    titles = None
    key_cols = list(title_key_cols) if title_key_cols else [0, 2]
    if titles_csv and os.path.exists(titles_csv):
        t_header, t_rows = _rows(titles_csv)
        titles = {}
        for tr in t_rows:
            k = "␟".join(tr[i] for i in key_cols if i < len(tr))
            titles[k] = tr
    # "_cm:<col>:TP|TN|FP|FN" sidecar columns carry per-row confusion counts for
    # the footer's pooled-aggregate mode; they never get a <th>/<td> of their
    # own. Split them out into a {visibleColIdx: {tp,tn,fp,fn: sidecarColIdx}}
    # map (indices into the ORIGINAL header/row arrays, which are still sent
    # whole - simplest to keep row-array indices stable end to end).
    cm = {}
    for i, h in enumerate(header):
        if h.startswith("_cm:"):
            _, col, kind = h.split(":", 2)
            cm.setdefault(col, {})[kind.lower()] = i
    cm_by_visible_idx = {}
    for i, h in enumerate(header):
        if h in cm and set(cm[h]) == {"tp", "tn", "fp", "fn"}:
            cm_by_visible_idx[i] = cm[h]
    payload = json.dumps({
        "header": header, "rows": rows, "heatFrom": heat_from,
        "freeze": sorted(freeze_cols or []), "filters": list(filter_cols or []),
        "weightCol": weight_col, "qchart": list(qchart_cols) if qchart_cols else None,
        "titles": titles, "titleKeyCols": key_cols, "cm": cm_by_visible_idx,
    })
    doc = _TEMPLATE.replace("/*DATAJSON*/", payload).replace(
        "{{TITLE}}", html.escape(title))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return html_path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root{
    --bg:#0b0f14; --panel:#121821; --line:#243040; --ink:#e6edf3; --mut:#8aa0b4;
    --accent:#39c2c9; --chip:#1b2430;
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f6f8fa; --panel:#fff; --line:#dfe6ee; --ink:#0f1720; --mut:#5a6b7b;
           --accent:#0d8f97; --chip:#eef2f6; }
  }
  :root[data-theme="dark"]{ --bg:#0b0f14; --panel:#121821; --line:#243040; --ink:#e6edf3;
    --mut:#8aa0b4; --accent:#39c2c9; --chip:#1b2430; }
  :root[data-theme="light"]{ --bg:#f6f8fa; --panel:#fff; --line:#dfe6ee; --ink:#0f1720;
    --mut:#5a6b7b; --accent:#0d8f97; --chip:#eef2f6; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header{ padding:16px 22px 8px; }
  h1{ margin:0 0 3px; font-size:19px; letter-spacing:.2px; }
  .sub{ color:var(--mut); font-size:12.5px; }
  .filters{ padding:6px 22px 4px; display:flex; flex-direction:column; gap:6px; }
  .fgroup{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
  .flabel{ color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.5px;
    min-width:104px; }
  .pill{ background:var(--chip); border:1px solid var(--line); color:var(--ink);
    border-radius:999px; padding:4px 12px; cursor:pointer; font-size:12.5px; }
  .pill:hover{ border-color:var(--accent); }
  .pill.on{ background:var(--accent); border-color:var(--accent); color:#03181a; font-weight:700; }
  #instseg{ margin-left:auto; display:flex; gap:4px; align-items:center; }
  .bar{ display:flex; flex-wrap:wrap; gap:10px; align-items:center;
    padding:8px 22px; border-bottom:1px solid var(--line); position:sticky; top:0;
    background:var(--bg); z-index:20; }
  input[type=search]{ background:var(--panel); border:1px solid var(--line);
    color:var(--ink); border-radius:8px; padding:7px 11px; min-width:200px; font-size:13px; }
  button{ background:var(--chip); border:1px solid var(--line); color:var(--ink);
    border-radius:8px; padding:7px 11px; cursor:pointer; font-size:12.5px; }
  button:hover{ border-color:var(--accent); }
  #qtype{ margin-left:auto; font-weight:700; }
  .count{ color:var(--mut); font-size:12.5px; }
  #cols{ display:none; flex-wrap:wrap; gap:6px 14px; padding:10px 22px;
    border-bottom:1px solid var(--line); background:var(--panel); }
  #cols.open{ display:flex; }
  #cols label{ display:inline-flex; gap:6px; align-items:center; font-size:12.5px;
    color:var(--mut); cursor:pointer; white-space:nowrap; }
  .wrap{ overflow:auto; max-height:calc(100vh - 190px); }
  table{ border-collapse:separate; border-spacing:0; width:100%; }
  th,td{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:right;
    white-space:nowrap; font-variant-numeric:tabular-nums; background:var(--bg); }
  th{ position:sticky; top:0; background:var(--panel); cursor:pointer; z-index:10;
    font-size:11px; letter-spacing:.4px; text-transform:uppercase; color:var(--mut);
    user-select:none; }
  th:hover{ color:var(--accent); }
  td.txt,th.txt{ text-align:left; white-space:normal; }
  .frozen{ position:sticky; z-index:11; background:var(--panel); }
  th.frozen{ z-index:12; }
  .frozen.edge{ box-shadow:2px 0 0 var(--line); }
  tr:hover td:not(.frozen){ background:rgba(127,127,127,.05); }
  tbody td.frozen{ background:var(--bg); }
  tbody tr:hover td.frozen{ background:var(--panel); }
  tfoot td{ position:sticky; bottom:0; background:var(--panel); border-top:2px solid var(--accent);
    font-weight:700; z-index:9; }
  tfoot td.frozen{ z-index:13; }
  .cell{ display:inline-block; min-width:44px; padding:2px 7px; border-radius:6px; font-weight:600; }
  .best{ box-shadow:0 0 0 2px var(--accent) inset; border-radius:6px; }
  .chip2{ display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px;
    font-weight:700; letter-spacing:.3px; }
  td.q{ max-width:340px; white-space:normal; }
  .qm{ font-size:9px; color:var(--mut); margin-left:3px; vertical-align:super; opacity:.7; }
  #tip{ position:fixed; z-index:200; display:none; max-width:380px; background:var(--panel);
    color:var(--ink); border:1px solid var(--accent); border-radius:8px; padding:9px 11px;
    font:12px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; text-transform:none;
    letter-spacing:normal; white-space:pre-line; box-shadow:0 8px 30px rgba(0,0,0,.35);
    pointer-events:none; }
  /* question-distribution overlay */
  .overlay{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.55); z-index:100;
    align-items:center; justify-content:center; padding:24px; }
  .overlay.open{ display:flex; }
  .modal{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
    width:min(920px,96vw); max-height:88vh; display:flex; flex-direction:column;
    box-shadow:0 20px 60px rgba(0,0,0,.4); }
  .mhead{ display:flex; align-items:center; gap:12px; padding:14px 18px;
    border-bottom:1px solid var(--line); }
  .mhead strong{ font-size:15px; }
  .mmodes{ display:flex; gap:6px; }
  .mhead button{ margin-left:auto; }
  .chartbody{ overflow:auto; padding:12px 18px 18px; }
  .brow{ display:grid; grid-template-columns:minmax(240px,420px) 1fr auto; gap:10px;
    align-items:center; padding:3px 0; }
  .blabel{ font-size:12px; color:var(--ink); overflow-wrap:anywhere; }
  .clegend{ display:flex; flex-wrap:wrap; gap:12px; padding:0 0 12px; margin-bottom:10px;
    border-bottom:1px solid var(--line); }
  .clegend .li{ display:flex; align-items:center; gap:6px; font-size:11.5px; color:var(--mut);
    text-transform:uppercase; letter-spacing:.4px; }
  .clegend .sw{ width:12px; height:12px; border-radius:3px; flex:none; }
  .btrack{ background:var(--chip); border-radius:6px; height:16px; overflow:hidden; }
  .bfill{ height:100%; border-radius:6px; min-width:2px; }
  .bval{ font-size:12px; color:var(--mut); font-variant-numeric:tabular-nums;
    white-space:nowrap; }
</style>
</head>
<body>
<header>
  <h1>{{TITLE}}</h1>
  <div class="sub">Select a checkpoint / question group above · frozen left columns stay put on scroll · click a header to sort · “Columns” to hide/show · bottom row = weighted average or pooled TP/TN/FP/FN over the shown rows (toggle in the bar)</div>
</header>
<div class="filters" id="filters"></div>
<div class="bar">
  <input id="q" type="search" placeholder="filter rows…">
  <button id="toggleCols">Columns ▾</button>
  <button id="showAll">Show all</button>
  <span class="fgroup" id="footModeGroup">
    <span class="flabel">Bottom row</span>
    <span class="pill footmode on" data-m="avg">Weighted avg</span>
    <span class="pill footmode" data-m="pooled">Pooled TP/TN/FP/FN</span>
  </span>
  <span id="instseg">
    <span class="pill instp on" data-inst="ALL">All</span>
    <span class="pill instp" data-inst="UCD">UCD</span>
    <span class="pill instp" data-inst="NONUCD">Non-UCD</span>
  </span>
  <button id="theme" title="Switch between light and dark mode">☾ Dark</button>
  <button id="qtype">Question type ▤</button>
  <span class="count" id="count"></span>
</div>
<div id="cols"></div>
<div class="wrap"><table>
  <thead><tr id="head"></tr></thead>
  <tbody id="body"></tbody>
  <tfoot id="foot"></tfoot>
</table></div>

<div class="overlay" id="overlay"><div class="modal">
  <div class="mhead">
    <strong>Question distribution</strong>
    <span class="mmodes">
      <span class="pill mmode on" data-m="question">By question</span>
      <span class="pill mmode" data-m="group">By group</span>
    </span>
    <button id="closeOverlay">✕ Close</button>
  </div>
  <div class="chartbody" id="chartBody"></div>
</div></div>
<div id="tip"></div>

<script>
const DATA = /*DATAJSON*/;
const {header, rows, heatFrom, freeze, filters, weightCol, qchart, titles, titleKeyCols, cm} = DATA;
function cellTitle(r,c){
  if(!titles) return "";
  const k = (titleKeyCols||[0,2]).map(i=>r[i]).join("␟");
  const t = titles[k];
  return (t && t[c]) ? t[c] : "";
}
const FROZEN = new Set(freeze);
const GROUP_COLORS = {OPACIFICATION:"#0d8f97",LOCATION:"#3b5bdb",PATHOLOGY:"#c0392b",
  DEVICE:"#c98a1b",ACCESS:"#7048e8",OTHER:"#5a6b7b"};
// "_cm:<col>:KIND" sidecar columns hold per-row confusion counts for the
// pooled footer mode - never a real table column, always excluded from
// display (headers/cells/column-toggle/sort/search/heat).
const CM_COL = new Set(header.map((h,c)=>h.startsWith("_cm:")?c:-1).filter(c=>c>=0));
const isNum = header.map((_,c)=> rows.length>0 &&
  rows.every(r=> r[c]===""||r[c]==null|| !isNaN(parseFloat(r[c]))));
const LABEL_COL = Math.max(0, header.findIndex(h=>h.toLowerCase()==="question"));
let hidden = new Set([...CM_COL]), sortCol=null, sortDir=-1, footMode='avg', footCM={};
const pick = {};  // col -> selected value ("" = All)
filters.forEach(c=> pick[c]="");

// ── institution split: any <col>_UCD / <col>_NONUCD pair alongside a bare
// <col> becomes a 3-way variant group; instVariant[c] tags which one a
// column is ('ALL'|'UCD'|'NONUCD'), instBase[c] the shared name. Columns
// with no split twins are untouched by the toggle (instVariant[c] undefined).
const instVariant = {}, instBase = {};
header.forEach((h,c)=>{
  if(h.endsWith('_UCD') && header.includes(h.slice(0,-4))){
    instVariant[c]='UCD'; instBase[c]=h.slice(0,-4);
  } else if(h.endsWith('_NONUCD') && header.includes(h.slice(0,-7))){
    instVariant[c]='NONUCD'; instBase[c]=h.slice(0,-7);
  }
});
header.forEach((h,c)=>{
  if(instVariant[c]) return;
  if(header.includes(h+'_UCD') || header.includes(h+'_NONUCD')){
    instVariant[c]='ALL'; instBase[c]=h;
  }
});
const HAS_INST_SPLIT = Object.keys(instVariant).length>0;
let instMode='ALL';  // 'ALL' | 'UCD' | 'NONUCD'
function instVisible(c){ return !instVariant[c] || instVariant[c]===instMode; }
function effHidden(c){ return hidden.has(c) || !instVisible(c); }
function activeWeightCol(){
  if(weightCol==null || instMode==='ALL') return weightCol;
  const cand = header.indexOf(header[weightCol]+'_'+instMode);
  return cand>=0 ? cand : weightCol;
}

function esc(s){ return String(s).replace(/[&<>"]/g,m=>(
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }

// ── column documentation (hover a header to read what the column means) ──────
const PROBE_METHOD =
  "Frozen-feature probe — a logistic-regression classifier (StandardScaler + "+
  "LogisticRegression, max_iter=5000) trained on the FROZEN embeddings of the "+
  "fine-tuned model (the tower's weights are never updated). Scored out-of-fold "+
  "with GroupKFold(5) grouped by accession, so no study appears in both train "+
  "and test. Cell = macro-F1 over this question's rows.";
const COLDOC = {
  "checkpoint":"Fine-tuned model + epoch whose frozen embeddings were probed "+
    "(best epoch chosen by validation QA readout).",
  "group":"Coarse question category, keyword-classified: LOCATION, DEVICE, "+
    "PATHOLOGY, ACCESS, OPACIFICATION, OTHER.",
  "question":"The binary (yes/no) clinical question asked about the sequence.",
  "n":"Number of validation rows (image–question pairs) for this question.",
  "yes_rate":"Fraction of this question's rows whose ground-truth answer is "+
    "'yes' (its class balance).",
  "no_probe":"NO-PROBE readout — the fine-tuned checkpoint answering yes/no ON "+
    "ITS OWN, with NO probe and NO LR trained. For each question it encodes a "+
    "'yes' and a 'no' hypothesis as text and picks whichever is more "+
    "cosine-similar to the image embedding (this is what checkpoint selection "+
    "scores). Per-question F1 over the SAME rows as the probe columns. Contrast "+
    "it with F1_BASELINE: same frozen embedding, but F1_BASELINE adds a trained "+
    "linear head — the gap is what a probe recovers that the model's own "+
    "readout cannot.",
  "F1_BASELINE":PROBE_METHOD+"\n\nFeatures: [ image_final ; question ; "+
    "image_final × question ]. The default probe — one image view plus a single "+
    "elementwise image×question interaction.",
  "F1_A1_pre":PROBE_METHOD+"\n\nApproach A1 — features: the PRE-projection image "+
    "embedding only (before the contrastive projection head). Tests whether the "+
    "earlier-layer representation carries more usable signal.",
  "F1_A2_attn":PROBE_METHOD+"\n\nApproach A2 — features: [ image_attn ; question ; "+
    "image_attn × question ], using the ATTENTION-pooled image view in place of "+
    "the final embedding.",
  "F1_A3_xmodal":PROBE_METHOD+"\n\nApproach A3 — features: [ image_final ; question ; "+
    "image_final × question ; |image_final − question| ; cos(image_final, question) ]. "+
    "Adds an explicit cross-modal distance and cosine-similarity term.",
  "F1_A2+A3":PROBE_METHOD+"\n\nApproach A2+A3 — features: [ image_attn ; question ; "+
    "image_attn × question ; |image_attn − question| ]. Attention view plus the "+
    "cross-modal difference term.",
  "F1_B3margins":PROBE_METHOD+"\n\nApproach B3+margins (round-4 winner) — features: "+
    "[ image_final ; image_attn ; question ; final×question ; attn×question ; "+
    "|final−question| ; |attn−question| ; cos(final,question) ; cos(attn,question) ; "+
    "zero-shot yes/no similarity margin (final & attn view, raw and per-family "+
    "debiased) ]. Only computed for checkpoints with captured yes/no hypothesis "+
    "embeddings — blank elsewhere.",
  "best_A":"Which A-approach (A1_pre / A2_attn / A3_xmodal / A2+A3) scored the "+
    "highest macro-F1 on this question.",
  "best_A_F1":"The macro-F1 of that best A-approach for this question (the max "+
    "across the A columns).",
  "delta_vs_base":"best_A_F1 − F1_BASELINE. How much the best A-approach beats "+
    "the default baseline probe on this question (positive = an A-approach helps).",
  "ALL-YES":"Constant control: always predict 'yes'. Per-question macro-F1 "+
    "computed from the ground truth.",
  "ALL-NO":"Constant control: always predict 'no'. Per-question macro-F1 from "+
    "the ground truth.",
  "RANDOM":"Constant control: a seeded coin-flip prediction. Per-question macro-F1.",
  "model":"The model or constant control being scored.",
  "accuracy":"Overall accuracy across all of the model's rows.",
  "F1_macro":"Macro-averaged F1 across the yes/no classes, over all rows.",
  "F1_yes":"F1 for the positive ('yes') class.",
  "F1_no":"F1 for the negative ('no') class.",
  "F1_flipped":"Macro-F1 if the predictions were inverted (yes↔no). If this beats "+
    "F1_macro, the model's labels are inverted — use the flipped labels downstream.",
  "TP":"True positives — predicted yes, ground truth yes.",
  "TN":"True negatives — predicted no, ground truth no.",
  "FP":"False positives — predicted yes, ground truth no.",
  "FN":"False negatives — predicted no, ground truth yes.",
};
function baseDocFor(h){
  if(COLDOC[h]!==undefined) return COLDOC[h];
  return "Per-question macro-F1 for model \""+h+"\": it answered this yes/no "+
    "question from the sequence mosaic (zero-shot image–text similarity, or a "+
    "generative VLM). Directly comparable, cell-to-cell, with the probe columns.";
}
function docFor(h){
  if(COLDOC[h]!==undefined) return COLDOC[h];
  for(const [suf,scope] of [['_UCD','UCD institution rows only'],['_NONUCD','non-UCD institution rows only']]){
    if(h.endsWith(suf)){
      return baseDocFor(h.slice(0,-suf.length)) + "\n\nRestricted to "+scope+".";
    }
  }
  return baseDocFor(h);
}
const _tip = () => document.getElementById('tip');
function positionTip(el){
  const t=_tip(); t.style.display='block';
  const r=el.getBoundingClientRect();
  t.style.left=Math.max(6, Math.min(r.left, window.innerWidth-t.offsetWidth-12))+'px';
  const below=r.bottom+6;
  t.style.top=(below+t.offsetHeight>window.innerHeight? r.top-t.offsetHeight-6: below)+'px';
}
function showFootTip(el){
  const c=+el.dataset.c, r=footCM[c]; if(!r) return;
  const f1y=(2*r.tp)/((2*r.tp+r.fp+r.fn)||1), f1n=(2*r.tn)/((2*r.tn+r.fn+r.fp)||1);
  _tip().textContent = `Pooled over shown rows — ${header[c]}\n`+
    `TP=${r.tp}   TN=${r.tn}\nFP=${r.fp}   FN=${r.fn}\n`+
    `F1(yes)=${f1y.toFixed(3)}   F1(no)=${f1n.toFixed(3)}   macro=${r.val.toFixed(3)}`;
  positionTip(el);
}
function showTip(el){
  _tip().textContent=docFor(header[+el.dataset.c]);
  positionTip(el);
}
function hideTip(){ _tip().style.display='none'; }
function shadeable(v){ return !isNaN(v) && v>=0 && v<=1; }
function heat(v){
  if(!shadeable(v)) return "transparent";
  const t=Math.max(0,Math.min(1,(v-0.40)/0.50));
  return `hsla(${t*130},62%,45%,.30)`;
}
function fmtCell(val,c,rowFloats){
  if(GROUP_COLORS[val]!==undefined && header[c].toLowerCase()==="group"){
    return `<span class="chip2" style="background:${GROUP_COLORS[val]}22;color:${GROUP_COLORS[val]}">${val}</span>`;
  }
  if(c>=heatFrom && isNum[c] && val!==""){
    const v=parseFloat(val);
    if(!shadeable(v)) return val;
    const best = rowFloats.length>1 && v===Math.max(...rowFloats);
    return `<span class="cell ${best?'best':''}" style="background:${heat(v)}">${val}</span>`;
  }
  return val===""? '<span style="color:var(--mut)">·</span>' : val;
}
function rowFloatsOf(r){
  const out=[];
  for(let c=heatFrom;c<header.length;c++){ if(isNum[c]&&!effHidden(c)&&r[c]!==""){
    const v=parseFloat(r[c]); if(shadeable(v)) out.push(v);} }
  return out;
}
function classOf(c){
  let cl = isNum[c] ? "" : "txt";
  if(header[c].toLowerCase()==="question") cl += " q";
  if(FROZEN.has(c)) cl += " frozen";
  return cl.trim();
}
function drawFilters(){
  const el=document.getElementById('filters'); el.innerHTML="";
  filters.forEach(c=>{
    const vals=[...new Set(rows.map(r=>r[c]))];
    const chips=['<span class="flabel">'+header[c]+'</span>',
      `<span class="pill ${pick[c]===''?'on':''}" data-c="${c}" data-v="">All</span>`]
      .concat(vals.map(v=>`<span class="pill ${pick[c]===v?'on':''}" data-c="${c}" data-v="${esc(v)}">${esc(v)}</span>`));
    const g=document.createElement('div'); g.className="fgroup"; g.innerHTML=chips.join("");
    el.appendChild(g);
  });
  el.querySelectorAll('.pill').forEach(p=>p.onclick=()=>{
    pick[+p.dataset.c]=p.dataset.v;
    drawFilters(); drawBody();
  });
}
function drawHead(){
  document.getElementById('head').innerHTML = header.map((h,c)=>{
    if(effHidden(c)) return "";
    const arrow = sortCol===c ? (sortDir<0?" ▾":" ▴") : "";
    return `<th class="${classOf(c)}" data-c="${c}">${esc(h)}<span class="qm">ⓘ</span>${arrow}</th>`;
  }).join("");
  document.querySelectorAll('#head th').forEach(th=>{
    th.onclick=()=>{
      const c=+th.dataset.c; if(sortCol===c) sortDir*=-1; else {sortCol=c; sortDir=-1;}
      draw();
    };
    th.onmouseenter=()=>showTip(th);
    th.onmouseleave=hideTip;
  });
}
function drawCols(){
  document.getElementById('cols').innerHTML = header.map((h,c)=> CM_COL.has(c) ? "" :
    `<label><input type="checkbox" data-c="${c}" ${hidden.has(c)?"":"checked"}>${esc(h)}</label>`
  ).join("");
  document.querySelectorAll('#cols input').forEach(cb=>cb.onchange=()=>{
    const c=+cb.dataset.c; if(cb.checked) hidden.delete(c); else hidden.add(c);
    draw();
  });
}
function currentView(){
  const q=document.getElementById('q').value.trim().toLowerCase();
  return rows.filter(r=>{
    for(const c of filters){ if(pick[c]!=="" && r[c]!==pick[c]) return false; }
    return !q || r.some(v=> String(v).toLowerCase().includes(q));
  });
}
function drawBody(){
  let view = currentView();
  if(sortCol!=null){
    view = view.slice().sort((a,b)=>{
      let x=a[sortCol], y=b[sortCol];
      if(isNum[sortCol]){ x=parseFloat(x); y=parseFloat(y);
        x=isNaN(x)?-Infinity:x; y=isNaN(y)?-Infinity:y; return (x-y)*sortDir; }
      return String(x).localeCompare(String(y))*sortDir;
    });
  }
  document.getElementById('count').textContent = `${view.length} / ${rows.length} rows`;
  document.getElementById('body').innerHTML = view.map(r=>{
    const rf = rowFloatsOf(r);
    return "<tr>"+header.map((_,c)=>{ if(effHidden(c)) return "";
      const tt = cellTitle(r,c);
      return `<td class="${classOf(c)}" data-c="${c}"${tt?` data-tip="${esc(tt)}"`:""}>${fmtCell(r[c],c,rf)}</td>`;
    }).join("")+"</tr>";
  }).join("");
  drawFoot(view);
  applyFreeze();
}
function weightedAvgCell(c, view){
  const wc = activeWeightCol();    // n / n_UCD / n_NONUCD depending on the toggle
  let num=0, den=0;                // weighted mean over shown rows, weight = n
  for(const r of view){
    const v=parseFloat(r[c]), w=parseFloat(r[wc]);
    if(!isNaN(v) && !isNaN(w)){ num+=v*w; den+=w; }
  }
  return den===0 ? null : num/den;
}
function pooledCell(c, view){
  // Pooled aggregate: sum TP/TN/FP/FN over the shown rows for this column's
  // approach/model, THEN derive macro-F1 from the totals - not an average of
  // per-row F1 scores. Requires confusion sidecar columns (cm[c]); columns
  // without sidecars (e.g. institution-split twins, or raw predictions no
  // longer on disk) fall back to the weighted average, so the footer is
  // never blank.
  const idx = cm[c];
  if(!idx){ const val=weightedAvgCell(c, view); return val==null? null : {val, cm:false}; }
  let tp=0, tn=0, fp=0, fn=0, any=false;
  for(const r of view){
    const t=parseFloat(r[idx.tp]), n=parseFloat(r[idx.tn]),
          p=parseFloat(r[idx.fp]), g=parseFloat(r[idx.fn]);
    if(!isNaN(t)&&!isNaN(n)&&!isNaN(p)&&!isNaN(g)){ tp+=t; tn+=n; fp+=p; fn+=g; any=true; }
  }
  if(!any) return null;
  const f1yes = (2*tp)/((2*tp+fp+fn)||1), f1no = (2*tn)/((2*tn+fn+fp)||1);
  return {val:(f1yes+f1no)/2, cm:true, tp, tn, fp, fn};
}
function drawFoot(view){
  const foot=document.getElementById('foot');
  footCM = {};
  if(weightCol==null){ foot.innerHTML=""; return; }
  const wc = activeWeightCol();  // n / n_UCD / n_NONUCD depending on the toggle
  const cells = header.map((h,c)=>{
    if(effHidden(c)) return "";
    if(c===wc){  // n column: total sample count
      const s = view.reduce((a,r)=> a + (parseFloat(r[c])||0), 0);
      return `<td class="${classOf(c)}" data-c="${c}">${s}</td>`;
    }
    if(!isNum[c] || c<heatFrom){  // text cols: label goes in the question column
      const lbl = c===LABEL_COL ? (footMode==='pooled'?"pooled (TP/TN/FP/FN)":"weighted average") : "";
      return `<td class="${classOf(c)}" data-c="${c}">${lbl}</td>`;
    }
    const res = footMode==='pooled' ? pooledCell(c, view) : {val: weightedAvgCell(c, view)};
    if(res==null || res.val==null) return `<td class="${classOf(c)}" data-c="${c}">·</td>`;
    const disp=res.val.toFixed(3);
    const bg = shadeable(res.val)? heat(res.val):"transparent";
    let attr = "";
    if(footMode==='pooled'){
      if(res.cm){ footCM[c]=res; attr=' data-foothover="1"'; }
      else attr=' title="no confusion counts for this column - showing weighted average"';
    }
    return `<td class="${classOf(c)}" data-c="${c}"${attr}><span class="cell" style="background:${bg}">${disp}</span></td>`;
  }).join("");
  foot.innerHTML = `<tr>${cells}</tr>`;
  foot.querySelectorAll('td[data-foothover]').forEach(td=>{
    td.onmouseenter=()=>showFootTip(td);
    td.onmouseleave=hideTip;
  });
}
function applyFreeze(){
  // cumulative left offset for visible frozen columns, in original order
  const vis = [...FROZEN].filter(c=>!effHidden(c)).sort((a,b)=>a-b);
  let left=0;
  vis.forEach((c,i)=>{
    const th=document.querySelector(`#head th[data-c="${c}"]`);
    const w=th?th.getBoundingClientRect().width:0;
    document.querySelectorAll(`[data-c="${c}"]`).forEach(cell=>{
      if(cell.tagName==="TD"||cell.tagName==="TH"){
        cell.style.left=left+"px";
        cell.classList.toggle('edge', i===vis.length-1);
      }
    });
    left+=w;
  });
}
function draw(){ drawHead(); drawBody(); }

/* ── question-distribution overlay ─────────────────────────────────────── */
function qchartData(){
  const [gC,qC,nC]=qchart;
  const seen=new Map();                 // question -> {group, n}  (dedup across checkpoints)
  for(const r of rows){ const q=r[qC];
    if(!seen.has(q)) seen.set(q,{group:r[gC], n:parseFloat(r[nC])||0}); }
  const byQ=[...seen.values()].map((o,i)=>({label:[...seen.keys()][i], group:o.group, count:o.n}));
  const byQs=[...seen.entries()].map(([q,o])=>({label:q, group:o.group, count:o.n}))
    .sort((a,b)=>b.count-a.count);
  const gmap=new Map();
  for(const d of byQs){ const g=gmap.get(d.group)||{q:0,n:0}; g.q++; g.n+=d.count; gmap.set(d.group,g); }
  const byG=[...gmap.entries()].map(([group,o])=>({label:group, group, qcount:o.q, count:o.n}))
    .sort((a,b)=>b.qcount-a.qcount);
  return {byQ:byQs, byG};
}
function drawChart(mode){
  const {byQ,byG}=qchartData();
  const data = mode==='group'? byG : byQ;
  const maxv = mode==='group'
    ? Math.max(1,...data.map(d=>d.qcount)) : Math.max(1,...data.map(d=>d.count));
  const groups=[...new Set(data.map(d=>d.group))]
    .sort((a,b)=>(GROUP_COLORS[a]?0:1)-(GROUP_COLORS[b]?0:1));
  const legend='<div class="clegend">'+groups.map(g=>
    `<span class="li"><span class="sw" style="background:${GROUP_COLORS[g]||'#5a6b7b'}"></span>${esc(g)}</span>`
  ).join("")+'</div>';
  document.getElementById('chartBody').innerHTML = legend + data.map(d=>{
    const val = mode==='group'? d.qcount : d.count;
    const w = val/maxv*100;
    const color = GROUP_COLORS[d.group]||'#5a6b7b';
    const meta = mode==='group'
      ? `${d.qcount} questions · ${d.count} samples` : `n=${d.count}`;
    return `<div class="brow"><div class="blabel" title="${esc(d.label)}">${esc(d.label)}</div>`+
      `<div class="btrack"><div class="bfill" style="width:${w}%;background:${color}"></div></div>`+
      `<div class="bval">${meta}</div></div>`;
  }).join("");
}
let chartMode='question';
function openOverlay(){ drawChart(chartMode); document.getElementById('overlay').classList.add('open'); }
function closeOverlay(){ document.getElementById('overlay').classList.remove('open'); }

document.querySelectorAll('.footmode').forEach(p=>p.onclick=()=>{
  footMode=p.dataset.m;
  document.querySelectorAll('.footmode').forEach(x=>x.classList.toggle('on',x===p));
  drawBody();
});
document.getElementById('q').oninput=drawBody;
document.getElementById('toggleCols').onclick=()=>document.getElementById('cols').classList.toggle('open');
document.getElementById('showAll').onclick=()=>{hidden=new Set([...CM_COL]); drawCols(); draw();};
document.getElementById('instseg').style.display = HAS_INST_SPLIT ? 'flex' : 'none';
document.querySelectorAll('.instp').forEach(p=>p.onclick=()=>{
  instMode = p.dataset.inst;
  document.querySelectorAll('.instp').forEach(x=>x.classList.toggle('on', x===p));
  draw();
});
function effectiveTheme(){
  return document.documentElement.getAttribute('data-theme')
      || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
}
function applyTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('theme').textContent = t==='light' ? '☀ Light' : '☾ Dark';
}
document.getElementById('theme').onclick=()=>applyTheme(effectiveTheme()==='light'?'dark':'light');
applyTheme(effectiveTheme());   // pin to the OS-effective theme + label the button on load
const qtypeBtn=document.getElementById('qtype');
if(qchart){
  qtypeBtn.onclick=openOverlay;
  document.getElementById('closeOverlay').onclick=closeOverlay;
  document.getElementById('overlay').onclick=e=>{ if(e.target.id==='overlay') closeOverlay(); };
  document.querySelectorAll('.mmode').forEach(p=>p.onclick=()=>{
    chartMode=p.dataset.m;
    document.querySelectorAll('.mmode').forEach(x=>x.classList.toggle('on',x===p));
    drawChart(chartMode);
  });
} else { qtypeBtn.style.display='none'; }
window.addEventListener('resize', applyFreeze);
drawFilters(); drawCols(); draw();

// ── instant hover tooltip for per-cell confusion (TP/TN/FP/FN) ──────────────
(function(){
  const tip = document.createElement('div');
  tip.id = 'celltip';
  tip.style.cssText = "position:fixed;z-index:9999;pointer-events:none;display:none;"+
    "max-width:320px;padding:6px 9px;border-radius:6px;font:12px/1.4 ui-monospace,"+
    "SFMono-Regular,Menlo,monospace;background:#111a24;color:#e6edf3;"+
    "border:1px solid #39c2c9;box-shadow:0 4px 14px rgba(0,0,0,.4);white-space:nowrap;";
  document.body.appendChild(tip);
  function move(e){ tip.style.left=(e.clientX+14)+'px';
    tip.style.top=(e.clientY+16)+'px'; }
  document.addEventListener('mouseover', e=>{
    const td = e.target.closest && e.target.closest('td[data-tip]');
    if(!td){ tip.style.display='none'; return; }
    tip.textContent = td.getAttribute('data-tip');
    tip.style.display='block'; move(e);
  });
  document.addEventListener('mousemove', e=>{
    if(tip.style.display==='block') move(e);
  });
  document.addEventListener('mouseout', e=>{
    const td = e.target.closest && e.target.closest('td[data-tip]');
    if(td) tip.style.display='none';
  });
})();
</script>
</body></html>
"""


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m bmk.html_report <in.csv> <out.html>")
    render(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
