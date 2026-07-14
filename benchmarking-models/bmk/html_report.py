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
  * F1 heat-shade (red .40 -> green .90) on values in [0,1]; best-in-row ringed.

Self-contained, no external assets. Kept inside the eval suite so the CSV->HTML
transform ships with the pipeline:  python3 -m bmk.html_report in.csv out.html
"""

from __future__ import annotations

import csv
import html
import json
import sys


def _rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r]


def render(csv_path: str, html_path: str, title: str = "Leaderboard",
           heat_from: int = 0, freeze_cols=None, filter_cols=None,
           weight_col=None, qchart_cols=None) -> str:
    header, rows = _rows(csv_path)
    payload = json.dumps({
        "header": header, "rows": rows, "heatFrom": heat_from,
        "freeze": sorted(freeze_cols or []), "filters": list(filter_cols or []),
        "weightCol": weight_col, "qchart": list(qchart_cols) if qchart_cols else None,
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
  .brow{ display:grid; grid-template-columns:minmax(140px,300px) 1fr auto; gap:10px;
    align-items:center; padding:3px 0; }
  .blabel{ font-size:12px; color:var(--ink); overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; }
  .btrack{ background:var(--chip); border-radius:6px; height:16px; overflow:hidden; }
  .bfill{ height:100%; border-radius:6px; min-width:2px; }
  .bval{ font-size:12px; color:var(--mut); font-variant-numeric:tabular-nums;
    white-space:nowrap; }
</style>
</head>
<body>
<header>
  <h1>{{TITLE}}</h1>
  <div class="sub">Select a checkpoint / question group above · frozen left columns stay put on scroll · click a header to sort · “Columns” to hide/show · bottom row = weighted average of the shown rows</div>
</header>
<div class="filters" id="filters"></div>
<div class="bar">
  <input id="q" type="search" placeholder="filter rows…">
  <button id="toggleCols">Columns ▾</button>
  <button id="showAll">Show all</button>
  <button id="theme">Theme</button>
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
const {header, rows, heatFrom, freeze, filters, weightCol, qchart} = DATA;
const FROZEN = new Set(freeze);
const GROUP_COLORS = {OPACIFICATION:"#0d8f97",LOCATION:"#3b5bdb",PATHOLOGY:"#c0392b",
  DEVICE:"#c98a1b",ACCESS:"#7048e8",OTHER:"#5a6b7b"};
const isNum = header.map((_,c)=> rows.length>0 &&
  rows.every(r=> r[c]===""||r[c]==null|| !isNaN(parseFloat(r[c]))));
const LABEL_COL = Math.max(0, header.findIndex(h=>h.toLowerCase()==="question"));
let hidden = new Set(), sortCol=null, sortDir=-1;
const pick = {};  // col -> selected value ("" = All)
filters.forEach(c=> pick[c]="");

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
function docFor(h){
  if(COLDOC[h]!==undefined) return COLDOC[h];
  return "Per-question macro-F1 for model \""+h+"\": it answered this yes/no "+
    "question from the sequence mosaic (zero-shot image–text similarity, or a "+
    "generative VLM). Directly comparable, cell-to-cell, with the probe columns.";
}
const _tip = () => document.getElementById('tip');
function showTip(el){
  const t=_tip(); t.textContent=docFor(header[+el.dataset.c]); t.style.display='block';
  const r=el.getBoundingClientRect();
  t.style.left=Math.max(6, Math.min(r.left, window.innerWidth-t.offsetWidth-12))+'px';
  const below=r.bottom+6;
  t.style.top=(below+t.offsetHeight>window.innerHeight? r.top-t.offsetHeight-6: below)+'px';
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
  for(let c=heatFrom;c<header.length;c++){ if(isNum[c]&&!hidden.has(c)&&r[c]!==""){
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
    if(hidden.has(c)) return "";
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
  document.getElementById('cols').innerHTML = header.map((h,c)=>
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
    return "<tr>"+header.map((_,c)=> hidden.has(c)? "" :
      `<td class="${classOf(c)}" data-c="${c}">${fmtCell(r[c],c,rf)}</td>`).join("")+"</tr>";
  }).join("");
  drawFoot(view);
  applyFreeze();
}
function drawFoot(view){
  const foot=document.getElementById('foot');
  if(weightCol==null){ foot.innerHTML=""; return; }
  const cells = header.map((h,c)=>{
    if(hidden.has(c)) return "";
    if(c===weightCol){  // n column: total sample count
      const s = view.reduce((a,r)=> a + (parseFloat(r[c])||0), 0);
      return `<td class="${classOf(c)}" data-c="${c}">${s}</td>`;
    }
    if(!isNum[c] || c<heatFrom){  // text cols: label goes in the question column
      const lbl = c===LABEL_COL ? "weighted average" : "";
      return `<td class="${classOf(c)}" data-c="${c}">${lbl}</td>`;
    }
    let num=0, den=0;                // weighted mean over shown rows, weight = n
    for(const r of view){
      const v=parseFloat(r[c]), w=parseFloat(r[weightCol]);
      if(!isNaN(v) && !isNaN(w)){ num+=v*w; den+=w; }
    }
    if(den===0) return `<td class="${classOf(c)}" data-c="${c}">·</td>`;
    const val=num/den, disp=val.toFixed(3);
    const bg = shadeable(val)? heat(val):"transparent";
    return `<td class="${classOf(c)}" data-c="${c}"><span class="cell" style="background:${bg}">${disp}</span></td>`;
  }).join("");
  foot.innerHTML = `<tr>${cells}</tr>`;
}
function applyFreeze(){
  // cumulative left offset for visible frozen columns, in original order
  const vis = [...FROZEN].filter(c=>!hidden.has(c)).sort((a,b)=>a-b);
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
  document.getElementById('chartBody').innerHTML = data.map(d=>{
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

document.getElementById('q').oninput=drawBody;
document.getElementById('toggleCols').onclick=()=>document.getElementById('cols').classList.toggle('open');
document.getElementById('showAll').onclick=()=>{hidden.clear(); drawCols(); draw();};
document.getElementById('theme').onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  document.documentElement.setAttribute('data-theme', cur==='light'?'dark':'light');
};
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
</script>
</body></html>
"""


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m bmk.html_report <in.csv> <out.html>")
    render(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
