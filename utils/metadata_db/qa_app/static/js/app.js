// ════════════════════════════════════════════════
// LOGIN
// ════════════════════════════════════════════════
const CREDENTIALS = [
  { user: 'goldman',  pass: 'xK9#mQ2$vL5@pN8!' },
  { user: 'vfilkov',  pass: 'ChangeMe#001!'     },
];
const MAX_ATTEMPTS = 5;
let loginAttempts = 0, lockoutTimer = null;

function togglePwVisibility(){
  const inp=document.getElementById('loginPass');
  const open=document.getElementById('eyeOpen'),off=document.getElementById('eyeOff');
  if(inp.type==='password'){inp.type='text';open.style.display='none';off.style.display='block';}
  else{inp.type='password';open.style.display='block';off.style.display='none';}
}

function handleLogin(){
  const u=document.getElementById('loginUser').value.trim();
  const p=document.getElementById('loginPass').value;
  const btn=document.getElementById('loginBtn'),err=document.getElementById('loginError');
  const card=document.getElementById('loginCard'),fill=document.getElementById('attemptFill');
  if(btn.disabled) return;
  const match=CREDENTIALS.find(c=>c.user===u&&c.pass===p);
  if(match){
    err.classList.remove('visible');
    document.getElementById('sessionUser').textContent=u;
    document.getElementById('loginOverlay').classList.add('hidden');
    document.getElementById('appWrapper').classList.add('visible');
    loadStats(); loadEmbeddingModels();
    document.getElementById('nlInput').focus();
  } else {
    loginAttempts++;
    fill.style.width=Math.min(100,(loginAttempts/MAX_ATTEMPTS)*100)+'%';
    err.classList.remove('visible'); void err.offsetWidth; err.classList.add('visible');
    card.classList.remove('shake'); void card.offsetWidth; card.classList.add('shake');
    document.getElementById('loginPass').value='';
    document.getElementById('loginPass').focus();
    if(loginAttempts>=MAX_ATTEMPTS){
      btn.disabled=true; let secs=30;
      err.textContent=`Too many attempts. Try again in ${secs}s.`;
      lockoutTimer=setInterval(()=>{secs--;err.textContent=`Too many attempts. Try again in ${secs}s.`;
        if(secs<=0){clearInterval(lockoutTimer);loginAttempts=0;fill.style.width='0%';
          btn.disabled=false;err.textContent='Invalid username or password.';
          err.classList.remove('visible');document.getElementById('loginUser').focus();}},1000);
    }
  }
}
function handleLogout(){
  document.getElementById('appWrapper').classList.remove('visible');
  document.getElementById('loginOverlay').classList.remove('hidden');
  document.getElementById('loginUser').value='';
  document.getElementById('loginPass').value='';
  document.getElementById('loginError').classList.remove('visible');
  document.getElementById('sessionUser').textContent='—';
  document.getElementById('loginUser').focus();
}
document.addEventListener('DOMContentLoaded',()=>{
  ['loginUser','loginPass'].forEach(id=>{
    document.getElementById(id).addEventListener('keydown',e=>{if(e.key==='Enter')handleLogin();});
  });
  document.getElementById('loginUser').focus();
  setupDragDrop();
});

// ════════════════════════════════════════════════
// MAIN APP STATE
// ════════════════════════════════════════════════
const API='';
let chatHistory=[], isLoading=false;
let attachedImage=null;  // { dataUrl, base64, filename }
let _abortController = null;

const inp     = document.getElementById('nlInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');

function setLoading(on){
  isLoading = on;
  sendBtn.disabled = on;
  sendBtn.style.display = on ? 'none'  : 'flex';
  stopBtn.style.display = on ? 'flex'  : 'none';
}

function stopQuery(){
  if(_abortController) _abortController.abort();
}

// Auto-resize textarea
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,130)+'px';});
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSend();}});

function fillQ(q){inp.value=q;inp.dispatchEvent(new Event('input'));inp.focus();}

function promptImageUpload(q){
  inp.value=q;inp.dispatchEvent(new Event('input'));
  triggerImagePick();
}

// ════════════════════════════════════════════════
// IMAGE ATTACH
// ════════════════════════════════════════════════
function triggerImagePick(){
  document.getElementById('imageFileInput').click();
}

function onImageSelected(e){
  const file=e.target.files[0];
  if(!file) return;
  const reader=new FileReader();
  reader.onload=ev=>{
    const dataUrl=ev.target.result;
    attachedImage={dataUrl, base64: dataUrl.split(',')[1], filename: file.name};
    document.getElementById('imgPreviewThumb').src=dataUrl;
    document.getElementById('imgPreviewName').textContent=file.name;
    document.getElementById('imgPreviewBar').style.display='flex';
    document.getElementById('attachBtn').classList.add('active');
    document.getElementById('inputWrap').classList.add('img-mode');
    sendBtn.classList.add('img-mode');
    inp.placeholder='Ask about this image — e.g. "show me similar cases"…';
    inp.focus();
  };
  reader.readAsDataURL(file);
  e.target.value='';
}

function clearImage(){
  attachedImage=null;
  document.getElementById('imgPreviewBar').style.display='none';
  document.getElementById('imgPreviewThumb').src='';
  document.getElementById('attachBtn').classList.remove('active');
  document.getElementById('inputWrap').classList.remove('img-mode');
  sendBtn.classList.remove('img-mode');
  inp.placeholder='Ask about DICOM metadata, radiology reports, or upload an image…';
}

// Drag-and-drop onto the messages area
function setupDragDrop(){
  const msgs=document.getElementById('messages');
  msgs.addEventListener('dragover',e=>{e.preventDefault();msgs.classList.add('drag-over');});
  msgs.addEventListener('dragleave',()=>msgs.classList.remove('drag-over'));
  msgs.addEventListener('drop',e=>{
    e.preventDefault(); msgs.classList.remove('drag-over');
    const file=e.dataTransfer.files[0];
    if(file && file.type.startsWith('image/')){
      const fakeEvt={target:{files:[file]}};
      onImageSelected(fakeEvt);
    }
  });
}

// ════════════════════════════════════════════════
// THEME TOGGLE
// ════════════════════════════════════════════════
(function(){
  // Restore preference on load
  const saved = localStorage.getItem('av-theme');
  if(saved === 'light'){
    document.documentElement.setAttribute('data-theme','light');
    document.getElementById('iconMoon').style.display='none';
    document.getElementById('iconSun').style.display='block';
  }
})();

function toggleTheme(){
  const root   = document.documentElement;
  const isLight = root.getAttribute('data-theme') === 'light';
  const moon   = document.getElementById('iconMoon');
  const sun    = document.getElementById('iconSun');
  if(isLight){
    root.removeAttribute('data-theme');
    moon.style.display='block';
    sun.style.display='none';
    localStorage.setItem('av-theme','dark');
  } else {
    root.setAttribute('data-theme','light');
    moon.style.display='none';
    sun.style.display='block';
    localStorage.setItem('av-theme','light');
  }
}

// ════════════════════════════════════════════════
// MODEL / STATUS
// ════════════════════════════════════════════════
function onModelChange(){
  const m=document.getElementById('modelSelect').value;
  fetch(`${API}/api/model`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m})})
    .then(r=>r.json()).then(d=>{if(d.ok)setStatus('ready',m);});
}

// ════════════════════════════════════════════════
// IMAGE EMBEDDING MODEL (image RAG)
// ════════════════════════════════════════════════
let embedModels = [];   // [{key,label,hf_id,collection}]

function getSelectedEmbedModel(){
  const sel=document.getElementById('embedModelSelect');
  return (sel && sel.value) || localStorage.getItem('av-embed-model') || 'rad-dino';
}

function shortModelLabel(key){
  const m=embedModels.find(x=>x.key===key);
  return m ? m.label : key;
}

function updateEmbedModelUI(key){
  const m=embedModels.find(x=>x.key===key);
  const label=m?m.label:key;
  const hint=document.getElementById('embedModelHint');
  const hdr=document.getElementById('hdrEmbedModel');
  const prev=document.getElementById('imgPreviewModel');
  const chromaLbl=document.getElementById('chromaSecLabel');
  if(hint) hint.textContent=m?m.hf_id:'';
  if(hdr) hdr.textContent=label;
  if(prev) prev.textContent=label;
  if(chromaLbl) chromaLbl.textContent='ChromaDB · '+label;
}

async function loadEmbeddingModels(){
  const sel=document.getElementById('embedModelSelect');
  try{
    const r=await fetch(`${API}/api/embedding-models`);
    const d=await r.json();
    embedModels=d.models||[];
    const saved=localStorage.getItem('av-embed-model');
    const initial=embedModels.some(m=>m.key===saved)?saved:(d.default||'rad-dino');
    if(sel){
      sel.innerHTML=embedModels.map(m=>`<option value="${m.key}">${m.label}</option>`).join('');
      sel.value=initial;
    }
    localStorage.setItem('av-embed-model',initial);
    updateEmbedModelUI(initial);
  }catch(e){
    if(sel) sel.innerHTML='<option value="rad-dino">RAD-DINO (radiology)</option>';
  }
  loadChromaStats();
}

function onEmbedModelChange(){
  const key=getSelectedEmbedModel();
  localStorage.setItem('av-embed-model',key);
  updateEmbedModelUI(key);
  loadChromaStats();
}

function setStatus(state,label){
  const dot=document.querySelector('.status-dot'),txt=document.getElementById('statusTxt');
  const colors={ready:'#3fb950',error:'#f85149',busy:'#d29922'};
  dot.style.background=colors[state]||colors.ready;
  dot.style.boxShadow=`0 0 5px ${colors[state]||colors.ready}`;
  txt.textContent=label||state;
}

// ════════════════════════════════════════════════
// STATS
// ════════════════════════════════════════════════
async function loadStats(){
  try{
    const r=await fetch(`${API}/api/stats`);
    const d=await r.json();
    if(d.error){setStatus('error','db error');return;}
    document.getElementById('sInst').textContent=fmtNum(d.instances);
    document.getElementById('sPat').textContent=fmtNum(d.patients);
    document.getElementById('sStu').textContent=fmtNum(d.studies);
    document.getElementById('sSer').textContent=fmtNum(d.series);
    document.getElementById('dbPath').textContent=d.db_path||'—';
    document.getElementById('sRptTotal').textContent=fmtNum(d.rpt_total);
    document.getElementById('sRptLinked').textContent=fmtNum(d.rpt_linked);
    if(d.rpt_total>0&&d.studies>0){
      const pct=Math.min(100,Math.round(d.rpt_linked/d.studies*100));
      document.getElementById('rptCoverageWrap').style.display='block';
      document.getElementById('rptCoverageBar').style.width=pct+'%';
      document.getElementById('rptCoverageLabel').textContent=`${pct}% of studies linked  ·  ${fmtNum(d.rpt_unlinked)} missing`;
    }
    if(d.modalities&&d.modalities.length){
      const max=Math.max(...d.modalities.map(m=>m.count));
      document.getElementById('modalityBars').innerHTML=d.modalities.map(m=>`
        <div class="modality-row">
          <span style="min-width:28px;color:var(--accent)">${m.modality}</span>
          <div class="mod-bar-wrap"><div class="mod-bar" style="width:${Math.round(m.count/max*100)}%"></div></div>
          <span style="min-width:40px;text-align:right">${fmtNum(m.count)}</span>
        </div>`).join('');
      document.getElementById('modalitySec').style.display='block';
    }
    setStatus('ready','ready');
  }catch(e){setStatus('error','offline');}
}

async function loadChromaStats(){
  const model=getSelectedEmbedModel();
  const ingestHint=`Not ingested · run_ingest.py --images-only --embedding-model ${model}`;
  try{
    const r=await fetch(`${API}/api/chroma-stats?model=${encodeURIComponent(model)}`);
    const d=await r.json();
    if(d.model_label){
      const chromaLbl=document.getElementById('chromaSecLabel');
      if(chromaLbl) chromaLbl.textContent='ChromaDB · '+d.model_label;
    }
    if(d.available && d.count>0){
      document.getElementById('sChromaFrames').textContent=fmtNum(d.count);
      document.getElementById('sChromaSeqs').textContent=d.sequences!=null?fmtNum(d.sequences):'—';
      document.getElementById('chromaStatusLabel').textContent=`Collection: ${d.collection}`;
    } else {
      document.getElementById('sChromaFrames').textContent=d.available?'0':'—';
      document.getElementById('sChromaSeqs').textContent=d.available?'0':'—';
      document.getElementById('chromaStatusLabel').textContent=d.error?'Unavailable':ingestHint;
    }
  }catch(e){
    document.getElementById('chromaStatusLabel').textContent='Unavailable';
  }
}

function fmtNum(n){if(n===undefined||n===null||n==='—')return '—';return Number(n).toLocaleString();}

// ════════════════════════════════════════════════
// SQL SYNTAX HIGHLIGHT
// ════════════════════════════════════════════════
function colorSQL(sql){
  const kws=/\b(SELECT|FROM|WHERE|AND|OR|NOT|IN|LIKE|BETWEEN|ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|OFFSET|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|DISTINCT|COUNT|SUM|AVG|MAX|MIN|CAST|LOWER|UPPER|NULL|IS\s+NOT|IS|CASE|WHEN|THEN|ELSE|END|WITH|UNION|ALL|EXISTS|USING|SUBSTR|INSTR|TRIM|COALESCE|IFNULL)\b/gi;
  return sql.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/'[^']*'/g,s=>`<span class="str">${s}</span>`)
    .replace(/--[^\n]*/g,s=>`<span class="cmt">${s}</span>`)
    .replace(/\b(\d+\.?\d*)\b/g,s=>`<span class="nm">${s}</span>`)
    .replace(kws,s=>`<span class="kw">${s}</span>`);
}

// ════════════════════════════════════════════════
// TABLE RENDERER
// ════════════════════════════════════════════════
function renderTable(rows){
  if(!rows||!rows.length)return '<div style="padding:16px;font-size:12px;color:var(--text3);font-family:var(--mono)">No rows returned.</div>';
  const cols=Object.keys(rows[0]);
  const thead=`<thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead>`;
  const tbody=`<tbody>${rows.map(r=>`<tr>${cols.map(c=>{
    const v=r[c];
    if(v===null||v===undefined)return `<td><span class="v-null">null</span></td>`;
    const s=String(v);
    if(c==='radrpt'||c==='radrpt_excerpt')return `<td><span class="v-report" title="${esc(s)}">${esc(s.length>80?s.slice(0,78)+'…':s)}</span></td>`;
    if(s.toLowerCase().endsWith('.dcm')&&s.startsWith('/')){
      const shortPath=s.length>40?'…'+s.slice(-38):s;
      const onerr="this.outerHTML='<span class=\\'tbl-thumb-err\\'>no px</span>'";
      const fc=frameCountOf(r);
      const multi=fc&&fc>1;
      const idxs=multi?sampleFrameIndices(fc,5):[0];
      const stripHtml=idxs.map(fi=>{
        const thumbUrl=API+'/api/thumbnail?path='+encodeURIComponent(s)+'&frame='+fi;
        const frameUrl=API+'/api/frame?path='+encodeURIComponent(s)+'&frame='+fi;
        const lbl=multi?(esc(s)+' · Frame '+fi):esc(s);
        const tip=multi?('Frame '+fi+' / '+(fc-1)):'Frame 0';
        return `<img src="${thumbUrl}" class="tbl-thumb" loading="lazy" title="${tip}" alt="Frame ${fi}" onclick="openLightbox('${frameUrl}','${lbl}')" onerror="${onerr}"/>`;
      }).join('');
      return `<td><div class="dcm-cell"><div class="tbl-frame-strip">${stripHtml}</div><span class="v-path" title="${esc(s)}">${esc(shortPath)}</span></div></td>`;
    }
    if(s.startsWith('/'))return `<td><span class="v-path" title="${esc(s)}">${esc(s.length>50?'…'+s.slice(-48):s)}</span></td>`;
    if(/^\d{8}$/.test(s)&&parseInt(s)>19000101)return `<td><span class="v-num">${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}</span></td>`;
    if(s.length>32&&s.includes('.'))return `<td><span class="v-id" title="${esc(s)}">${esc(s.slice(0,28)+'…')}</span></td>`;
    if(!isNaN(s)&&s.trim()!=='')return `<td><span class="v-num">${esc(s)}</span></td>`;
    return `<td>${esc(s.length>60?s.slice(0,58)+'…':s)}</td>`;
  }).join('')}</tr>`).join('')}</tbody>`;
  return `<div class="tbl-scroll"><table class="data-table">${thead}${tbody}</table></div>`;
}

function renderSteps(steps){
  if(!steps||!steps.length)return '<div class="agent-steps"><div class="agent-steps-empty">No tool calls recorded — the agent may have failed before calling sql_query. Check the error message in the Answer tab.</div></div>';
  let html='<div class="agent-steps">';
  steps.forEach(s=>{
    const cls=s.error?'step-err':'step-ok';
    const badge=s.error?'✗ error':'✓ ok';
    html+=`<div class="agent-step-item ${cls}">`;
    html+=`<div class="step-hdr"><span class="step-badge">${badge}</span><span class="step-meta">Step ${s.step} / ${s.max_steps}</span></div>`;
    html+=`<div class="step-sql">${esc(s.sql)}</div>`;
    if(s.error){
      html+=`<div class="step-error-msg">${esc(s.error)}</div>`;
    } else {
      html+=`<div class="step-result">${s.row_count} row${s.row_count!==1?'s':''} returned</div>`;
    }
    html+=`</div>`;
  });
  html+='</div>';
  return html;
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

// Detect a sequence's frame count from a SQL result row (best-effort, column-name agnostic).
function frameCountOf(row){
  for(const k in row){
    const lk=k.toLowerCase().replace(/[^a-z]/g,'');
    if(lk==='framecount'||lk==='frames'||lk==='numberofframes'||lk==='numframes'||lk==='nframes'){
      const n=parseInt(row[k]); if(!isNaN(n)&&n>0) return n;
    }
  }
  return null;
}

// Up to `max` evenly-spaced, de-duplicated frame indices across [0, count-1].
function sampleFrameIndices(count, max){
  const n=Math.min(max,count), out=[];
  for(let i=0;i<n;i++) out.push(Math.round(i*(count-1)/(n-1||1)));
  return [...new Set(out)];
}

// ════════════════════════════════════════════════
// MESSAGE HELPERS
// ════════════════════════════════════════════════
// Build the in-queue banner text from a queued / queue_update event.
function formatQueueMsg(evt){
  const pos=evt.position||0;
  const total=evt.total_waiting||pos;
  const ahead=(evt.ahead!=null)?evt.ahead:Math.max(0,pos-1);
  if(ahead<=0) return `You're next in line — your request will start the moment the server frees up.`;
  return `You're in the queue — position ${pos} of ${total} (${ahead} ahead of you). Your request will start automatically when the server frees up.`;
}

function addMsg(cls,html){
  const es=document.getElementById('emptyState');
  if(es)es.remove();
  const msgs=document.getElementById('messages');
  const d=document.createElement('div');
  d.className=`msg ${cls}`;d.innerHTML=html;
  msgs.appendChild(d);
  if(document.getElementById('autoScrollToggle').checked)msgs.scrollTop=msgs.scrollHeight;
  return d;
}

function addHistory(q,count,elapsed,isImage=false){
  chatHistory.unshift({q,count,elapsed,ts:new Date().toLocaleTimeString(),isImage});
  if(chatHistory.length>30)chatHistory.pop();
  document.getElementById('histScroll').innerHTML=chatHistory.map(h=>`
    <div class="hist-item" onclick="fillQ(${JSON.stringify(h.q)})">
      <div class="hist-q${h.isImage?' img-q':''}">${h.isImage?'📷 ':''}${esc(h.q)}</div>
      <div class="hist-meta">${h.ts} · ${h.count} result${h.count!==1?'s':''} · ${(h.elapsed/1000).toFixed(1)}s</div>
    </div>`).join('');
}

// ════════════════════════════════════════════════
// MAIN SEND ROUTER
// ════════════════════════════════════════════════
async function handleSend(){
  if(isLoading) return;
  const q=inp.value.trim();
  if(attachedImage){
    await handleImageSend(q||'Show me the 5 most visually similar cases to this image.');
  } else {
    if(!q) return;
    await handleTextSend(q);
  }
}

// ════════════════════════════════════════════════
// TEXT QUERY (NL → SQL → synthesis)
// ════════════════════════════════════════════════
async function handleTextSend(question, opts){
  opts = opts || {};
  const skipClarify = !!opts.skipClarify;                 // re-submit after a clarification
  const displayQuestion = opts.displayQuestion || question; // shown in the user bubble
  const clarifyEl = document.getElementById('clarifyToggle');
  const clarify = clarifyEl ? clarifyEl.checked : true;
  _abortController = new AbortController();
  setLoading(true);
  inp.value=''; inp.style.height='auto';
  addMsg('msg-user',`<div class="bubble bubble-user">${esc(displayQuestion)}</div>`);
  const thinkEl=addMsg('msg-bot',`<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Initializing agent…</span></div>`);
  const thinkTxt=thinkEl.querySelector('#thinkTxt');
  setStatus('busy','thinking…');
  const think=document.getElementById('thinkToggle').checked;
  const model=document.getElementById('modelSelect').value;
  let sql='',rows=[],rowCount=0,answer='',elapsed=0;
  let agentSteps=[];   // ← NEW: track every tool call
  let allSql=[];       // ← NEW: track every SQL the agent tried
  let charts=[];       // ← NEW: chart specs emitted by render_chart
  try{
    const resp=await fetch(`${API}/api/query`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,think,model,clarify,skip_clarify:skipClarify}),signal:_abortController.signal});

    /* ── NEW: catch non-SSE error responses (503, 400, etc.) ── */
    if(!resp.ok){
      let errMsg=`Server returned HTTP ${resp.status}`;
      try{const errData=await resp.json();errMsg=errData.error||errMsg;}catch{}
      thinkEl.remove();
      addMsg('msg-bot',`<div class="bubble-error">Error: ${esc(errMsg)}</div>`);
      setLoading(false);setStatus('error','error');return;
    }

    const reader=resp.body.getReader(),decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        let evt;try{evt=JSON.parse(line.slice(6));}catch{continue;}
        switch(evt.event){

          /* ── NEW: concurrency queue events (sent before the pipeline starts) ── */
          case 'queued':
          case 'queue_update':
            thinkEl.classList.add('queued');
            thinkTxt.textContent=formatQueueMsg(evt);
            setStatus('busy',`queued #${evt.position}`);
            break;
          case 'slot_acquired':
            thinkEl.classList.remove('queued');
            thinkTxt.textContent='Starting…';
            setStatus('busy','starting…');
            break;
          case 'busy':
            thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-notice">${esc(evt.message||'Server is at capacity, please retry shortly.')}</div>`);
            setLoading(false);setStatus('ready','ready');return;

          /* ── NEW: agent lifecycle events ── */
          case 'agent_start':
            thinkTxt.textContent=`Agent started (${evt.model}, max ${evt.max_steps} steps)…`;
            break;
          case 'agent_step':
            agentSteps.push(evt);
            if(evt.error){
              thinkTxt.textContent=`Step ${evt.step}/${evt.max_steps}: SQL error → retrying…`;
            } else {
              thinkTxt.textContent=`Step ${evt.step}/${evt.max_steps}: ${evt.row_count} row${evt.row_count!==1?'s':''} returned`;
            }
            break;
          case 'chart':
            charts.push(evt.spec);
            thinkTxt.textContent='Rendering chart…';
            break;

          /* ── NEW: pre-flight clarification gate ── */
          case 'clarify_check':
            thinkTxt.textContent='Checking if clarification is needed…';
            break;
          case 'clarification':
            thinkEl.remove();
            renderClarification(evt, question);   // `question` is the original here
            setLoading(false); setStatus('ready','ready');
            return;

          /* ── Original events ── */
          case 'sql_start':    thinkTxt.textContent='Agent is thinking…';break;
          case 'sql_done':     sql=evt.sql;allSql.push(evt.sql);thinkTxt.textContent='Executing query…';break;
          case 'sql_repaired': sql=evt.sql;allSql.push(evt.sql);thinkTxt.textContent=`Refining SQL (attempt ${allSql.length})…`;break;
          case 'exec_start':   thinkTxt.textContent='Running query…';break;
          case 'exec_done':    rows=evt.rows;rowCount=evt.row_count;thinkTxt.textContent='Waiting for agent…';break;
          case 'synth_start':  thinkTxt.textContent='Synthesizing answer…';break;
          case 'answer':       answer=evt.text;break;
          case 'done':         elapsed=evt.elapsed_ms;break;
          case 'error':
            thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-error">Error: ${esc(evt.message)}</div>`);
            setLoading(false);setStatus('error','error');return;
        }
      }
    }
    thinkEl.remove();
    const id='rc'+Date.now(),elapsedSec=(elapsed/1000).toFixed(1);
    const hasChart=charts.length>0;
    let tabs='',panes='';
    tabs+=`<button class="rc-tab${hasChart?'':' active'}" onclick="switchRcTab(this,'${id}-ans')">Answer</button>`;
    panes+=`<div class="rc-pane${hasChart?'':' active'}" id="${id}-ans"><div class="answer-body">${esc(answer)}</div></div>`;
    if(hasChart){
      tabs+=`<button class="rc-tab active" onclick="switchRcTab(this,'${id}-chart')">Chart <span style="font-size:9px;opacity:.7">${charts.length}</span></button>`;
      panes+=`<div class="rc-pane active" id="${id}-chart"><div class="chart-wrap" id="${id}-charts"></div></div>`;
    }
    tabs+=`<button class="rc-tab" onclick="switchRcTab(this,'${id}-sql')">SQL</button>`;
    panes+=`<div class="rc-pane" id="${id}-sql"><div class="sql-block">${colorSQL(sql)}</div></div>`;
    tabs+=`<button class="rc-tab" onclick="switchRcTab(this,'${id}-tbl')">Table <span style="font-size:9px;opacity:.7">${rowCount}</span></button>`;
    panes+=`<div class="rc-pane" id="${id}-tbl">${renderTable(rows)}${rowCount?`<div class="tbl-footer"><span>${rowCount} row${rowCount!==1?'s':''}</span><button class="copy-btn" onclick="copySQL('${id}')">copy SQL</button><span style="margin-left:auto">${elapsedSec}s</span></div>`:''}</div>`;

    /* ── NEW: Steps tab showing every tool call the agent made ── */
    tabs+=`<button class="rc-tab" onclick="switchRcTab(this,'${id}-steps')">Steps <span style="font-size:9px;opacity:.7">${agentSteps.length}</span></button>`;
    panes+=`<div class="rc-pane" id="${id}-steps">${renderSteps(agentSteps)}</div>`;

    addMsg('msg-bot',`<div class="result-card"><div class="rc-tabs">${tabs}<div class="rc-meta"><span>${elapsedSec}s</span></div></div>${panes}</div>`);
    if(hasChart) renderCharts(`${id}-charts`, charts);
    addHistory(displayQuestion,rowCount,elapsed,false);
    setStatus('ready','ready');
  }catch(e){
    if(e.name==='AbortError'){
      addMsg('msg-bot','<div class="bubble-error">Query stopped.</div>');
      setStatus('ready','ready');
    } else {
      thinkEl.remove();
      addMsg('msg-bot',`<div class="bubble-error">Request failed: ${esc(String(e))}</div>`);
      setStatus('error','error');
    }
  }
  setLoading(false);inp.focus();
}

// ════════════════════════════════════════════════
// CLARIFICATION  (pre-flight ambiguity gate)
// Renders a focused question with selectable options + an "Other" free-text
// option, then re-submits the ORIGINAL question augmented with the answer so
// the agent keeps full context. skip_clarify guarantees only one round.
// ════════════════════════════════════════════════
function renderClarification(evt, baseQuestion){
  const cid='clar'+Date.now()+Math.floor(Math.random()*1000);
  const q=evt.question||'Could you clarify your request?';
  const opts=Array.isArray(evt.options)?evt.options:[];
  let optsHtml=opts.map(o=>
    `<button type="button" class="clarify-opt" data-val="${esc(o)}" onclick="chooseClarifyOption('${cid}',this,false)">${esc(o)}</button>`
  ).join('');
  optsHtml+=`<button type="button" class="clarify-opt clarify-other" onclick="chooseClarifyOption('${cid}',this,true)">Other (specify)…</button>`;
  const reason=evt.reason?`<div class="clarify-reason">${esc(evt.reason)}</div>`:'';
  const html=`
    <div class="clarify-card" id="${cid}" data-base="${esc(baseQuestion)}" data-q="${esc(q)}">
      <div class="clarify-hdr">
        <svg class="clarify-ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span>Need a bit more detail</span>
      </div>
      <div class="clarify-q">${esc(q)}</div>
      ${reason}
      <div class="clarify-opts">${optsHtml}</div>
      <div class="clarify-other-wrap" id="${cid}-otherwrap" style="display:none">
        <input type="text" class="clarify-other-input" id="${cid}-other" placeholder="Type your answer…"
               oninput="document.getElementById('${cid}-send').disabled=!this.value.trim();"
               onkeydown="if(event.key==='Enter'){event.preventDefault();submitClarification('${cid}');}">
      </div>
      <div class="clarify-actions">
        <span class="clarify-hint">Pick an option or choose “Other”.</span>
        <button class="clarify-send" id="${cid}-send" onclick="submitClarification('${cid}')" disabled>Send</button>
      </div>
    </div>`;
  addMsg('msg-bot', html);
}

function chooseClarifyOption(cid, btn, isOther){
  const card=document.getElementById(cid);
  if(!card || card.classList.contains('answered')) return;
  card.querySelectorAll('.clarify-opt').forEach(b=>b.classList.remove('selected'));
  btn.classList.add('selected');
  const otherWrap=document.getElementById(cid+'-otherwrap');
  const sendBtn=document.getElementById(cid+'-send');
  if(isOther){
    otherWrap.style.display='block';
    const inp=document.getElementById(cid+'-other');
    card.dataset.choice='__other__';
    if(sendBtn) sendBtn.disabled=!(inp && inp.value.trim());
    if(inp) inp.focus();
  } else {
    otherWrap.style.display='none';
    card.dataset.choice=btn.dataset.val||'';
    if(sendBtn) sendBtn.disabled=false;
  }
}

function submitClarification(cid){
  const card=document.getElementById(cid);
  if(!card || card.classList.contains('answered')) return;
  const base=card.dataset.base||'';
  const clarQ=card.dataset.q||'';
  let answer=card.dataset.choice||'';
  if(answer==='__other__' || !answer){
    const inp=document.getElementById(cid+'-other');
    answer=(inp && inp.value.trim())||'';
    if(!answer){ if(inp) inp.focus(); return; }
  }
  // Lock the card so it cannot be answered twice.
  card.classList.add('answered');
  card.querySelectorAll('.clarify-opt').forEach(b=>{
    b.disabled=true;
    if(b.dataset.val===answer) b.classList.add('selected');
  });
  const otherInp=document.getElementById(cid+'-other'); if(otherInp) otherInp.disabled=true;
  const sendBtn=document.getElementById(cid+'-send'); if(sendBtn){sendBtn.disabled=true;sendBtn.textContent='Sent';}
  // Re-issue the original question + the clarification as one augmented turn.
  const augmented=`${base}\n\n[Clarification]\nQ: ${clarQ}\nA: ${answer}`;
  handleTextSend(augmented, {skipClarify:true, displayQuestion:answer});
}

// ════════════════════════════════════════════════
// IMAGE QUERY (ChromaDB vector search with RAD-DINO)
// ════════════════════════════════════════════════
async function handleImageSend(question){
  if(!attachedImage) return;
  const img=attachedImage;
  clearImage();
  _abortController = new AbortController();
  setLoading(true);
  inp.value=''; inp.style.height='auto';

  addMsg('msg-user',`
    <div class="bubble-with-img">
      <img src="${img.dataUrl}" class="bubble-img-preview" alt="uploaded image">
      <span class="bubble-img-caption">${esc(question)}</span>
    </div>`);

  const thinkEl=addMsg('msg-bot',`<div class="thinking"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span id="thinkTxt">Decoding image…</span></div>`);
  const thinkTxt=thinkEl.querySelector('#thinkTxt');
  setStatus('busy','searching…');

  let results=[],answer='',elapsed=0;
  const cardId='irc'+Date.now();

  try{
    const resp=await fetch(`${API}/api/image-query`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal:_abortController.signal,
      body:JSON.stringify({
        image:    img.base64,
        question: question,
        n_results: 5,
        think:    document.getElementById('thinkToggle').checked,
        embedding_model: getSelectedEmbedModel(),
      })
    });
    const reader=resp.body.getReader(),decoder=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        let evt;try{evt=JSON.parse(line.slice(6));}catch{continue;}
        switch(evt.event){
          /* ── NEW: concurrency queue events (sent before the pipeline starts) ── */
          case 'queued':
          case 'queue_update':
            thinkEl.classList.add('queued');
            thinkTxt.textContent=formatQueueMsg(evt);
            setStatus('busy',`queued #${evt.position}`);
            break;
          case 'slot_acquired':
            thinkEl.classList.remove('queued');
            thinkTxt.textContent='Starting…';
            setStatus('busy','starting…');
            break;
          case 'busy':
            thinkEl && thinkEl.parentNode && thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-notice">${esc(evt.message||'Server is at capacity, please retry shortly.')}</div>`);
            setLoading(false);setStatus('ready','ready');return;
          case 'decode_start':  thinkTxt.textContent='Decoding image…';break;
          case 'decode_done':   thinkTxt.textContent=`Querying ChromaDB (${evt.width}×${evt.height} px)…`;break;
          case 'chroma_start':  thinkTxt.textContent=`Searching vector database (${shortModelLabel(getSelectedEmbedModel())})…`;break;
          case 'chroma_done':
            results=evt.results;
            thinkTxt.textContent=`Found ${evt.count} cases · enriching…`;
            thinkEl.remove();
            addMsg('msg-bot', renderSimilarCasesCard(results, '(loading…)', 0, cardId));
            break;
          case 'enrich_done':   results=evt.results;break;
          case 'answer':        answer=evt.text;break;
          case 'done':          elapsed=evt.elapsed_ms;break;
          case 'error':
            thinkEl && thinkEl.parentNode && thinkEl.remove();
            addMsg('msg-bot',`<div class="bubble-error">Image search error: ${esc(evt.message)}</div>`);
            setLoading(false);setStatus('error','error');return;
        }
      }
    }
    const ansPane=document.getElementById(`${cardId}-answer`);
    if(ansPane) ansPane.innerHTML=`<div class="answer-body">${esc(answer)}</div>`;
    const timeEl=document.querySelector(`#${cardId} .irc-time`);
    if(timeEl) timeEl.textContent=`${(elapsed/1000).toFixed(1)}s`;
    addHistory(`[Image] ${question}`, results.length, elapsed, true);
    setStatus('ready','ready');
    loadChromaStats();
  }catch(e){
    try{ thinkEl && thinkEl.parentNode && thinkEl.remove(); }catch(_){}
    if(e.name==='AbortError'){
      addMsg('msg-bot','<div class="bubble-error">Query stopped.</div>');
      setStatus('ready','ready');
    } else {
      addMsg('msg-bot',`<div class="bubble-error">Image search failed: ${esc(String(e))}</div>`);
      setStatus('error','error');
    }
  }
  setLoading(false);inp.focus();
}

// ════════════════════════════════════════════════
// SIMILAR CASES CARD RENDERER
// ════════════════════════════════════════════════
function simClass(pct){
  if(pct>=80) return 'high';
  if(pct>=60) return 'med';
  if(pct>=40) return 'low';
  return 'vlow';
}

function fmtDate(d){
  if(!d||String(d).length!==8) return d||'';
  return `${String(d).slice(0,4)}-${String(d).slice(4,6)}-${String(d).slice(6,8)}`;
}

function renderSimilarCasesCard(results, answer, elapsed, cardId){
  const elapsedSec=(elapsed/1000).toFixed(1);

  const casesHtml = results.length ? results.map(r=>{
    const pct     = r.similarity_pct||0, sc=simClass(pct);
    const sopGroups = r.sop_groups||[];
    const totalFrames = sopGroups.reduce((n,sg)=>n+sg.frames.length, 0);

    const chips=[
      r.study_date        ? `<span class="case-chip">${esc(fmtDate(r.study_date))}</span>` : '',
      r.modality          ? `<span class="case-chip mod">${esc(r.modality)}</span>` : '',
      r.series_description? `<span class="case-chip">${esc(String(r.series_description).slice(0,30))}</span>` : '',
      r.patient_sex       ? `<span class="case-chip">${esc(r.patient_sex)}</span>` : '',
      r.patient_age       ? `<span class="case-chip">${esc(r.patient_age)}</span>` : '',
      `<span class="frame-count-badge">${sopGroups.length} SOP · ${totalFrames} frame${totalFrames!==1?'s':''}</span>`,
    ].filter(Boolean).join('');

    const rptHtml = r.radrpt_excerpt
      ? `<div class="case-rpt">${esc(String(r.radrpt_excerpt).slice(0,220))}</div>` : '';

    // One block per SOP UID, each with its own frame strip
    const sopBlocksHtml = sopGroups.map(sg=>{
      const sopSim   = sg.similarity_pct||0;
      const sopSc    = simClass(sopSim);
      const frameStripHtml = sg.frames.map(f=>{
        const hasPath  = f.source_path && f.source_path !== '';
        const fi       = parseInt(f.frame_index)||0;
        const sim      = f.similarity_pct||0;
        const tip      = `SOP: ${sg.sop_uid}\nFrame: ${fi}  Sim: ${sim}%`;
        const lbMeta   = `${esc(r.accession_number||'?')} · Frame ${fi} · ${sim}%`;
        const thumbUrl = hasPath ? `/api/thumbnail?path=${encodeURIComponent(f.source_path)}&frame=${fi}` : '';
        const frameUrl = hasPath ? `/api/frame?path=${encodeURIComponent(f.source_path)}&frame=${fi}` : '';
        return hasPath
          ? `<img class="frame-thumb" src="${thumbUrl}" loading="lazy"
                  title="${tip}" alt="Frame ${fi}"
                  onclick="openLightbox('${frameUrl}','${lbMeta}')">`
          : `<div class="frame-thumb" style="display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text3)">?</div>`;
      }).join('');

      return `
        <div class="sop-group">
          <div class="sop-header">
            <span class="sop-uid" title="${esc(sg.sop_uid)}">${esc(sg.sop_uid||'—')}</span>
            <span class="sop-sim ${sopSc}">${sopSim}%</span>
          </div>
          <div class="frame-strip">${frameStripHtml}</div>
        </div>`;
    }).join('');

    return `
      <div class="case-item" style="flex-direction:column;align-items:stretch;gap:6px;">
        <div style="display:flex;align-items:center;gap:12px;">
          <div class="case-rank">${r.rank}</div>
          <div class="case-body">
            <div class="case-top">
              <span class="case-acc" title="${esc(r.accession_number||'')}">${esc(String(r.accession_number||'—').slice(0,32))}</span>
              <div class="sim-wrap">
                <div class="sim-bar"><div class="sim-fill ${sc}" style="width:${pct}%"></div></div>
                <span class="case-sim-pct ${sc}">${pct}%</span>
              </div>
            </div>
            ${chips?`<div class="case-chips">${chips}</div>`:''}
            ${rptHtml}
          </div>
        </div>
        ${sopBlocksHtml}
      </div>`;
  }).join('')
  : '<div class="no-results">No similar cases found in ChromaDB.</div>';

  return `
    <div class="img-result-card" id="${cardId}">
      <div class="irc-header">
        <div class="irc-icon">
          <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </div>
        <span class="irc-title">Similar Cases</span>
        <span class="irc-badge">RAD-DINO · ChromaDB</span>
        <span class="irc-time">${elapsedSec}s</span>
      </div>
      <div class="irc-tabs">
        <button class="irc-tab active" onclick="switchIrcTab(this,'${cardId}-cases')">
          Cases<span class="irc-count">${results.length}</span>
        </button>
        <button class="irc-tab" onclick="switchIrcTab(this,'${cardId}-answer')">Answer</button>
      </div>
      <div class="irc-pane active" id="${cardId}-cases">
        <div class="similar-cases">${casesHtml}</div>
      </div>
      <div class="irc-pane" id="${cardId}-answer">
        <div class="answer-body">${esc(answer)}</div>
      </div>
    </div>`;
}

// ════════════════════════════════════════════════
// LIGHTBOX
// ════════════════════════════════════════════════
function openLightbox(frameUrl, metaText){
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightboxImg');
  img.src = '';              // clear first so loading spinner fires
  img.src = frameUrl;
  document.getElementById('lightboxMeta').textContent = metaText || '';
  lb.classList.add('visible');
  document.body.style.overflow = 'hidden';
}

function closeLightbox(){
  document.getElementById('lightbox').classList.remove('visible');
  document.getElementById('lightboxImg').src = '';
  document.body.style.overflow = '';
}

document.addEventListener('keydown', e => {
  if(e.key === 'Escape') closeLightbox();
});

// ════════════════════════════════════════════════
// TAB SWITCHERS
// ════════════════════════════════════════════════
function switchRcTab(btn,paneId){
  const card=btn.closest('.result-card');
  card.querySelectorAll('.rc-tab').forEach(b=>b.classList.remove('active'));
  card.querySelectorAll('.rc-pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}

function switchIrcTab(btn,paneId){
  const card=btn.closest('.img-result-card');
  card.querySelectorAll('.irc-tab').forEach(b=>b.classList.remove('active'));
  card.querySelectorAll('.irc-pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}

function copySQL(id){
  const pane=document.getElementById(id+'-sql');
  const code=pane.querySelector('.sql-block');
  navigator.clipboard.writeText(code.textContent.trim()).then(()=>{
    const btn=document.querySelector(`[onclick="copySQL('${id}')"]`);
    if(btn){btn.textContent='copied!';setTimeout(()=>btn.textContent='copy SQL',1500);}
  });
}

// ════════════════════════════════════════════════
// CHART RENDERER  (self-contained <canvas>; no external libraries)
// Renders the spec emitted by the agent's render_chart tool.
// spec = { chart_type, title, labels:[], values:[], series_label }
// ════════════════════════════════════════════════
const CHART_PALETTE=['#58a6ff','#3fb950','#d29922','#a371f7','#f85149','#ec6cb9','#39c5cf','#7ee787','#ffa657','#ff7b72','#79c0ff','#d2a8ff'];

function cssVar(name,fallback){
  const v=getComputedStyle(document.documentElement).getPropertyValue(name);
  return (v&&v.trim())||fallback;
}
function chTrunc(s,n){s=String(s);return s.length>n?s.slice(0,n-1)+'…':s;}
function chFmt(v){
  if(!isFinite(v))return '';
  const a=Math.abs(v);
  if(a>=1e6)return (v/1e6).toFixed(1).replace(/\.0$/,'')+'M';
  if(a>=1e3)return (v/1e3).toFixed(1).replace(/\.0$/,'')+'k';
  if(Number.isInteger(v))return String(v);
  return String(Math.round(v*100)/100);
}
function chNiceMax(v){
  if(!(v>0))return 1;
  const exp=Math.floor(Math.log10(v)),f=v/Math.pow(10,exp);
  const nf=f<=1?1:f<=2?2:f<=2.5?2.5:f<=5?5:10;
  return nf*Math.pow(10,exp);
}

function renderCharts(wrapId,charts){
  const wrap=document.getElementById(wrapId);
  if(!wrap)return;
  wrap.innerHTML='';
  (charts||[]).forEach(spec=>{
    const fig=document.createElement('div');fig.className='chart-fig';
    const cv=document.createElement('canvas');cv.className='chart-canvas';
    fig.appendChild(cv);wrap.appendChild(fig);
    try{drawChart(cv,spec||{});}
    catch(e){fig.innerHTML=`<div class="chart-err">Could not render chart: ${esc(String(e))}</div>`;}
  });
}

function drawChart(canvas,spec){
  const ctype=String(spec.chart_type||'bar');
  const labels=(spec.labels||[]).map(String);
  const values=(spec.values||[]).map(v=>{const n=Number(v);return isFinite(n)?n:0;});
  const title=spec.title||'';
  const series=spec.series_label||'';
  const isPie=ctype==='pie'||ctype==='doughnut';
  const isH=ctype==='horizontal_bar';

  // Fixed backing-store size so drawing works even while the tab is hidden;
  // CSS scales it responsively. Multiply by dpr for crisp output.
  const W=isPie?440:760,H=420,dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.maxWidth=W+'px';
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);

  const colText=cssVar('--text','#e6edf3');
  const colSub =cssVar('--text3','#57636d');
  const colGrid=cssVar('--border','#30363d');
  const mono=cssVar('--mono','monospace');

  if(title){
    ctx.fillStyle=colText;ctx.font='600 15px '+cssVar('--sans','sans-serif');
    ctx.textAlign='center';ctx.textBaseline='top';
    ctx.fillText(chTrunc(title,64),W/2,12);
  }
  if(!labels.length){
    ctx.fillStyle=colSub;ctx.font='12px '+mono;
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText('No data to chart.',W/2,H/2);
    return;
  }
  if(isPie){drawPie(ctx,W,H,labels,values,ctype,colText);return;}

  // ── Cartesian charts: bar / horizontal_bar / line ──
  const top=title?46:22;
  const padL=isH?Math.min(170,14+chMaxLabelW(ctx,labels)):58;
  const padR=20,padB=62;
  const plotW=W-padL-padR,plotH=H-top-padB;
  const x0=padL,y0=top,x1=W-padR,y1=top+plotH;
  const maxV=Math.max(0,...values),niceMax=chNiceMax(maxV);
  const n=labels.length,ticks=5;

  ctx.lineWidth=1;ctx.font='11px '+mono;
  for(let i=0;i<=ticks;i++){
    const val=niceMax*i/ticks;
    if(!isH){
      const yy=y1-plotH*i/ticks;
      ctx.strokeStyle=colGrid;ctx.globalAlpha=.45;ctx.beginPath();ctx.moveTo(x0,yy);ctx.lineTo(x1,yy);ctx.stroke();ctx.globalAlpha=1;
      ctx.fillStyle=colSub;ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(chFmt(val),x0-8,yy);
    }else{
      const xx=x0+plotW*i/ticks;
      ctx.strokeStyle=colGrid;ctx.globalAlpha=.45;ctx.beginPath();ctx.moveTo(xx,y0);ctx.lineTo(xx,y1);ctx.stroke();ctx.globalAlpha=1;
      ctx.fillStyle=colSub;ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(chFmt(val),xx,y1+8);
    }
  }
  ctx.strokeStyle=colGrid;ctx.globalAlpha=1;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x0,y1);ctx.lineTo(x1,y1);ctx.stroke();

  if(ctype==='line'){
    const step=n>1?plotW/(n-1):0;
    const px=i=>x0+(n>1?step*i:plotW/2);
    const py=v=>y1-plotH*(niceMax?v/niceMax:0);
    ctx.strokeStyle=CHART_PALETTE[0];ctx.lineWidth=2;ctx.beginPath();
    values.forEach((v,i)=>{const xx=px(i),yy=py(v);i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});
    ctx.stroke();
    values.forEach((v,i)=>{
      const xx=px(i),yy=py(v);
      ctx.fillStyle=CHART_PALETTE[0];ctx.beginPath();ctx.arc(xx,yy,3,0,Math.PI*2);ctx.fill();
      chDrawXLabel(ctx,labels[i],xx,y1+8,colSub,mono,n>1?step:plotW);
    });
  }else if(isH){
    const band=plotH/n,bh=Math.min(30,band*.66);
    values.forEach((v,i)=>{
      const cy=y0+band*i+band/2,bw=plotW*(niceMax?v/niceMax:0);
      ctx.fillStyle=CHART_PALETTE[i%CHART_PALETTE.length];ctx.fillRect(x0,cy-bh/2,Math.max(0,bw),bh);
      ctx.fillStyle=colSub;ctx.font='11px '+mono;ctx.textAlign='right';ctx.textBaseline='middle';
      ctx.fillText(chTrunc(labels[i],20),x0-8,cy);
      ctx.fillStyle=colText;ctx.textAlign='left';ctx.fillText(chFmt(v),x0+Math.max(0,bw)+6,cy);
    });
  }else{ // vertical bar
    const band=plotW/n,bw=Math.min(54,band*.66);
    values.forEach((v,i)=>{
      const cx=x0+band*i+band/2,bh=plotH*(niceMax?v/niceMax:0);
      ctx.fillStyle=CHART_PALETTE[i%CHART_PALETTE.length];ctx.fillRect(cx-bw/2,y1-bh,bw,Math.max(0,bh));
      ctx.fillStyle=colText;ctx.font='10px '+mono;ctx.textAlign='center';ctx.textBaseline='bottom';
      ctx.fillText(chFmt(v),cx,y1-bh-3);
      chDrawXLabel(ctx,labels[i],cx,y1+8,colSub,mono,band);
    });
  }

  if(series){
    ctx.save();ctx.fillStyle=colSub;ctx.font='11px '+mono;
    if(!isH){ctx.translate(15,(y0+y1)/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(chTrunc(series,28),0,0);}
    else{ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText(chTrunc(series,40),(x0+x1)/2,H-6);}
    ctx.restore();
  }
}

function chMaxLabelW(ctx,labels){
  ctx.font='11px '+cssVar('--mono','monospace');
  let m=0;labels.forEach(l=>{const w=ctx.measureText(chTrunc(l,20)).width;if(w>m)m=w;});
  return m;
}
function chDrawXLabel(ctx,label,cx,y,col,mono,band){
  ctx.fillStyle=col;ctx.font='10px '+mono;
  const s=chTrunc(label,14),w=ctx.measureText(s).width;
  if(w>band-4){
    ctx.save();ctx.translate(cx,y);ctx.rotate(-Math.PI/4);ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(s,0,0);ctx.restore();
  }else{
    ctx.textAlign='center';ctx.textBaseline='top';ctx.fillText(s,cx,y);
  }
}
function drawPie(ctx,W,H,labels,values,ctype,colText){
  const mono=cssVar('--mono','monospace');
  const total=values.reduce((a,b)=>a+(b>0?b:0),0)||1;
  const cx=W*0.33,cy=H*0.55,r=Math.min(W*0.27,H*0.34);
  let a0=-Math.PI/2;
  values.forEach((v,i)=>{
    const a1=a0+((v>0?v:0)/total)*Math.PI*2;
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,a0,a1);ctx.closePath();
    ctx.fillStyle=CHART_PALETTE[i%CHART_PALETTE.length];ctx.fill();
    a0=a1;
  });
  if(ctype==='doughnut'){
    ctx.save();ctx.globalCompositeOperation='destination-out';
    ctx.beginPath();ctx.arc(cx,cy,r*0.56,0,Math.PI*2);ctx.fill();ctx.restore();
  }
  const lx=cx+r+22,lh=20,ly0=Math.max(28,cy-r);
  ctx.font='11px '+mono;ctx.textAlign='left';ctx.textBaseline='middle';
  labels.forEach((l,i)=>{
    const ly=ly0+lh*i;
    if(ly>H-14)return;
    ctx.fillStyle=CHART_PALETTE[i%CHART_PALETTE.length];ctx.fillRect(lx,ly-5,11,11);
    ctx.fillStyle=colText;
    const pct=Math.round((values[i]>0?values[i]:0)/total*100);
    ctx.fillText(chTrunc(l,16)+'  '+pct+'%',lx+17,ly);
  });
}
