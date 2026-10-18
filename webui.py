"""Lightweight single-page UI (Chat / Historique / Config).

Served by app.py at ``/`` and ``/ui``. Deliberately dependency-free: one HTML
document, vanilla JS, no build step and no CDN bundle, so it stays usable on a
phone browser where the previous Gradio UI was too heavy.
"""

PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Qwen local</title>
<style>
  :root{
    --bg:#0b1220; --panel:#121a2b; --panel2:#0f1626; --line:#233049;
    --txt:#e6ecf5; --muted:#8ea0bd; --accent:#4f8cff; --accent2:#1f6feb;
    --user:#1f6feb; --bot:#182338; --ok:#31c48d; --err:#f87171;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--txt);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    display:flex;flex-direction:column;height:100dvh;overflow:hidden}
  header{display:flex;align-items:center;gap:8px;padding:10px 12px;background:var(--panel);
    border-bottom:1px solid var(--line);flex:0 0 auto}
  header .brand{font-weight:700;letter-spacing:.2px;margin-right:auto;white-space:nowrap}
  header .brand small{color:var(--muted);font-weight:500}
  .tabs{display:flex;gap:6px}
  .tab{background:transparent;border:1px solid transparent;color:var(--muted);
    padding:7px 12px;border-radius:9px;cursor:pointer;font-size:14px;white-space:nowrap}
  .tab.active{background:var(--panel2);color:var(--txt);border-color:var(--line)}
  main{flex:1 1 auto;min-height:0;display:flex;flex-direction:column}
  .view{flex:1 1 auto;min-height:0;display:none;flex-direction:column}
  .view.active{display:flex}

  /* Chat */
  #messages{flex:1 1 auto;overflow-y:auto;padding:16px 12px 8px;display:flex;flex-direction:column;gap:12px}
  .msg{max-width:min(760px,88%);padding:10px 14px;border-radius:14px;white-space:pre-wrap;
    word-wrap:break-word;overflow-wrap:anywhere}
  .msg.user{align-self:flex-end;background:var(--user);color:#fff;border-bottom-right-radius:4px}
  .msg.assistant{align-self:flex-start;background:var(--bot);border:1px solid var(--line);border-bottom-left-radius:4px}
  .msg .meta{font-size:11px;color:var(--muted);margin-top:6px}
  .empty{margin:auto;text-align:center;color:var(--muted);padding:24px}
  .empty h2{color:var(--txt);margin:0 0 6px}
  #composer{flex:0 0 auto;padding:10px 12px calc(10px + env(safe-area-inset-bottom));
    background:var(--panel);border-top:1px solid var(--line);display:flex;gap:8px;align-items:flex-end}
  #input{flex:1 1 auto;resize:none;max-height:40vh;min-height:44px;padding:11px 12px;border-radius:12px;
    background:var(--panel2);color:var(--txt);border:1px solid var(--line);font:inherit;outline:none}
  #input:focus{border-color:var(--accent)}
  button.primary{background:var(--accent2);color:#fff;border:none;border-radius:12px;
    padding:11px 16px;font-weight:600;cursor:pointer}
  button.primary:disabled{opacity:.5;cursor:not-allowed}
  button.ghost{background:transparent;color:var(--muted);border:1px solid var(--line);
    border-radius:10px;padding:8px 11px;cursor:pointer}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .chat-bar{display:flex;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);
    background:var(--panel2);flex:0 0 auto}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--muted);display:inline-block}
  .dot.ok{background:var(--ok)} .dot.err{background:var(--err)}

  /* Generic panes */
  .pane{flex:1 1 auto;overflow-y:auto;padding:14px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:14px}
  .card h3{margin:0 0 4px;font-size:15px}
  .card p.sub{margin:0 0 12px;color:var(--muted);font-size:13px}
  label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
  input[type=text],input[type=number],select{width:100%;padding:10px;border-radius:10px;
    background:var(--panel2);border:1px solid var(--line);color:var(--txt);font:inherit}
  .val{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;
    background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px;
    display:flex;gap:8px;align-items:center;justify-content:space-between;overflow:auto}
  .val code{white-space:nowrap;overflow:auto}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:13px}
  .kv .k{color:var(--muted)}
  .list{display:flex;flex-direction:column;gap:8px}
  .item{display:flex;gap:10px;align-items:center;background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:11px 12px;cursor:pointer}
  .item:hover{border-color:var(--accent)}
  .item .t{flex:1 1 auto;min-width:0}
  .item .t b{display:block;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .item .t span{color:var(--muted);font-size:12px}
  .iconbtn{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:9px;
    width:34px;height:34px;display:grid;place-items:center;cursor:pointer;flex:0 0 auto}
  .iconbtn:hover{color:var(--err);border-color:var(--err)}
  .hint{color:var(--muted);font-size:12px;margin-top:8px}
  .toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);background:#0d1a33;
    border:1px solid var(--line);color:var(--txt);padding:9px 14px;border-radius:10px;opacity:0;
    transition:opacity .2s;pointer-events:none;z-index:50}
  .toast.show{opacity:1}
  @media (max-width:560px){ .tab{padding:7px 9px;font-size:13px} header .brand small{display:none} }
</style>
</head>
<body>
<header>
  <div class="brand">Qwen&nbsp;local <small id="hdrModel">·&nbsp;0.6B</small></div>
  <nav class="tabs">
    <button class="tab active" data-view="chat">Chat</button>
    <button class="tab" data-view="history">Historique</button>
    <button class="tab" data-view="config">Config</button>
  </nav>
</header>

<main>
  <!-- CHAT -->
  <section class="view active" id="view-chat">
    <div class="chat-bar">
      <span class="dot" id="statusDot"></span>
      <span id="statusText" style="color:var(--muted);font-size:13px">…</span>
      <span style="flex:1"></span>
      <button class="ghost" id="newChat">+ Nouveau</button>
    </div>
    <div id="messages"><div class="empty"><h2>Qwen local</h2><p>Pose ta question ci-dessous.</p></div></div>
    <div id="composer">
      <textarea id="input" rows="1" placeholder="Écris un message…  (Entrée = envoyer, Maj+Entrée = nouvelle ligne)"></textarea>
      <button class="primary" id="send">Envoyer</button>
    </div>
  </section>

  <!-- HISTORIQUE -->
  <section class="view" id="view-history">
    <div class="pane">
      <div class="card">
        <h3>Conversations</h3>
        <p class="sub">Historique stocké localement (SQLite). Clique pour rouvrir.</p>
        <div class="list" id="convList"></div>
      </div>
    </div>
  </section>

  <!-- CONFIG -->
  <section class="view" id="view-config">
    <div class="pane">
      <div class="card">
        <h3>API locale</h3>
        <p class="sub">Utilise cette URL dans n'importe quel client <b>OpenAI-compatible</b> (ta clé API peut rester vide).</p>
        <label>URL de base (réseau local)</label>
        <div class="val"><code id="apiDirect">…</code><button class="ghost" data-copy="apiDirect">Copier</button></div>
        <label>URL via la passerelle VPS (proxy)</label>
        <div class="val"><code id="apiProxy">…</code><button class="ghost" data-copy="apiProxy">Copier</button></div>
        <label>Nom du modèle (à mettre dans <code>"model"</code>)</label>
        <div class="val"><code id="apiModel">…</code><button class="ghost" data-copy="apiModel">Copier</button></div>
        <div class="hint">Endpoint : <code>POST /v1/chat/completions</code> · modèles : <code>GET /v1/models</code> · santé : <code>GET /health</code></div>
      </div>

      <div class="card">
        <h3>Paramètres de génération</h3>
        <p class="sub">Appliqués à la page Chat.</p>
        <label>Température : <span id="tempOut">0.7</span></label>
        <input type="number" id="temp" min="0" max="2" step="0.1" value="0.7">
        <label>Max tokens : <span id="maxtokOut">512</span></label>
        <input type="number" id="maxtok" min="16" max="4096" step="16" value="512">
        <label>Top-p : <span id="toppOut">0.9</span></label>
        <input type="number" id="topp" min="0.1" max="1" step="0.05" value="0.9">
        <label>Répétition (repeat penalty) : <span id="repOut">1.1</span></label>
        <input type="number" id="rep" min="1" max="2" step="0.05" value="1.1">
        <div class="row" style="margin-top:12px">
          <button class="ghost" id="resetCfg">Réinitialiser</button>
        </div>
      </div>

      <div class="card">
        <h3>Modèle</h3>
        <p class="sub">Charger un autre GGUF (repo Hugging Face ou chemin local).</p>
        <div class="kv" style="margin-bottom:10px">
          <span class="k">Actuel</span><span id="curModel">…</span>
          <span class="k">État</span><span id="curState">…</span>
          <span class="k">Threads / ctx</span><span id="curThreads">…</span>
        </div>
        <label>Repo ID ou chemin</label>
        <input type="text" id="modelInput" placeholder="ex: MaziyarPanahi/Qwen3-0.6B-GGUF ou /root/models/xxx.gguf">
        <div class="row" style="margin-top:12px">
          <button class="primary" id="loadModel">Charger</button>
          <span id="modelMsg" style="color:var(--muted);font-size:13px"></span>
        </div>
      </div>
    </div>
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
const $ = (id) => document.getElementById(id);
const DEFAULTS = {temp:0.7, maxtok:512, topp:0.9, rep:1.1};
let currentCid = null;
let sending = false;

function toast(msg){
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toast._t); toast._t = setTimeout(()=>t.classList.remove('show'), 1600);
}
function copy(id){
  const txt = $(id).textContent;
  navigator.clipboard?.writeText(txt).then(()=>toast('Copié'), ()=>toast(txt));
}

/* ---- Tabs ---- */
function show(view){
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active', b.dataset.view===view));
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active', v.id==='view-'+view));
  if(view==='history') loadHistory();
  if(view==='config') loadConfig();
}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>show(b.dataset.view)));
document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',()=>copy(b.dataset.copy)));

/* ---- Settings (persisted) ---- */
function loadSettings(){
  for(const k of Object.keys(DEFAULTS)){
    const v = localStorage.getItem('qwen.'+k);
    if(v!==null) $(k).value = v;
    $(k).addEventListener('input', ()=>{
      localStorage.setItem('qwen.'+k, $(k).value);
      updateOuts();
    });
  }
  $('resetCfg').addEventListener('click', ()=>{
    for(const k of Object.keys(DEFAULTS)){ $(k).value = DEFAULTS[k]; localStorage.removeItem('qwen.'+k); }
    updateOuts(); toast('Paramètres réinitialisés');
  });
  updateOuts();
}
function updateOuts(){
  $('tempOut').textContent = $('temp').value;
  $('maxtokOut').textContent = $('maxtok').value;
  $('toppOut').textContent = $('topp').value;
  $('repOut').textContent = $('rep').value;
}
function params(){
  return { temperature:+$('temp').value, max_tokens:+$('maxtok').value,
           top_p:+$('topp').value, repeat_penalty:+$('rep').value };
}

/* ---- Chat rendering ---- */
function clearEmpty(){
  const m = $('messages'); const e = m.querySelector('.empty'); if(e) e.remove();
}
function addMsg(role, text){
  clearEmpty();
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text || '';
  $('messages').appendChild(d);
  $('messages').scrollTop = $('messages').scrollHeight;
  return d;
}
function renderConversation(rows){
  const m = $('messages'); m.innerHTML = '';
  if(!rows.length){
    m.innerHTML = '<div class="empty"><h2>Qwen local</h2><p>Pose ta question ci-dessous.</p></div>';
    return;
  }
  for(const r of rows) addMsg(r.role, r.content);
  m.scrollTop = m.scrollHeight;
}

async function send(){
  const text = $('input').value.trim();
  if(!text || sending) return;
  sending = true; $('send').disabled = true;
  $('input').value = ''; $('input').style.height = 'auto';
  addMsg('user', text);
  const bot = addMsg('assistant', '');
  bot.innerHTML = '<span style="color:var(--muted)">…</span>';
  try{
    const res = await fetch('ui/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(Object.assign({ conversation_id: currentCid, message: text }, params()))
    });
    if(!res.ok || !res.body) throw new Error('HTTP '+res.status);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '', acc = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let idx;
      while((idx = buf.indexOf('\n\n')) >= 0){
        const evt = buf.slice(0, idx); buf = buf.slice(idx+2);
        const line = evt.split('\n').find(l=>l.startsWith('data:'));
        if(!line) continue;
        let obj; try{ obj = JSON.parse(line.slice(5).trim()); }catch(e){ continue; }
        if(obj.conversation_id){ currentCid = obj.conversation_id; }
        if(obj.token){ acc += obj.token; bot.textContent = acc; $('messages').scrollTop = $('messages').scrollHeight; }
        if(obj.error){ acc += '\n[erreur] ' + obj.error; bot.textContent = acc; }
      }
    }
    if(!acc) bot.textContent = '(réponse vide)';
  }catch(err){
    bot.textContent = '[échec] ' + err.message;
  }finally{
    sending = false; $('send').disabled = false; $('input').focus();
  }
}

$('send').addEventListener('click', send);
$('input').addEventListener('keydown', (e)=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
});
$('input').addEventListener('input', (e)=>{
  e.target.style.height='auto'; e.target.style.height = Math.min(e.target.scrollHeight, 220)+'px';
});
$('newChat').addEventListener('click', ()=>{
  currentCid = null; renderConversation([]); $('input').focus(); show('chat');
});

/* ---- History ---- */
async function loadHistory(){
  const box = $('convList'); box.innerHTML = '<div class="hint">Chargement…</div>';
  try{
    const rows = await (await fetch('conversations')).json();
    if(!rows.length){ box.innerHTML = '<div class="hint">Aucune conversation.</div>'; return; }
    box.innerHTML = '';
    for(const c of rows){
      const d = document.createElement('div'); d.className='item';
      const when = c.created_at ? String(c.created_at) : '';
      d.innerHTML = '<div class="t"><b></b><span></span></div>';
      d.querySelector('b').textContent = c.title || ('Conversation #'+c.id);
      d.querySelector('span').textContent = '#'+c.id + (when?' · '+when:'');
      const del = document.createElement('button'); del.className='iconbtn'; del.title='Supprimer'; del.textContent='🗑';
      del.addEventListener('click', async (e)=>{
        e.stopPropagation();
        if(!confirm('Supprimer cette conversation ?')) return;
        await fetch('conversation/'+c.id, {method:'DELETE'});
        if(currentCid===c.id){ currentCid=null; renderConversation([]); }
        loadHistory(); toast('Supprimée');
      });
      d.appendChild(del);
      d.addEventListener('click', async ()=>{
        const msgs = await (await fetch('ui/conversation/'+c.id)).json();
        currentCid = c.id; renderConversation(msgs); show('chat');
      });
      box.appendChild(d);
    }
  }catch(e){ box.innerHTML = '<div class="hint">Erreur: '+e.message+'</div>'; }
}

/* ---- Config ---- */
async function loadConfig(){
  try{
    const cfg = await (await fetch('ui/config')).json();
    $('apiDirect').textContent = 'http://' + cfg.lan_ip + ':' + cfg.port + '/v1';
    $('apiProxy').textContent = location.origin + location.pathname.replace(/\/+$/, '') + '/v1';
    $('apiModel').textContent = cfg.model_name;
    $('curModel').textContent = cfg.model_path;
    $('curState').textContent = cfg.model_ready ? 'prêt' : 'chargement…';
    $('curThreads').textContent = cfg.n_threads + ' / ' + cfg.n_ctx;
    $('hdrModel').textContent = '· ' + cfg.model_name;
  }catch(e){ toast('Config indisponible'); }
}
$('loadModel').addEventListener('click', async ()=>{
  const model = $('modelInput').value.trim();
  if(!model){ toast('Indique un repo ou un chemin'); return; }
  $('modelMsg').textContent = 'Chargement…';
  try{
    const r = await fetch('ui/model', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model})});
    const j = await r.json();
    if(j.ok){ $('modelMsg').textContent = '✅ ' + j.model; loadConfig(); }
    else{ $('modelMsg').textContent = '❌ ' + (j.error||'erreur'); }
  }catch(e){ $('modelMsg').textContent = '❌ ' + e.message; }
});

/* ---- Status + boot ---- */
async function refreshStatus(){
  try{
    const h = await (await fetch('health')).json();
    const ready = h.model_ready;
    $('statusDot').className = 'dot ' + (ready?'ok':'err');
    $('statusText').textContent = ready ? 'Modèle prêt' : ('Chargement… ' + (h.error||''));
  }catch(e){ $('statusDot').className='dot err'; $('statusText').textContent='Serveur injoignable'; }
}
loadSettings();
refreshStatus(); setInterval(refreshStatus, 8000);
$('input').focus();
</script>
</body>
</html>
"""
