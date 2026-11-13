"""Build the self-contained HTML trace viewer from simulator runs.

    uv run python experiments/build_viewer.py --seed 3 --regime base

Writes:
    results/viewer/<policy>.trace.jsonl   append-only decision log (the source of truth)
    results/viewer/<policy>.truth.jsonl    the oracle-only truth layer
    results/viewer/viewer.html             one offline file that renders those logs

The HTML file embeds the logs, so it opens by double-click with no server and no
network.  Nothing is displayed that is not in the log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ee_update import Config, run_one
from ee_update.config import regime as mk_regime

POLICIES = ["c3_voi_propose", "c1_voi", "greedy_judge", "accept_all", "threshold_promote", "static"]

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explore / exploit in the learning-update rule - decision trace</title>
<style>
  :root{
    --bg:#fbfbfc; --fg:#16181d; --muted:#6b7280; --line:#dcdfe5; --card:#ffffff;
    --explore:#009E73; --exploit:#7F7F7F; --accept:#0072B2; --reject:#D55E00;
    --band:rgba(0,114,178,.16);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
  h1{font-size:15px;margin:0 0 8px;font-weight:600}
  .row{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
  label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  select,button,input[type=range]{font:inherit}
  select,button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:3px 8px;cursor:pointer}
  button.on{background:#16181d;color:#fff;border-color:#16181d}
  .chip{display:inline-block;font-size:10px;font-weight:700;letter-spacing:.05em;padding:1px 6px;border-radius:9px;vertical-align:middle}
  .chip.explore{background:rgba(0,158,115,.14);color:var(--explore)}
  .chip.exploit{background:#eceef1;color:var(--exploit)}
  .chip.accept{background:rgba(0,114,178,.14);color:var(--accept)}
  .chip.reject{background:rgba(213,94,0,.14);color:var(--reject)}
  main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(340px,1fr);gap:14px;padding:14px 16px;align-items:start}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 8px;font-weight:600}
  .tabs{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}
  .tabs button{font-size:11px;padding:3px 9px}
  .tabs button.on{background:#16181d;color:#fff;border-color:#16181d}
  .tabpane{display:none;max-height:72vh;overflow:auto}
  .tabpane.on{display:block}
  svg{display:block;max-width:100%}
  .laneLabel{font:10px var(--f, sans-serif);fill:var(--muted)}
  .card{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-bottom:8px;background:#fff}
  .card.dim{opacity:.42}
  .card .head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:3px}
  .card .id{font-weight:600;font-size:12px}
  .card .rule{font-size:10px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .card .reason{font-size:12px;margin:3px 0}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:1px 10px;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#333}
  .kv b{color:var(--muted);font-weight:500}
  .scores{margin-top:5px;font-size:10px;font-family:ui-monospace,monospace}
  .bar{height:7px;background:#eef0f3;border-radius:4px;position:relative;overflow:hidden;margin-bottom:2px}
  .bar i{position:absolute;top:0;bottom:0;display:block}
  .pos i{background:#0072B2}.neg i{background:#D55E00}
  .muted{color:var(--muted);font-size:11px}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:6px}
  .legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
  table{border-collapse:collapse;width:100%;font-size:11px}
  td,th{text-align:left;padding:3px 6px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600}
  .node{cursor:pointer}
  .node:hover circle{stroke-width:2.5}
</style>
</head>
<body>
<header>
  <h1>Explore / exploit inside the learning-update rule &mdash; decision trace</h1>
  <div class="row">
    <span><label>policy</label> <select id="pol"></select></span>
    <span><label>compare</label> <select id="pol2"><option value="">(off)</option></select></span>
    <span><label>seed</label> <span id="seed"></span></span>
    <span><label>regime</label> <span id="reg"></span></span>
    <button id="play">play</button>
    <button id="truth">truth layer</button>
    <button id="expl">explore only</button>
    <span style="flex:1"><input id="scrub" type="range" min="0" max="100" value="100" style="width:100%"></span>
    <span class="muted" id="pos"></span>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Task stream, per family</h2>
    <div id="timeline"></div>
    <div class="legend">
      <span><i style="background:#0072B2"></i>task success</span>
      <span><i style="background:#D55E00"></i>task failure</span>
      <span><i style="background:#009E73"></i>candidate on live trial</span>
      <span><i style="background:#16181d"></i>skill accepted</span>
      <span><i style="background:#999"></i>skill rejected</span>
    </div>
    <h2 style="margin-top:14px">Cost ledger</h2>
    <div id="ledger"></div>
  </section>
  <section class="panel">
    <div class="tabs">
      <button data-tab="dec" class="on">decisions</button>
      <button data-tab="belief">belief</button>
      <button data-tab="lineage">lineage</button>
      <button data-tab="table">cost table</button>
    </div>
    <div class="tabpane on" id="tab-dec"></div>
    <div class="tabpane" id="tab-belief"></div>
    <div class="tabpane" id="tab-lineage"></div>
    <div class="tabpane" id="tab-table"></div>
  </section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const POLS = P.policies;
const NAMES = Object.keys(POLS);
const state = {pol:NAMES[0], pol2:"", t:P.T, truth:false, exploreOnly:false, playing:false, tab:"dec"};
const byId = id => document.getElementById(id);
const esc = s => String(s==null?"":s).replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmt = x => (x==null||Number.isNaN(x)) ? "-" : (Math.abs(x)>=100? x.toFixed(0) : x.toFixed(3));

function events(pol){ return POLS[pol].events; }
function truthMap(pol){
  const m = {}; (POLS[pol].truth||[]).forEach(t=>{ if(t.skill_id && t.true_value!=null) m[t.skill_id]=t; }); return m;
}
function tRange(pol){
  const es = events(pol); let mx=0; es.forEach(e=>{ if(e.t>mx) mx=e.t; }); return Math.max(mx,1);
}

function fillSelects(){
  const s1=byId("pol"), s2=byId("pol2");
  NAMES.forEach(n=>{ const o=document.createElement("option"); o.value=n; o.textContent=n; s1.appendChild(o); });
  NAMES.forEach(n=>{ const o=document.createElement("option"); o.value=n; o.textContent=n; s2.appendChild(o); });
  s1.value=state.pol;
  byId("seed").textContent=P.seed; byId("reg").textContent=P.regime;
  byId("scrub").max=P.T; byId("scrub").value=P.T;
}

function timeline(pol, tNow){
  const es = events(pol).filter(e=>e.type==="TASK_SERVED");
  const fams = [...new Set(es.map(e=>e.family))].sort((a,b)=>a-b);
  const W=900, padL=54, rowH=22, H=fams.length*rowH+8;
  const T=Math.max(tRange(pol),1);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">`;
  fams.forEach((f,i)=>{
    const y=i*rowH+14;
    s += `<text class="laneLabel" x="0" y="${y+4}">family ${f}</text>`;
    s += `<line x1="${padL}" y1="${y}" x2="${W-8}" y2="${y}" stroke="#eceef1"/>`;
  });
  es.forEach(e=>{
    const i=fams.indexOf(e.family); if(i<0) return;
    const y=i*rowH+14, x=padL+(e.t/T)*(W-padL-10);
    const col = e.arm==="candidate" ? "#009E73" : (e.outcome? "#0072B2":"#D55E00");
    const w = Math.max(1.6, (W-padL-10)/T*0.8);
    s += `<rect x="${x.toFixed(2)}" y="${y-5}" width="${w.toFixed(2)}" height="10" fill="${col}" opacity="${e.arm==="candidate"?0.95:0.75}"><title>t${e.t} family ${e.family} ${e.skill_id} ${e.outcome?"success":"failure"} (${e.routing_rule})</title></rect>`;
  });
  events(pol).forEach(e=>{
    const i=fams.indexOf(e.family); if(i<0) return;
    const y=i*rowH+14, x=padL+(e.t/T)*(W-padL-10);
    if(e.type==="SKILL_CREATED"){ s+=`<polygon points="${x},${y-9} ${x-4},${y-4} ${x+4},${y-4}" fill="#16181d"><title>created ${e.skill_id} from ${e.parent_id}</title></polygon>`; }
    if(e.type==="SKILL_ACCEPTED"){ s+=`<circle cx="${x}" cy="${y}" r="4" fill="#16181d" stroke="#fff" stroke-width="1"><title>accepted ${e.skill_id}</title></circle>`; }
    if(e.type==="SKILL_REJECTED"){ s+=`<rect x="${x-3.5}" y="${y-3.5}" width="7" height="7" fill="#999"><title>rejected ${e.skill_id}</title></rect>`; }
  });
  const x=padL+(tNow/T)*(W-padL-10);
  s += `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="#16181d" stroke-width="1" stroke-dasharray="3 3"/>`;
  s += `</svg>`;
  return s;
}

function ledger(pol, tNow){
  const es = events(pol).filter(e=>e.type==="TASK_SERVED" && e.t<=tNow);
  const comps={fail:0, task:0};
  es.forEach(e=>{ comps.task+=e.tokens||0; if(!e.outcome) comps.fail+=(e.cost-(e.tokens||0)); });
  let prop=0, ver=0, jud=0;
  events(pol).forEach(e=>{ if(e.type==="COST_CHARGED" && e.t<=tNow){ if(e.component==="proposal")prop+=e.amount; if(e.component==="verify")ver+=e.amount; if(e.component==="judge")jud+=e.amount; }});
  const rows=[["failed tasks x c_fail",comps.fail],["task rollouts",comps.task],["proposals",prop],["verification",ver],["judge calls",jud]];
  const total=rows.reduce((a,r)=>a+r[1],0);
  let s=`<svg viewBox="0 0 900 92" width="100%" height="92">`;
  let x=0; const W=890;
  const cols=["#D55E00","#0072B2","#E69F00","#009E73","#CC79A7"];
  rows.forEach((r,i)=>{ const w=total? W*r[1]/total : 0; if(w<=0) return;
    s+=`<rect x="${x.toFixed(1)}" y="10" width="${Math.max(w-1,0).toFixed(1)}" height="22" fill="${cols[i]}" opacity=".85"><title>${r[0]}: ${r[1].toFixed(1)}</title></rect>`; x+=w; });
  s+=`<text x="0" y="56" font-size="12" fill="#16181d">total cost ${total.toFixed(0)} rollout-equivalents so far</text>`;
  const ref = P.reference != null ? P.reference : null;
  if(ref!=null) s+=`<text x="0" y="74" font-size="11" fill="#6b7280">reference (cheaper of never-updating and the free-information oracle) ${ref.toFixed(0)} - regret ${(total-ref).toFixed(0)}</text>`;
  s+=`</svg>`;
  s+=`<table><tr><th>component</th><th>cost</th><th>share</th></tr>`;
  rows.forEach(r=>{ s+=`<tr><td>${r[0]}</td><td>${r[1].toFixed(1)}</td><td>${total? (100*r[1]/total).toFixed(1):0}%</td></tr>`; });
  s+=`</table>`;
  return s;
}

function decisions(pol, tNow){
  const es = events(pol).filter(e=>e.type==="ACTION_CHOSEN" || e.type==="PROPOSE_DECISION");
  const tm = truthMap(pol);
  let s="";
  es.filter(e=>e.t<=tNow).forEach(e=>{
    if(state.exploreOnly && e.mode!=="EXPLORE") return;
    const isAct = e.type==="ACTION_CHOSEN";
    const mode = e.mode || (e.chosen?"EXPLORE":"EXPLOIT");
    const cid = isAct? e.candidate_id : "-";
    s += `<div class="card"><div class="head">`;
    s += `<span class="id">${esc(cid)}</span>`;
    if(isAct) s += `<span class="chip ${e.action==="ACCEPT"?"accept":"explore"}">${esc(e.action)}${e.k? " k="+e.k:""}</span>`;
    else s += `<span class="chip ${e.chosen?"explore":"exploit"}">${e.chosen?"PROPOSE":"HOLD"}</span>`;
    s += `<span class="chip ${mode==="EXPLORE"?"explore":"exploit"}">${mode}</span>`;
    s += `<span class="muted">t=${e.t} &middot; family ${e.family}</span>`;
    s += `</div>`;
    if(e.reason) s += `<div class="reason">${esc(e.reason)}</div>`;
    if(e.rule_id) s += `<div class="rule">${esc(e.rule_id)}</div>`;
    if(isAct && e.greedy_action) s += `<div class="muted">greedy action without buying information: <b>${esc(e.greedy_action)}</b>${e.greedy_action!==e.action?" (this decision buys information)":""}</div>`;
    if(e.rule_inputs){
      const rows = Object.entries(e.rule_inputs).filter(([k])=>k!=="probe_info"&&k!=="verify_info"&&k!=="judge_info");
      if(rows.length){ s += `<div class="kv">`; rows.forEach(([k,v])=>{ s+=`<b>${esc(k)}</b><span>${typeof v==="number"?fmt(v):esc(v)}</span>`; }); s+=`</div>`; }
    }
    if(e.scores){
      const mx = Math.max(...Object.values(e.scores).map(v=>Math.abs(v)),1);
      s += `<div class="scores">`;
      Object.entries(e.scores).forEach(([k,v])=>{ const w=Math.min(100,Math.abs(v)/mx*100);
        s+=`<div>${esc(k)} ${fmt(v)}</div><div class="bar ${v>=0?"pos":"neg"}"><i style="left:${v>=0?50:50-w}%;width:${w/2}%"></i></div>`; });
      s += `</div>`;
    }
    if(state.truth){
      const tr = tm[cid];
      if(tr && tr.true_value!=null){ const d = tr.delta!=null? tr.delta : null;
        s += `<div class="muted">truth: true value ${fmt(tr.true_value)}${d!=null? `, delta vs parent ${fmt(d)}`:""}</div>`; }
    }
    s += `</div>`;
  });
  if(!s) s = `<div class="muted">no decision recorded yet - move the scrubber forward.</div>`;
  return s;
}

function belief(pol, tNow){
  const es = events(pol).filter(e=>e.type==="ACTION_CHOSEN" && e.rule_inputs && e.rule_inputs.mu_D!=null && e.t<=tNow);
  const tm = truthMap(pol);
  if(!es.length) return `<div class="muted">no belief updates yet.</div>`;
  const W=520,H=190,pad=34;
  const T=Math.max(tRange(pol),1);
  const xs=es.map(e=>e.t);
  const mus=es.map(e=>e.rule_inputs.mu_D), sds=es.map(e=>e.rule_inputs.s_D||0);
  const lo=Math.min(-0.1,...mus.map((m,i)=>m-sds[i])), hi=Math.max(0.1,...mus.map((m,i)=>m+sds[i]));
  const X=t=>pad+(t/T)*(W-pad-10), Y=v=>H-pad-((v-lo)/(hi-lo))*(H-2*pad);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  s+=`<line x1="${pad}" y1="${Y(0)}" x2="${W-10}" y2="${Y(0)}" stroke="#c9ccd3" stroke-dasharray="3 3"/>`;
  s+=`<polygon points="${es.map((e,i)=>X(e.t)+","+Y(mus[i]-sds[i])).join(" ")} ${es.slice().reverse().map((e,i)=>{const k=es.length-1-i; return X(e.t)+","+Y(mus[k]+sds[k]);}).join(" ")}" fill="#0072B2" fill-opacity=".16"/>`;
  s+=`<polyline points="${es.map((e,i)=>X(e.t)+","+Y(mus[i])).join(" ")}" fill="none" stroke="#0072B2" stroke-width="1.6"/>`;
  es.forEach((e,i)=>{ const col=e.action==="ACCEPT"?"#0072B2":(e.action==="SKIP"?"#999":"#E69F00");
    s+=`<circle cx="${X(e.t)}" cy="${Y(mus[i])}" r="3" fill="${col}"><title>t${e.t} ${e.action} mu=${fmt(mus[i])} sd=${fmt(sds[i])}</title></circle>`; });
  s+=`<text x="2" y="${Y(0)+3}" font-size="9" fill="#6b7280">0</text>`;
  s+=`<text x="${pad}" y="${H-6}" font-size="10" fill="#6b7280">task index</text>`;
  s+=`<text x="2" y="12" font-size="10" fill="#6b7280">belief on v_cand - v_inc</text>`;
  s+=`</svg>`;
  s+=`<div class="legend"><span><i style="background:#0072B2"></i>accept</span><span><i style="background:#E69F00"></i>explore (probate / verify / judge)</span><span><i style="background:#999"></i>skip</span></div>`;
  const last = es[es.length-1];
  const tr = tm[last.candidate_id];
  s += `<div class="muted" style="margin-top:6px">latest: ${esc(last.candidate_id)} mu=${fmt(last.rule_inputs.mu_D)} sd=${fmt(last.rule_inputs.s_D)} P(better)=${fmt(last.rule_inputs.p_better)} action=${esc(last.action)}${state.truth && tr? ` | true delta ${fmt(tr.delta)}`:""}</div>`;
  return s;
}

function lineage(pol){
  const es = events(pol);
  const created = es.filter(e=>e.type==="SKILL_CREATED");
  const acc = new Map(es.filter(e=>e.type==="SKILL_ACCEPTED").map(e=>[e.skill_id,e]));
  const rej = new Map(es.filter(e=>e.type==="SKILL_REJECTED").map(e=>[e.skill_id,e]));
  const served = new Map();
  es.filter(e=>e.type==="TASK_SERVED").forEach(e=>served.set(e.skill_id,(served.get(e.skill_id)||0)+1));
  const tm = truthMap(pol);
  if(!created.length) return `<div class="muted">no candidate was created by this policy.</div>`;
  let s=`<table><tr><th>skill</th><th>parent</th><th>outcome</th><th>tasks served</th><th>rule</th>`;
  if(state.truth) s+=`<th>true value</th>`;
  s+=`</tr>`;
  created.forEach(e=>{
    const id=e.skill_id, a=acc.get(id), r=rej.get(id);
    s+=`<tr><td>${esc(id)}</td><td>${esc(e.parent_id)}</td>`;
    s+=`<td>${a?`<span class="chip accept">accepted</span>`:(r?`<span class="chip reject">rejected</span>`:`<span class="chip exploit">not decided</span>`)}</td>`;
    s+=`<td>${served.get(id)||0}</td><td>${esc((a||r||{}).rule_id||"-")}</td>`;
    if(state.truth) s+=`<td>${tm[id]&&tm[id].true_value!=null?fmt(tm[id].true_value):"-"}</td>`;
    s+=`</tr>`;
  });
  s+=`</table>`;
  return s;
}

function costTable(pol){
  const es=events(pol);
  const rows={};
  es.filter(e=>e.type==="COST_CHARGED").forEach(e=>{ rows[e.component]=(rows[e.component]||0)+e.amount; });
  const task=es.filter(e=>e.type==="TASK_SERVED");
  const fail=task.filter(e=>!e.outcome).length;
  let s=`<table><tr><th>quantity</th><th>value</th></tr>`;
  s+=`<tr><td>tasks</td><td>${task.length}</td></tr>`;
  s+=`<tr><td>failed tasks</td><td>${fail}</td></tr>`;
  s+=`<tr><td>final success rate</td><td>${(1-fail/Math.max(task.length,1)).toFixed(3)}</td></tr>`;
  Object.entries(rows).forEach(([k,v])=>{ s+=`<tr><td>${esc(k)} cost</td><td>${v.toFixed(1)}</td></tr>`; });
  const decisions = es.filter(e=>e.type==="ACTION_CHOSEN");
  const ex = decisions.filter(e=>e.mode==="EXPLORE").length;
  s+=`<tr><td>update decisions</td><td>${decisions.length}</td></tr>`;
  s+=`<tr><td>of which explore</td><td>${ex}</td></tr>`;
  s+=`</table>`;
  return s;
}

function render(){
  const pol=state.pol;
  byId("pos").textContent = "task "+state.t+" / "+P.T;
  byId("timeline").innerHTML = timeline(pol, state.t);
  byId("ledger").innerHTML = ledger(pol, state.t);
  byId("tab-dec").innerHTML = decisions(pol, state.t);
  byId("tab-belief").innerHTML = belief(pol, state.t);
  byId("tab-lineage").innerHTML = lineage(pol);
  byId("tab-table").innerHTML = costTable(pol);
  if(state.pol2){
    byId("timeline").innerHTML += `<h2 style="margin-top:14px">compare: ${esc(state.pol2)}</h2>` + timeline(state.pol2, state.t);
  }
}

function init(){
  fillSelects();
  byId("pol").onchange = e=>{state.pol=e.target.value; render();};
  byId("pol2").onchange = e=>{state.pol2=e.target.value; render();};
  byId("scrub").oninput = e=>{state.t=+e.target.value; state.playing=false; byId("play").textContent="play"; render();};
  byId("truth").onclick = e=>{state.truth=!state.truth; e.target.classList.toggle("on",state.truth); render();};
  byId("expl").onclick = e=>{state.exploreOnly=!state.exploreOnly; e.target.classList.toggle("on",state.exploreOnly); render();};
  byId("play").onclick = e=>{
    state.playing=!state.playing; e.target.textContent=state.playing?"pause":"play";
    if(state.playing){ const iv=setInterval(()=>{ state.t=Math.min(P.T,state.t+Math.ceil(P.T/120)); byId("scrub").value=state.t; render();
      if(state.t>=P.T){ clearInterval(iv); state.playing=false; byId("play").textContent="play"; } }, 60); }
  };
  document.querySelectorAll(".tabs button").forEach(b=>{ b.onclick=()=>{ state.tab=b.dataset.tab;
    document.querySelectorAll(".tabs button").forEach(x=>x.classList.toggle("on",x===b));
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.toggle("on",p.id==="tab-"+state.tab)); }; });
  render();
}
init();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--regime", default="base")
    ap.add_argument("--out", default="results/viewer")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = mk_regime(args.regime)
    payload = {"seed": args.seed, "regime": args.regime, "T": cfg.T, "policies": {}}
    ref = None
    for pol in POLICIES:
        r = run_one(cfg, pol, args.seed, trace=True)
        tr = r["trace"]
        tr.write(out, pol)
        payload["policies"][pol] = {
            "events": tr.events,
            "truth": tr.truth,
            "summary": {k: v for k, v in r.items() if isinstance(v, (int, float))},
        }
    # reference cost: the cheaper of never-updating and the free-information oracle (same seed)
    oracle_cost = run_one(cfg, "oracle", args.seed)["total_cost"]
    static_cost = payload["policies"]["static"]["summary"]["total_cost"]
    payload["reference"] = min(oracle_cost, static_cost)

    html = HTML.replace("__PAYLOAD__", json.dumps(payload))
    (out / "viewer.html").write_text(html)
    print(f"wrote {out/'viewer.html'} ({len(html)/1e6:.1f} MB) and {len(POLICIES)} trace files in {out}")


if __name__ == "__main__":
    main()
