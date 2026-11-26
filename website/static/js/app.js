/* Smart Investment Portfolio System — frontend logic
   Interactive builder: the server sends one "profile" per risk level (weights,
   prices, risk metrics and a pre-computed Monte Carlo). Everything else — amount,
   horizon, shares, forecast, histogram — is recomputed here in the browser, so
   the sliders update every chart instantly with no waiting. */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const money = (v) => "$" + Math.round(v).toLocaleString("en-US");
const pct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
const pctFlat = (v) => (v * 100).toFixed(0) + "%";

// theme colors for charts
const C = { navy:"#0B2545", green:"#0FB981", greenSoft:"rgba(15,185,129,.16)",
  ice:"#9DB4CE", red:"#DC2626", line:"#E4EAF1", muted:"#5B6B7F" };
Chart.defaults.font.family = "'IBM Plex Mono','Inter',monospace";
Chart.defaults.font.size = 11;
Chart.defaults.color = C.muted;

const RISKS = ["Low", "Medium", "High"];
const RISK_TEXT = ["steadiest names, risk-balanced", "the model's top picks, by conviction", "most volatile of the top picks"];
const MIN_A = 1000, MAX_A = 1000000;           // amount slider range (log scale)
const REFRESH_MS = 10 * 60 * 1000;             // re-fetch profiles (fresh prices) every 10 min

const state = { riskIdx:1, amount:10000, horizon:60, profiles:{}, inflight:{}, lastPortfolio:null, shown:false };
const charts = {};
function destroy(k){ if(charts[k]){ charts[k].destroy(); delete charts[k]; } }

/* ---------- chart helper: update in place so dragging is smooth ---------- */
function upsert(key, canvas, cfg){
  const ch = charts[key];
  if(ch){ ch.data = cfg.data; ch.options = cfg.options; ch.update("none"); return ch; }
  return (charts[key] = new Chart(canvas, cfg));
}

/* ---------- profiles from the server ---------- */
async function loadProfile(risk, force){
  const cur = state.profiles[risk];
  if(cur && !force) return cur;
  if(state.inflight[risk]) return state.inflight[risk];
  state.inflight[risk] = (async () => {
    try{
      const res = await fetch("/api/profile?risk=" + risk, {cache:"no-store"});
      if(!res.ok) throw new Error("bad status " + res.status);
      const P = await res.json(); P.ts = Date.now();
      state.profiles[risk] = P; return P;
    } finally { delete state.inflight[risk]; }
  })();
  return state.inflight[risk];
}

/* ---------- the maths (mirrors portfolio_engine.py) ---------- */
function histFromQuantiles(q, amount, nSims, bins = 30){
  const lo = q[0], hi = q[q.length-1], width = (hi - lo) / bins;
  if(!(width > 0)) return { x:[lo*amount], y:[nSims] };
  const cdf = (x) => {                         // piecewise-linear CDF through the quantile knots
    if(x <= lo) return 0; if(x >= hi) return 1;
    let a = 0, b = q.length - 1;
    while(b - a > 1){ const m = (a + b) >> 1; if(q[m] <= x) a = m; else b = m; }
    const span = q[b] - q[a];
    return (a + (span > 0 ? (x - q[a]) / span : 0)) / (q.length - 1);
  };
  const x = [], y = [];
  for(let j = 0; j < bins; j++){
    const e0 = lo + width*j, e1 = lo + width*(j+1);
    x.push((e0 + e1) / 2 * amount);
    y.push(Math.round(nSims * (cdf(e1) - cdf(e0))));
  }
  return { x, y };
}

function compute(P, amount, horizon){
  const h = Math.max(5, Math.min(P.hz_max, Math.round(horizon / 5) * 5));
  const holdings = P.holdings.map((x) => {
    const allocation = x.weight * amount;
    return { ticker:x.ticker, weight:x.weight, allocation, shares:Math.floor(allocation / x.price),
             price:x.price, score:x.score, model_price:x.model_price };
  });
  const invested = holdings.reduce((s, x) => s + x.shares * x.price, 0);
  const b = P.bands, scale = (arr) => arr.slice(0, h + 1).map((v) => v * amount);
  const hist = histFromQuantiles(P.q[P.hz.indexOf(h)], amount, P.n_sims);
  return {
    as_of:P.as_of, live:P.live,
    inputs:{ amount, risk:P.risk, horizon:h },
    invested, cash_left:amount - invested, n_stocks:P.n_stocks, holdings,
    risk:P.metrics,
    value_curve:P.curve.map((v) => v * amount),
    forecast:{
      expected_value:b.p50[h] * amount, expected_return:b.p50[h] - 1,
      worst_case:b.p5[h] * amount, best_case:b.p95[h] * amount, prob_profit:P.prob[h],
      days:Array.from({length:h + 1}, (_, i) => i),
      p5:scale(b.p5), p50:scale(b.p50), p95:scale(b.p95), hist_x:hist.x, hist_y:hist.y
    }
  };
}

/* ---------- controls (hero console + sticky dock stay in sync) ---------- */
const posToAmount = (p) => {
  const raw = MIN_A * Math.pow(MAX_A / MIN_A, p / 1000);
  const step = raw < 10000 ? 100 : raw < 100000 ? 500 : 5000;
  return Math.round(raw / step) * step;
};
const amountToPos = (a) => Math.round(1000 * Math.log(Math.min(Math.max(a, MIN_A), MAX_A) / MIN_A) / Math.log(MAX_A / MIN_A));
const hzLabel = (d) => d + " days · ~" + Math.max(1, Math.round(d / 21)) + " mo";

function syncUI(source){
  $$('[data-ctl="amount"]').forEach((el) => { if(el !== source) el.value = amountToPos(state.amount); });
  $$('[data-ctl="risk"]').forEach((el) => { if(el !== source){ el.value = state.riskIdx; } el.setAttribute("aria-valuetext", RISKS[state.riskIdx]); });
  $$('[data-ctl="horizon"]').forEach((el) => { if(el !== source) el.value = state.horizon; });
  const num = $("#amount"); if(num && num !== source) num.value = state.amount;
  $$('[data-out="amount"]').forEach((el) => el.textContent = money(state.amount));
  $$('[data-out="risk"]').forEach((el) => el.textContent = RISKS[state.riskIdx]);
  $$('[data-out="horizon"]').forEach((el) => el.textContent = hzLabel(state.horizon));
  $$("[data-risk-set]").forEach((b) => b.classList.toggle("on", +b.dataset.riskSet === state.riskIdx));
  const P = state.profiles[RISKS[state.riskIdx]];
  $$('[data-out="riskdesc"]').forEach((el) => el.textContent =
    (P ? P.n_stocks : [25, 15, 8][state.riskIdx]) + " stocks · " + RISK_TEXT[state.riskIdx]);
}

let frame = 0;
function scheduleUpdate(){ if(!frame) frame = requestAnimationFrame(() => { frame = 0; update(); }); }

$$("[data-ctl]").forEach((el) => el.addEventListener("input", (e) => {
  const k = el.dataset.ctl, v = +el.value;
  if(k === "amount") state.amount = posToAmount(v);
  else if(k === "risk") state.riskIdx = v;
  else if(k === "horizon") state.horizon = v;
  syncUI(el); scheduleUpdate();
}));
$$("[data-risk-set]").forEach((b) => b.addEventListener("click", () => { state.riskIdx = +b.dataset.riskSet; syncUI(); scheduleUpdate(); }));

const amountBox = $("#amount");
amountBox.addEventListener("input", () => {
  const v = +amountBox.value;
  if(v >= 100 && v <= 10000000){ state.amount = v; syncUI(amountBox); scheduleUpdate(); }
});
amountBox.addEventListener("change", () => {
  state.amount = Math.min(10000000, Math.max(100, +amountBox.value || 10000)); syncUI(); scheduleUpdate();
});
$("#seeResults").addEventListener("click", () => $("#results").scrollIntoView({behavior:"smooth"}));
$("#saveBtn").addEventListener("click", savePortfolio);
$("#clearBtn").addEventListener("click", clearSaved);

function fitDock(){ document.documentElement.style.setProperty("--topbar-h", ($(".topbar").offsetHeight || 58) + "px"); }
window.addEventListener("resize", fitDock); fitDock();

/* ---------- update loop ---------- */
async function update(){
  const risk = RISKS[state.riskIdx];
  const P = state.profiles[risk];
  if(!P){
    $("#livePreview").innerHTML = '<div class="lp wide">Loading ' + risk.toLowerCase() + '-risk portfolio…</div>';
    try{ await loadProfile(risk); }catch(e){
      $("#livePreview").innerHTML = '<div class="lp wide">Could not reach the server — retrying…</div>';
      return setTimeout(scheduleUpdate, 3000);
    }
    return scheduleUpdate();
  }
  const d = compute(P, state.amount, state.horizon);
  state.lastPortfolio = d;
  renderAll(d);
  if(!state.shown){
    state.shown = true;
    const r = $("#results"); r.hidden = false;
    r.querySelectorAll(".block").forEach((b, i) => { b.style.animationDelay = (i * 60) + "ms"; b.classList.add("reveal"); });
  }
  syncUI();
  if(Date.now() - P.ts > REFRESH_MS) loadProfile(risk, true).then(scheduleUpdate).catch(() => {});
}

/* ---------- render ---------- */
function renderAll(d){
  const inp = d.inputs, f = d.forecast, lv = d.live || {};
  $("#recSub").textContent = `${inp.risk}-risk · ${money(inp.amount)} · ${inp.horizon}-day horizon · `
    + (lv.active ? `live prices${lv.prices_as_of ? " (close " + lv.prices_as_of + ")" : ""} · model data as of ${d.as_of}`
                 : `as of ${d.as_of}`);

  // hero live preview
  $("#livePreview").innerHTML =
      `<div class="lp"><div class="v ${f.expected_return>=0?"pos":"neg"}">${money(f.expected_value)}</div><div class="l">Expected value (${pct(f.expected_return)})</div></div>`
    + `<div class="lp"><div class="v">${pctFlat(f.prob_profit)}</div><div class="l">Chance of profit</div></div>`
    + `<div class="lp"><div class="v neg">${money(f.worst_case)}</div><div class="l">Worst case (5%)</div></div>`
    + `<div class="lp"><div class="v pos">${money(f.best_case)}</div><div class="l">Best case (95%)</div></div>`;

  // recommendation stats
  $("#recStats").innerHTML = stat(money(inp.amount),"You invest","accent")
    + stat(d.n_stocks,"Stocks held","accent")
    + stat(money(d.invested),"Deployed","accent")
    + stat(money(d.cash_left),"Cash left over","accent");

  // holdings table
  const priceHead = (d.live && d.live.active) ? "Live price" : "Price";
  let h = "<thead><tr><th>Ticker</th><th>Weight</th><th>Allocation</th><th>Shares</th><th>" + priceHead + "</th><th>Score</th></tr></thead><tbody>";
  d.holdings.forEach(x=>{ h += `<tr><td class="tkr">${x.ticker}</td><td>${(x.weight*100).toFixed(1)}%</td>`
    + `<td>${money(x.allocation)}</td><td>${x.shares}</td><td>$${x.price.toFixed(2)}${driftTag(x)}</td><td>${x.score.toFixed(3)}</td></tr>`; });
  $("#holdings").innerHTML = h + "</tbody>";

  drawAllocation(d.holdings);

  // risk
  const r = d.risk;
  $("#riskStats").innerHTML = stat((r.annual_vol*100).toFixed(1)+"%","Annual volatility","accent")
    + stat(pct(r.annual_return),"Historical return", r.annual_return>=0?"pos":"neg")
    + stat(r.sharpe.toFixed(2),"Sharpe ratio", r.sharpe>=0?"pos":"neg")
    + stat((r.max_drawdown*100).toFixed(1)+"%","Worst drawdown","neg");
  const match = r.band === inp.risk;
  const bn = $("#bandNote"); bn.className = "band-note " + (match?"match":"mismatch");
  bn.innerHTML = match
    ? `<strong>Realised risk band: ${r.band}</strong> — matches your chosen appetite. ${r.band_note}`
    : `You chose <strong>${inp.risk}</strong>, but this basket reads as <strong>${r.band}</strong>. ${r.band_note}`;
  drawValueCurve(d.value_curve, inp.amount);

  // forecast
  $("#fcSub").textContent = `Where ${money(inp.amount)} could be after ${inp.horizon} trading days.`;
  $("#fcStats").innerHTML = stat(money(f.expected_value),"Expected value", f.expected_return>=0?"pos":"neg")
    + stat(money(f.best_case),"Best case (95%)","pos")
    + stat(money(f.worst_case),"Worst case (5%)","neg")
    + stat(pctFlat(f.prob_profit),"Chance of profit","accent");
  drawCone(f, inp.amount);
  drawHist(f);

  renderConclusion(d);
}

function stat(v,l,cls=""){ return `<div class="stat"><div class="v num ${cls}">${v}</div><div class="l">${l}</div></div>`; }

/* ---------- charts (created once, then updated in place) ---------- */
function drawAllocation(holdings){
  const top = holdings.slice(0,8);
  const others = holdings.slice(8).reduce((s,x)=>s+x.allocation,0);
  const labels = top.map(x=>x.ticker).concat(others>0?["Other"]:[]);
  const vals = top.map(x=>x.allocation).concat(others>0?[others]:[]);
  const palette = ["#0FB981","#13315C","#1C4270","#38B2AC","#2C7A7B","#4A6FA5","#7FA8D4","#0A8F63","#9DB4CE"];
  upsert("alloc", $("#allocChart"), {type:"doughnut",
    data:{labels,datasets:[{data:vals,backgroundColor:palette,borderColor:"#fff",borderWidth:2}]},
    options:{cutout:"62%",plugins:{legend:{position:"right",labels:{boxWidth:10,padding:8,font:{size:10}}},
      tooltip:{callbacks:{label:(c)=>` ${c.label}: ${money(c.raw)}`}}}}});
}

function drawValueCurve(curve, amount){
  upsert("value", $("#valueChart"), {type:"line",
    data:{labels:curve.map((_,i)=>i),datasets:[{data:curve,borderColor:C.navy,borderWidth:2,
      fill:true,backgroundColor:"rgba(11,37,69,.06)",pointRadius:0,tension:.15}]},
    options:{plugins:{legend:{display:false}},scales:{
      x:{display:false},
      y:{ticks:{callback:v=>"$"+(v/1000).toFixed(0)+"k"},grid:{color:C.line}}}}});
}

function drawCone(f, amount){
  upsert("cone", $("#coneChart"), {type:"line",
    data:{labels:f.days,datasets:[
      {label:"Worst 5%",data:f.p5,borderColor:"rgba(220,38,38,.55)",borderWidth:1.5,pointRadius:0,fill:false,tension:.1},
      {label:"Best 95%",data:f.p95,borderColor:"rgba(15,185,129,.55)",borderWidth:1.5,pointRadius:0,fill:"-1",backgroundColor:C.greenSoft,tension:.1},
      {label:"Expected",data:f.p50,borderColor:C.green,borderWidth:2.5,pointRadius:0,tension:.1},
    ]},
    options:{interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{boxWidth:12,font:{size:10}}},
        tooltip:{callbacks:{label:(c)=>` ${c.dataset.label}: ${money(c.raw)}`}}},
      scales:{x:{title:{display:true,text:"trading days",font:{size:10}},grid:{display:false},ticks:{maxTicksLimit:8}},
        y:{ticks:{callback:v=>"$"+(v/1000).toFixed(1)+"k"},grid:{color:C.line}}}}});
}

function drawHist(f){
  upsert("hist", $("#histChart"), {type:"bar",
    data:{labels:f.hist_x.map(v=>Math.round(v/1000)+"k"),datasets:[{data:f.hist_y,
      backgroundColor:C.green,borderRadius:2,barPercentage:1,categoryPercentage:.98}]},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{title:(c)=>"~$"+c[0].label,label:(c)=>` ${c.raw} simulations`}}},
      scales:{x:{grid:{display:false},ticks:{maxTicksLimit:7}},y:{grid:{color:C.line},title:{display:true,text:"simulations",font:{size:10}}}}}});
}

/* ---------- conclusion ---------- */
function renderConclusion(d){
  const inp=d.inputs, r=d.risk, f=d.forecast;
  const dir = f.expected_return>=0 ? "grow" : "decline";
  const top3 = d.holdings.slice(0,3).map(x=>x.ticker).join(", ");
  let verdictClass="mid", verdictText="Roughly balanced odds over this horizon.";
  if(f.prob_profit>=.55){verdictClass="good";verdictText="A reasonably favourable setup on these assumptions.";}
  else if(f.prob_profit<.45){verdictClass="bad";verdictText="Odds lean unfavourable — consider a longer horizon or lower risk.";}
  $("#conclusion").innerHTML =
    `For <b>${money(inp.amount)}</b> at a <b>${inp.risk}</b> risk appetite, the system holds `
    + `<b>${d.n_stocks}</b> stocks (led by ${top3}), with realised volatility reading as <b>${r.band}</b> `
    + `and a Sharpe ratio of <b>${r.sharpe.toFixed(2)}</b>. Over <b>${inp.horizon}</b> trading days the Monte Carlo `
    + `forecast expects the investment to ${dir} to about <b>${money(f.expected_value)}</b> (<b>${pct(f.expected_return)}</b>), `
    + `ranging from <b>${money(f.worst_case)}</b> to <b>${money(f.best_case)}</b>, with a <b>${pctFlat(f.prob_profit)}</b> chance of profit.`
    + `<br><span class="verdict ${verdictClass}">${verdictText}</span>`;
}

/* ---------- save / compare ---------- */
async function savePortfolio(){
  if(!state.lastPortfolio){ return; }
  const res = await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(state.lastPortfolio)});
  const data = await res.json();
  if(res.ok){ $("#saveMsg").textContent = `Saved as #${data.id}.`; renderSaved(data.saved);
    setTimeout(()=>$("#saveMsg").textContent="",4000); }
}

async function loadSaved(){
  const res = await fetch("/api/saved"); renderSaved(await res.json());
}

function renderSaved(rows){
  const t = $("#savedTable");
  if(!rows || !rows.length){ t.innerHTML = ""; t.insertAdjacentHTML("afterend",""); 
    t.innerHTML = "<tbody><tr><td class='empty' colspan='8'>No saved portfolios yet — generate one and click “Save this portfolio”.</td></tr></tbody>"; return; }
  let h = "<thead><tr><th>Label</th><th>Risk</th><th>Amount</th><th>Vol</th><th>Return</th><th>Sharpe</th><th>P(profit)</th><th>Stocks</th></tr></thead><tbody>";
  rows.forEach(r=>{ h += `<tr><td>${r.label}</td><td>${r.risk}</td><td>${money(r.amount)}</td>`
    + `<td>${(r.annual_vol*100).toFixed(1)}%</td>`
    + `<td class="${r.expected_return>=0?'pos':'neg'}">${pct(r.expected_return)}</td>`
    + `<td class="${r.sharpe>=0?'pos':'neg'}">${r.sharpe.toFixed(2)}</td>`
    + `<td>${pctFlat(r.prob_profit)}</td><td>${r.n_stocks}</td></tr>`; });
  t.innerHTML = h + "</tbody>";
}

async function clearSaved(){
  await fetch("/api/clear",{method:"POST"}); renderSaved([]);
}

/* ---------- model backtest (loads on page load) ---------- */
async function loadBacktest(){
  try{
    const res = await fetch("/api/backtest"); const d = await res.json();
    const colors = d.deciles.map((_,i)=> i>=8 ? C.green : (i>=5 ? "#4A6FA5" : "#9DB4CE"));
    destroy("decile");
    charts.decile = new Chart($("#decileChart"),{type:"bar",
      data:{labels:d.deciles.map((_,i)=>"D"+i),datasets:[{data:d.deciles,backgroundColor:colors,borderRadius:3}]},
      options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:(c)=>` ${c.raw}%`}}},
        scales:{x:{grid:{display:false}},y:{grid:{color:C.line},ticks:{callback:v=>v+"%"}}}}});
    $("#modelStats").innerHTML =
      `<div class="mini"><div class="v pos">+${d.top}%</div><div class="l">Top decile</div></div>`
      + `<div class="mini"><div class="v">+${d.market}%</div><div class="l">Market avg</div></div>`
      + `<div class="mini"><div class="v pos">+${(d.top-d.bottom).toFixed(2)}%</div><div class="l">Long-short edge</div></div>`;
  }catch(e){ /* backtest optional */ }
}

loadBacktest();
loadSaved();

/* ---------- live market strip ---------- */
function driftTag(x){
  if(!x.model_price || Math.abs(x.price - x.model_price) < 0.005) return "";
  const dr = x.price / x.model_price - 1;
  return `<span class="drift ${dr>=0?"pos":"neg"}" title="Change vs. the price in the saved model data">${pct(dr)}</span>`;
}

function spark(vals, up){
  if(!vals || vals.length < 2) return "";
  const w=54, h=20, mn=Math.min(...vals), mx=Math.max(...vals), rng=(mx-mn)||1;
  const pts = vals.map((v,i)=>`${(i/(vals.length-1)*w).toFixed(1)},${(h-2-((v-mn)/rng)*(h-4)).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${up?"#34D399":"#F87171"}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}

function ago(sec){
  if(sec == null) return "";
  if(sec < 90) return "just now";
  if(sec < 5400) return Math.round(sec/60) + " min ago";
  if(sec < 172800) return Math.round(sec/3600) + " h ago";
  return Math.round(sec/86400) + " days ago";
}

let marketTimer = null, marketTries = 0;
async function loadMarket(){
  const box = $("#ticker"); if(!box) return;
  let m = null;
  try{ const r = await fetch("/api/market", {cache:"no-store"}); if(r.ok) m = await r.json(); }catch(e){}
  if(!m){ box.className = "ticker offline"; $("#tickerStatusText").textContent = "Market feed unreachable"; return schedule(60000); }

  box.className = "ticker " + m.status;
  const label = {
    live:    `Live · updated ${ago(m.age_seconds)}`,
    closed:  `Market closed · last update ${ago(m.age_seconds)}`,
    stale:   `Delayed · last update ${ago(m.age_seconds)}`,
    loading: "Fetching live prices…",
    offline: "Live feed unavailable · showing saved data",
    off:     "Live data turned off"
  }[m.status] || "";
  $("#tickerStatusText").textContent = label;

  const track = $("#tickerTrack");
  if(m.indices && m.indices.length){
    track.innerHTML = m.indices.map(i=>{
      const up = i.pct >= 0;
      return `<div class="tk"><span class="tk-name">${i.name}</span>`
        + `<span class="tk-px">${i.price.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2})}</span>`
        + `<span class="tk-chg ${up?"up":"down"}">${up?"▲":"▼"} ${Math.abs(i.pct).toFixed(2)}%</span>${spark(i.spark, up)}</div>`;
    }).join("");
  } else if(m.status === "loading"){
    track.innerHTML = '<div class="tk-skel"></div><div class="tk-skel"></div><div class="tk-skel"></div>';
  } else {
    track.innerHTML = "";
  }

  const pf = $("#priceFreshness");
  if(pf){
    pf.innerHTML = m.prices_count > 0
      ? `stock prices are live${m.prices_date ? " (latest close " + m.prices_date + ")" : ""}<span class="live-note">LIVE</span>`
      : "prices update live when the market feed is available";
  }

  // first load on a cold server: poll quickly until data arrives, then every minute
  if(m.status === "loading" && marketTries++ < 12) return schedule(4000);
  schedule(m.market_open ? 60000 : 300000);
}
function schedule(ms){ clearTimeout(marketTimer); marketTimer = setTimeout(()=>{ if(!document.hidden) loadMarket(); else schedule(ms); }, ms); }
document.addEventListener("visibilitychange", ()=>{ if(!document.hidden) loadMarket(); });
loadMarket();

/* ---------- start ---------- */
syncUI();
loadProfile("Medium").then(() => { scheduleUpdate(); setTimeout(() => { loadProfile("Low").catch(()=>{}); loadProfile("High").catch(()=>{}); }, 600); })
  .catch(() => { $("#livePreview").innerHTML = '<div class="lp wide">Could not reach the server — retrying…</div>'; setTimeout(() => location.reload(), 5000); });
