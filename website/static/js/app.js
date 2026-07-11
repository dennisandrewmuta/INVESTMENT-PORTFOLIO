/* Smart Investment Portfolio System — frontend logic */
"use strict";

const $ = (s) => document.querySelector(s);
const money = (v) => "$" + Math.round(v).toLocaleString("en-US");
const pct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
const pctFlat = (v) => (v * 100).toFixed(0) + "%";

// theme colors for charts
const C = { navy:"#0B2545", green:"#0FB981", greenSoft:"rgba(15,185,129,.16)",
  ice:"#9DB4CE", red:"#DC2626", line:"#E4EAF1", muted:"#5B6B7F" };
Chart.defaults.font.family = "'IBM Plex Mono','Inter',monospace";
Chart.defaults.font.size = 11;
Chart.defaults.color = C.muted;

let state = { risk:"Medium", lastPortfolio:null };
const charts = {};
function destroy(k){ if(charts[k]){ charts[k].destroy(); delete charts[k]; } }

/* ---------- inputs ---------- */
$("#risk").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if(!b) return;
  $("#risk").querySelectorAll("button").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); state.risk = b.dataset.risk;
});
$("#horizon").addEventListener("input", (e) => {
  $("#hzVal").textContent = e.target.value + " days";
});

$("#generate").addEventListener("click", generate);
$("#saveBtn").addEventListener("click", savePortfolio);
$("#clearBtn").addEventListener("click", clearSaved);

/* ---------- generate ---------- */
async function generate(){
  const btn = $("#generate");
  btn.disabled = true; btn.textContent = "Running simulations…";
  try{
    const body = { amount:+$("#amount").value, risk:state.risk, horizon:+$("#horizon").value };
    const res = await fetch("/api/portfolio", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const data = await res.json();
    if(!res.ok){ alert(data.error || "Something went wrong."); return; }
    state.lastPortfolio = data;
    renderAll(data);
    $("#results").hidden = false;
    $("#results").querySelectorAll(".block").forEach((b,i)=>{ b.classList.remove("reveal"); void b.offsetWidth; b.style.animationDelay=(i*60)+"ms"; b.classList.add("reveal"); });
    $("#results").scrollIntoView({behavior:"smooth"});
  } catch(err){ alert("Could not reach the server. Is it running?"); }
  finally{ btn.disabled=false; btn.textContent="Generate portfolio"; }
}

/* ---------- render ---------- */
function renderAll(d){
  const inp = d.inputs;
  $("#recSub").textContent = `${inp.risk}-risk · ${money(inp.amount)} · ${inp.horizon}-day horizon · as of ${d.as_of}`;

  // recommendation stats
  $("#recStats").innerHTML = stat(money(inp.amount),"You invest","accent")
    + stat(d.n_stocks,"Stocks held","accent")
    + stat(money(d.invested),"Deployed","accent")
    + stat(money(d.cash_left),"Cash left over","accent");

  // holdings table
  let h = "<thead><tr><th>Ticker</th><th>Weight</th><th>Allocation</th><th>Shares</th><th>Price</th><th>Score</th></tr></thead><tbody>";
  d.holdings.forEach(x=>{ h += `<tr><td class="tkr">${x.ticker}</td><td>${(x.weight*100).toFixed(1)}%</td>`
    + `<td>${money(x.allocation)}</td><td>${x.shares}</td><td>$${x.price.toFixed(2)}</td><td>${x.score.toFixed(3)}</td></tr>`; });
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
  const f = d.forecast;
  $("#fcSub").textContent = `Where ${money(inp.amount)} could be after ${inp.horizon} trading days.`;
  $("#fcStats").innerHTML = stat(money(f.expected_value),"Expected value", f.expected_return>=0?"pos":"neg")
    + stat(money(f.best_case),"Best case (95%)","pos")
    + stat(money(f.worst_case),"Worst case (5%)","neg")
    + stat(pctFlat(f.prob_profit),"Chance of profit","accent");
  drawCone(f, inp.amount);
  drawHist(f);

  renderConclusion(d);
  loadSaved();
}

function stat(v,l,cls=""){ return `<div class="stat"><div class="v num ${cls}">${v}</div><div class="l">${l}</div></div>`; }

/* ---------- charts ---------- */
function drawAllocation(holdings){
  destroy("alloc");
  const top = holdings.slice(0,8);
  const others = holdings.slice(8).reduce((s,x)=>s+x.allocation,0);
  const labels = top.map(x=>x.ticker).concat(others>0?["Other"]:[]);
  const vals = top.map(x=>x.allocation).concat(others>0?[others]:[]);
  const palette = ["#0FB981","#13315C","#1C4270","#38B2AC","#2C7A7B","#4A6FA5","#7FA8D4","#0A8F63","#9DB4CE"];
  charts.alloc = new Chart($("#allocChart"),{type:"doughnut",
    data:{labels,datasets:[{data:vals,backgroundColor:palette,borderColor:"#fff",borderWidth:2}]},
    options:{cutout:"62%",plugins:{legend:{position:"right",labels:{boxWidth:10,padding:8,font:{size:10}}},
      tooltip:{callbacks:{label:(c)=>` ${c.label}: ${money(c.raw)}`}}}}});
}

function drawValueCurve(curve, amount){
  destroy("value");
  charts.value = new Chart($("#valueChart"),{type:"line",
    data:{labels:curve.map((_,i)=>i),datasets:[{data:curve,borderColor:C.navy,borderWidth:2,
      fill:true,backgroundColor:"rgba(11,37,69,.06)",pointRadius:0,tension:.15}]},
    options:{plugins:{legend:{display:false}},scales:{
      x:{display:false},
      y:{ticks:{callback:v=>"$"+(v/1000).toFixed(0)+"k"},grid:{color:C.line}}}}});
}

function drawCone(f, amount){
  destroy("cone");
  charts.cone = new Chart($("#coneChart"),{type:"line",
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
  destroy("hist");
  charts.hist = new Chart($("#histChart"),{type:"bar",
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
