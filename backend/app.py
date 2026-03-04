# =============================================================================
# backend/app.py  —  QuantVision Pro V2 (Fixed)
# =============================================================================

import os, json, logging, asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from model import get_model
from trading_engine import get_engine, TICKERS, market_status, asdict_safe

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("app")

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="QuantVision Pro V2",
    description="XGBoost + LightGBM Ensemble | Real-Time AI Predictions",
    version="2.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def on_startup():
    model  = get_model()
    engine = get_engine()
    ms     = market_status()
    logger.info(f"Market status: {ms['message']}")

    # Auto-start simulation if env says so, OR if market is closed
    auto_sim = os.environ.get("AUTO_SIM", "true").lower() == "true"
    if auto_sim:
        engine.start(simulate=True)
        logger.info("AUTO_SIM=true — simulation started on startup")

    if model.loaded:
        logger.info(f"Model ready  τ={model.threshold:.4f}  features={len(model.features)}")
    else:
        logger.warning("No model loaded — upload artifacts to enable real predictions")


@app.on_event("shutdown")
async def on_shutdown():
    get_engine().stop_hard()


# =============================================================================
# HTML DASHBOARD
# =============================================================================

DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantVision Pro V2</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#04060F;--card:#0D1224;--card2:#111830;--border:#1A2040;--ba:#00D4FF44;--txt:#E8EDF8;--muted:#8892AB;--blue:#00D4FF;--green:#00FF88;--red:#FF4466;--yellow:#FFB800;--purple:#8B5CF6;--mono:"JetBrains Mono",monospace}
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;600&display=swap');
body{background:var(--bg);color:var(--txt);font-family:'Space Grotesk',sans-serif;font-size:13px}
h1,h2,h3{font-family:'Orbitron',sans-serif}

/* Topbar */
#top{position:sticky;top:0;z-index:100;background:rgba(4,6,15,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px;padding:10px 20px}
#top h1{font-size:15px;color:var(--blue)}
.spacer{flex:1}
.dot{width:8px;height:8px;border-radius:50%;background:var(--red);display:inline-block;transition:background .4s}
.dot.live{background:var(--green);box-shadow:0 0 8px var(--green)}
.badge{background:#0a1535;border:1px solid var(--ba);border-radius:6px;padding:3px 10px;font-size:11px;color:var(--blue)}

/* Market banner */
#market-banner{padding:8px 20px;font-size:12px;text-align:center;border-bottom:1px solid var(--border)}
#market-banner.open{background:#00FF8812;color:var(--green)}
#market-banner.closed{background:#FF446612;color:var(--red)}

/* Layout */
#main{display:grid;grid-template-columns:1fr 320px;gap:14px;padding:14px;max-width:1500px;margin:0 auto}
@media(max-width:1000px){#main{grid-template-columns:1fr}}

/* Card */
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px;transition:border-color .2s}
.card:hover{border-color:var(--ba)}
.ctitle{font-size:10px;font-family:'Orbitron',sans-serif;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.ctitle .cdot{width:6px;height:6px;border-radius:50%;background:var(--blue)}

/* Stock grid */
#sgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px;margin-bottom:14px}
.sc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px;transition:border-color .3s}
.sc.flash{border-color:var(--blue)!important}
.sctop{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.scticker{font-family:'Orbitron',sans-serif;font-size:17px;font-weight:900}
.sbadge{font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;font-family:var(--mono)}
.s-strong-buy{background:#00FF8833;color:var(--green);border:1px solid var(--green)}
.s-buy{background:#00cc6633;color:#00cc66;border:1px solid #00cc66}
.s-watch{background:#FFB80033;color:var(--yellow);border:1px solid var(--yellow)}
.s-hold{background:#1a204040;color:var(--muted);border:1px solid var(--border)}
.s-sell{background:#FF446633;color:var(--red);border:1px solid var(--red)}
.scprice{font-family:var(--mono);font-size:20px;font-weight:600}
.scchg{font-size:11px;font-family:var(--mono);margin-left:8px}
.scchg.up{color:var(--green)}.scchg.dn{color:var(--red)}
.bar-row{display:flex;align-items:center;gap:8px;margin:6px 0}
.bar-lbl{font-size:10px;color:var(--muted);width:58px;flex-shrink:0}
.bar-track{flex:1;height:7px;background:#1a2040;border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .7s ease,background .7s ease}
.bar-val{font-family:var(--mono);font-size:11px;width:40px;text-align:right}
.chart-wrap{height:150px;border-radius:8px;overflow:hidden;margin:8px 0;position:relative}
.chart-loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px;background:var(--card)}
.scstats{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:8px}
.sstat{background:var(--card2);border-radius:6px;padding:5px 7px;text-align:center}
.sstat-l{font-size:9px;color:var(--muted)}
.sstat-v{font-family:var(--mono);font-size:11px;margin-top:2px}

/* Confidence bar */
.conf-row{display:flex;align-items:center;gap:8px;margin:4px 0}
.conf-grade{font-size:9px;font-family:var(--mono);width:68px;flex-shrink:0}

/* Controls */
#controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.btn{padding:7px 14px;border-radius:8px;border:none;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:600;transition:all .2s}
.btn-p{background:var(--blue);color:#000}.btn-p:hover{filter:brightness(1.15)}
.btn-s{background:#1a2040;color:var(--blue);border:1px solid var(--ba)}.btn-s:hover{background:#1f2a55}
.btn-r{background:#2a0a14;color:var(--red);border:1px solid #FF446630}.btn-r:hover{background:#3a0e1c}
.btn-g{background:#0a2a14;color:var(--green);border:1px solid #00FF8830}.btn-g:hover{background:#0e3a1c}
.ai-toggle{display:flex;align-items:center;gap:10px;background:var(--card);border:1px solid var(--ba);border-radius:10px;padding:8px 14px;margin-bottom:10px}
.toggle-switch{position:relative;width:44px;height:22px}
.toggle-switch input{opacity:0;width:0;height:0}
.toggle-slider{position:absolute;cursor:pointer;inset:0;background:#1a2040;border-radius:22px;transition:.4s}
.toggle-slider:before{content:"";position:absolute;width:18px;height:18px;left:2px;bottom:2px;background:#8892AB;border-radius:50%;transition:.4s}
input:checked + .toggle-slider{background:var(--green)}
input:checked + .toggle-slider:before{transform:translateX(22px);background:#000}

/* Log */
#log-wrap{max-height:320px;overflow-y:auto}
#log-wrap::-webkit-scrollbar{width:3px}
#log-wrap::-webkit-scrollbar-thumb{background:var(--border)}
.le{display:flex;gap:8px;padding:6px 0;border-bottom:1px solid var(--border);animation:sIn .3s ease}
@keyframes sIn{from{opacity:0;transform:translateY(-5px)}to{opacity:1;transform:none}}
.le-t{font-family:var(--mono);font-size:10px;color:var(--muted);flex-shrink:0}
.le-b{flex:1;font-size:11px}
.le.buy .le-b{color:var(--green)}.le.sell .le-b{color:var(--red)}.le.hold .le-b{color:var(--muted)}

/* Metrics */
.mrow{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.mname{width:78px;font-size:11px;color:var(--muted);flex-shrink:0}
.mtrack{flex:1;height:11px;background:#1a2040;border-radius:5px;overflow:hidden}
.mfill{height:100%;border-radius:5px;transition:width 1s ease}
.mval{width:40px;font-family:var(--mono);font-size:11px;text-align:right}

/* Portfolio */
#pstats{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}
.ps{background:var(--card2);border-radius:8px;padding:9px 11px}
.ps-l{font-size:10px;color:var(--muted);font-family:'Orbitron',sans-serif;letter-spacing:.05em}
.ps-v{font-family:var(--mono);font-size:15px;font-weight:600;margin-top:3px}

/* Toast */
#toast{position:fixed;bottom:20px;right:20px;z-index:1000;display:flex;flex-direction:column;gap:7px}
.toast{background:#0d1224;border:1px solid var(--ba);border-radius:8px;padding:9px 14px;font-size:12px;animation:tIn .3s ease;max-width:280px}
@keyframes tIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}

/* Equity */
#eq-wrap{height:100px;border-radius:8px;overflow:hidden;margin-top:8px}

.green{color:var(--green)}.red{color:var(--red)}.blue{color:var(--blue)}.muted{color:var(--muted)}
</style>
</head>
<body>

<div id="top">
  <h1>⚡ QUANTVISION PRO V2</h1>
  <span class="badge" id="model-badge">Loading model…</span>
  <div class="spacer"></div>
  <span><span class="dot" id="wsdot"></span>
  <span id="wsstat" style="font-size:11px;color:var(--muted)">Connecting…</span></span>
  <span style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--green)" id="eq-top">—</span>
</div>

<div id="market-banner" class="closed">Checking market hours…</div>

<div id="main">
<div id="left">

  <div id="controls">
    <button class="btn btn-p" onclick="startLive()">▶ Live Feed</button>
    <button class="btn btn-s" onclick="startSim()">⚙ Simulation</button>
    <button class="btn btn-r" onclick="gracefulStop()">⏹ Graceful Stop</button>
    <button class="btn btn-r" onclick="hardStop()" style="opacity:.7">✕ Force Stop</button>
    <a href="/docs" target="_blank" style="text-decoration:none"><button class="btn btn-s">📋 API</button></a>
  </div>

  <!-- AI Trading Toggle -->
  <div class="ai-toggle">
    <div>
      <div style="font-size:12px;font-weight:600;color:var(--blue)">⚡ AI Auto-Trade</div>
      <div style="font-size:10px;color:var(--muted);margin-top:2px" id="ai-status-text">
        Disabled — AI will only predict, not execute trades
      </div>
    </div>
    <div class="spacer"></div>
    <label class="toggle-switch">
      <input type="checkbox" id="ai-toggle-cb" onchange="toggleAI(this.checked)">
      <span class="toggle-slider"></span>
    </label>
  </div>

  <div id="sgrid"></div>

  <!-- Model Metrics -->
  <div class="card" style="margin-bottom:14px">
    <div class="ctitle"><span class="cdot"></span>AI Model Performance</div>
    <div id="model-metrics"></div>
  </div>

  <!-- Portfolio -->
  <div class="card">
    <div class="ctitle"><span class="cdot" style="background:var(--green)"></span>Portfolio</div>
    <div id="pstats">
      <div class="ps"><div class="ps-l">Equity</div><div class="ps-v" id="p-eq">$100,000</div></div>
      <div class="ps"><div class="ps-l">PnL</div><div class="ps-v" id="p-pnl">—</div></div>
      <div class="ps"><div class="ps-l">Max DD</div><div class="ps-v red" id="p-dd">—</div></div>
      <div class="ps"><div class="ps-l">Wins / Losses</div><div class="ps-v" id="p-wl">—</div></div>
    </div>
    <div class="ctitle" style="margin-top:12px"><span class="cdot" style="background:var(--purple)"></span>Equity Curve</div>
    <div id="eq-wrap"></div>
  </div>

</div><!-- /left -->

<div id="right">

  <!-- All-ticker probability summary -->
  <div class="card" style="margin-bottom:12px">
    <div class="ctitle" style="margin-bottom:12px">
      <span class="cdot"></span>AI PREDICTIONS — ALL TICKERS
      <a href="/predict" target="_blank" style="margin-left:auto;font-size:10px;color:var(--blue)">JSON →</a>
    </div>
    <div id="prob-summary"></div>
  </div>

  <!-- Signal Log -->
  <div class="card">
    <div class="ctitle">
      <span class="cdot" style="background:var(--green)"></span>Live Signal Log
      <span id="log-count" style="font-size:10px;color:var(--muted);margin-left:auto">0</span>
    </div>
    <div id="log-wrap"><div id="the-log"></div></div>
  </div>

</div>
</div>

<div id="toast"></div>

<script>
const BASE = window.location.origin;
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
const TICKERS = ["AAPL","MSFT","NVDA","AMZN","TSLA"];

const S = {
  ws: null, connected: false, tau: 0.5,
  tickerData: Object.fromEntries(TICKERS.map(t => [t, {price:null,openPrice:null,prob:null,signal:0,label:"—"}])),
  charts: {}, equityChart: null, equitySeries: null,
  logCount: 0,
};

window.addEventListener("DOMContentLoaded", async () => {
  buildStockCards(); buildEquityChart(); buildProbSummary();
  await loadModelInfo(); await loadStatus();
  connectWS();
  setInterval(loadStatus, 20000);
  updateMarketBanner();
  setInterval(updateMarketBanner, 60000);
});

async function updateMarketBanner() {
  try {
    const r = await fetch(`${BASE}/market_status`);
    const d = await r.json();
    const el = document.getElementById("market-banner");
    el.textContent = d.message;
    el.className = d.open ? "open" : "closed";
    if (!d.open) el.textContent += " — Simulation mode active";
  } catch(e) {}
}

function buildStockCards() {
  const g = document.getElementById("sgrid"); g.innerHTML = "";
  TICKERS.forEach(t => {
    const d = document.createElement("div");
    d.className = "sc"; d.id = `sc-${t}`;
    d.innerHTML = `
      <div class="sctop">
        <span class="scticker">${t}</span>
        <span class="sbadge s-hold" id="sig-${t}">LOADING</span>
      </div>
      <div style="display:flex;align-items:baseline;gap:4px;margin-bottom:6px">
        <span class="scprice" id="price-${t}">—</span>
        <span class="scchg" id="chg-${t}"></span>
      </div>
      <div class="chart-wrap" id="cw-${t}">
        <div class="chart-loading" id="cl-${t}">Warming buffer…</div>
      </div>
      <!-- AI Probability -->
      <div class="bar-row">
        <span class="bar-lbl">AI Prob</span>
        <div class="bar-track"><div class="bar-fill" id="pf-${t}" style="width:0%;background:var(--muted)"></div></div>
        <span class="bar-val" id="pv-${t}" style="font-family:var(--mono)">—</span>
      </div>
      <!-- Confidence -->
      <div class="bar-row">
        <span class="bar-lbl muted" style="font-size:9px">Confidence</span>
        <div class="bar-track"><div class="bar-fill" id="cf-${t}" style="width:0%;background:var(--purple)"></div></div>
        <span class="conf-grade muted" id="cg-${t}">—</span>
      </div>
      <!-- RSI -->
      <div class="bar-row">
        <span class="bar-lbl muted" style="font-size:9px">RSI-14</span>
        <div class="bar-track"><div class="bar-fill" id="rf-${t}" style="width:50%;background:var(--yellow)"></div></div>
        <span class="bar-val muted" id="rv-${t}" style="font-size:10px">—</span>
      </div>
      <div class="scstats">
        <div class="sstat"><div class="sstat-l">BB Pos</div><div class="sstat-v muted" id="bb-${t}">—</div></div>
        <div class="sstat"><div class="sstat-l">MACD</div><div class="sstat-v muted" id="macd-${t}">—</div></div>
        <div class="sstat"><div class="sstat-l">VWAP Δ</div><div class="sstat-v muted" id="vwap-${t}">—</div></div>
      </div>
      <div style="margin-top:8px;font-size:10px;color:var(--muted);font-style:italic;min-height:28px" id="reason-${t}"></div>`;
    g.appendChild(d);
    initChart(t);
  });
}

function initChart(t) {
  const wrap = document.getElementById(`cw-${t}`); if (!wrap) return;
  const chart = LightweightCharts.createChart(wrap, {
    width:wrap.clientWidth||260, height:150,
    layout:{background:{color:"#0D1224"},textColor:"#8892AB"},
    grid:{vertLines:{color:"#1A2040",style:1},horzLines:{color:"#1A2040",style:1}},
    rightPriceScale:{borderColor:"#1A2040"},
    timeScale:{borderColor:"#1A2040",timeVisible:true,secondsVisible:false},
    handleScroll:false, handleScale:false,
  });
  const cs = chart.addCandlestickSeries({upColor:"#00FF88",downColor:"#FF4466",borderVisible:false,wickUpColor:"#00FF88",wickDownColor:"#FF4466"});
  const vs = chart.addHistogramSeries({color:"#00D4FF22",priceFormat:{type:"volume"},priceScaleId:"vol",scaleMargins:{top:0.85,bottom:0}});
  S.charts[t] = {chart, cs, vs};
}

function updateChart(t, bar) {
  const c = S.charts[t]; if (!c) return;
  const ts = Math.floor(bar.ts/1000);
  try {
    c.cs.update({time:ts, open:bar.open, high:bar.high, low:bar.low, close:bar.close});
    c.vs.update({time:ts, value:bar.volume, color:bar.close>=bar.open?"#00FF8830":"#FF446630"});
  } catch(e) {}
  const cl = document.getElementById(`cl-${t}`); if (cl) cl.style.display="none";
}

function buildEquityChart() {
  const wrap = document.getElementById("eq-wrap"); if (!wrap) return;
  const chart = LightweightCharts.createChart(wrap, {
    width:wrap.clientWidth||300, height:100,
    layout:{background:{color:"transparent"},textColor:"#8892AB"},
    grid:{vertLines:{visible:false},horzLines:{color:"#1A2040",style:1}},
    rightPriceScale:{borderColor:"#1A2040"}, timeScale:{visible:false},
    handleScroll:false, handleScale:false,
  });
  S.equitySeries = chart.addAreaSeries({lineColor:"#8B5CF6",topColor:"#8B5CF644",bottomColor:"#8B5CF600",lineWidth:2});
  S.equityChart  = chart;
}

function buildProbSummary() {
  const el = document.getElementById("prob-summary"); if (!el) return;
  el.innerHTML = TICKERS.map(t => `
    <div class="bar-row" style="margin-bottom:5px">
      <span class="bar-lbl" style="font-family:'Orbitron',sans-serif;font-size:10px;color:var(--txt)">${t}</span>
      <div class="bar-track"><div class="bar-fill" id="ps-f-${t}" style="width:0%;background:var(--muted)"></div></div>
      <span class="bar-val" id="ps-v-${t}">—</span>
      <span class="sbadge s-hold" id="ps-s-${t}" style="margin-left:5px;font-size:9px">—</span>
    </div>`).join("");
}

async function loadModelInfo() {
  try {
    const r = await fetch(`${BASE}/model_info`); if (!r.ok) return;
    const d = await r.json();
    S.tau = d.threshold || 0.5;
    document.getElementById("model-badge").textContent = `V2 ENSEMBLE · τ=${S.tau.toFixed(4)} · ${d.n_features||"?"}F`;
    const m = d.test_metrics || d.val_metrics || {};
    const container = document.getElementById("model-metrics");
    const rows = [
      {name:"AUC-ROC",key:"auc",c:"#00D4FF"},{name:"Precision",key:"precision",c:"#00FF88"},
      {name:"Recall",key:"recall",c:"#FFB800"},{name:"F1",key:"f1",c:"#8B5CF6"},
    ];
    if (container) container.innerHTML = rows.map(row => {
      const raw = m[row.key]||0; const pct = Math.round(raw*100);
      return `<div class="mrow"><span class="mname" style="color:${row.c}">${row.name}</span>
        <div class="mtrack"><div class="mfill" style="width:${pct}%;background:${row.c}"></div></div>
        <span class="mval" style="color:${row.c}">${pct}%</span></div>`;
    }).join("");
  } catch(e) {}
}

async function loadStatus() {
  try {
    const r = await fetch(`${BASE}/status`); if (!r.ok) return;
    const d = await r.json();
    const p = d.engine?.portfolio; if (p) updatePortfolio(p);
    const ts = d.engine?.ticker_states || {};
    Object.entries(ts).forEach(([t,st]) => {
      if (st.last_close) updatePrice(t, st.last_close, null, st.last_prob, st.last_signal);
    });
  } catch(e) {}
}

function connectWS() {
  if (S.ws && S.ws.readyState < 2) return;
  setWS(false, "Connecting…");
  const ws = new WebSocket(WS_URL); S.ws = ws;
  ws.onopen = () => { setWS(true, "Connected"); };
  ws.onmessage = e => { try { handle(JSON.parse(e.data)); } catch(ex){} };
  ws.onclose = ws.onerror = () => {
    setWS(false, "Reconnecting…");
    setTimeout(connectWS, 4000);
  };
}

function setWS(live, txt) {
  S.connected = live;
  document.getElementById("wsdot").className = "dot"+(live?" live":"");
  document.getElementById("wsstat").textContent = txt;
}

function handle(ev) {
  switch(ev.event) {
    case "bar":           handleBar(ev); break;
    case "warming":       handleWarming(ev); break;
    case "warmup_complete": showToast("✓ "+ev.message,"green"); break;
    case "auto_simulation": showToast("⚡ "+ev.message,"yellow"); break;
    case "ai_stop_loss":  showToast("🔴 "+ev.message,"red"); break;
    case "ai_take_profit": showToast("💚 "+ev.message,"green"); break;
    case "ai_profit_prompt": showToast("🏆 "+ev.message,"green"); break;
    case "engine_stopped": showToast("✓ "+ev.message,"blue"); break;
    case "connected":     showToast("✓ "+ev.message,"blue"); break;
    case "welcome":       if(ev.portfolio) updatePortfolio(ev.portfolio); break;
  }
}

function handleBar(ev) {
  const t = ev.ticker; if (!TICKERS.includes(t)) return;
  if (!S.tickerData[t].openPrice && ev.close) S.tickerData[t].openPrice = ev.close;
  updateChart(t, ev);
  updatePrice(t, ev.close, ev.open, ev.probability, ev.signal, ev.signal_label,
              ev.confidence, ev.rsi_14, ev.bb_pos, ev.macd, ev.vwap_dev, ev.reason);
  updatePortfolio(ev.portfolio);
  if (ev.portfolio?.equity && ev.ts) {
    try { S.equitySeries?.update({time:Math.floor(ev.ts/1000), value:ev.portfolio.equity}); } catch(e){}
  }
  document.getElementById(`eq-top`).textContent = ev.portfolio?.equity
    ? "$"+(ev.portfolio.equity).toLocaleString("en-US",{maximumFractionDigits:2}) : "";
  appendLog(ev);
  const sc = document.getElementById(`sc-${t}`);
  if (sc) { sc.classList.add("flash"); setTimeout(()=>sc.classList.remove("flash"),600); }
}

function handleWarming(ev) {
  const cl = document.getElementById(`cl-${ev.ticker}`);
  if (cl) cl.textContent = `Warming ${ev.buffer}/${ev.needed}…`;
}

function updatePrice(t, close, open_, prob, signal, label, conf, rsi, bb, macd, vwap_dev, reason) {
  if (close) {
    const el = document.getElementById(`price-${t}`);
    if (el) el.textContent = "$"+close.toFixed(2);
    const td = S.tickerData[t];
    const chg = document.getElementById(`chg-${t}`);
    if (chg && td.openPrice && close) {
      const pct = ((close - td.openPrice)/td.openPrice*100).toFixed(2);
      chg.textContent = (pct>=0?"+":"")+pct+"%";
      chg.className = "scchg "+(pct>=0?"up":"dn");
    }
  }
  if (label) {
    const se = document.getElementById(`sig-${t}`);
    if (se) { se.className = "sbadge "+labelCls(label); se.textContent = label; }
  }
  if (prob !== null && prob !== undefined) {
    const pct = Math.round(prob*100);
    const fill = document.getElementById(`pf-${t}`);
    const val  = document.getElementById(`pv-${t}`);
    const col  = probCol(prob, S.tau);
    if (fill) { fill.style.width=pct+"%"; fill.style.background=col; }
    if (val)  { val.textContent=pct+"%"; val.style.color=col; }
    const psf = document.getElementById(`ps-f-${t}`);
    const psv = document.getElementById(`ps-v-${t}`);
    const pss = document.getElementById(`ps-s-${t}`);
    if (psf) { psf.style.width=pct+"%"; psf.style.background=col; }
    if (psv) { psv.textContent=pct+"%"; psv.style.color=col; }
    if (pss && label) { pss.className="sbadge "+labelCls(label); pss.textContent=label; }
    S.tickerData[t].prob=prob; S.tickerData[t].signal=signal;
  }
  if (conf) {
    const cf = document.getElementById(`cf-${t}`);
    const cg = document.getElementById(`cg-${t}`);
    if (cf) { cf.style.width=(conf.score_pct||0)+"%"; }
    if (cg) { cg.textContent=(conf.grade||"—"); }
  }
  if (rsi !== null && rsi !== undefined) {
    const rf = document.getElementById(`rf-${t}`); const rv = document.getElementById(`rv-${t}`);
    if (rf) { rf.style.width=rsi+"%"; rf.style.background=rsi>70?"var(--red)":rsi<30?"var(--green)":"var(--yellow)"; }
    if (rv) rv.textContent=rsi.toFixed(1);
  }
  const setMini = (id, val, fmt) => {
    const el = document.getElementById(id); if (!el || val===null||val===undefined) return;
    el.textContent = fmt(val); el.style.color = val>=0?"var(--green)":"var(--red)";
  };
  setMini(`bb-${t}`, bb, v=>v.toFixed(3));
  setMini(`macd-${t}`, macd, v=>v.toFixed(4));
  setMini(`vwap-${t}`, vwap_dev, v=>(v>=0?"+":"")+(v*100).toFixed(3)+"%");
  if (reason) {
    const re = document.getElementById(`reason-${t}`);
    if (re) re.textContent = "→ "+reason;
  }
}

function updatePortfolio(p) {
  if (!p) return;
  const eq = p.equity||100000;
  const set = (id, txt, col) => { const e=document.getElementById(id); if(e){e.textContent=txt; if(col)e.style.color=col;} };
  set("p-eq",  "$"+eq.toLocaleString("en-US",{maximumFractionDigits:2}));
  if (p.total_pnl !== undefined) {
    const pnl = p.total_pnl;
    set("p-pnl", (pnl>=0?"+":"")+(pnl*100).toFixed(4)+"%", pnl>=0?"var(--green)":"var(--red)");
  }
  if (p.max_drawdown !== undefined) set("p-dd", (p.max_drawdown*100).toFixed(3)+"%");
  if (p.n_wins !== undefined) set("p-wl", `${p.n_wins||0}W / ${p.n_losses||0}L`,
    (p.n_wins||0)>=(p.n_losses||0)?"var(--green)":"var(--red)");
}

function appendLog(ev) {
  const log = document.getElementById("the-log"); if (!log) return;
  S.logCount++;
  document.getElementById("log-count").textContent = S.logCount;
  const isBuy = ev.signal===1; const isSell = ev.signal===-1;
  const entry = document.createElement("div");
  entry.className = "le "+(isBuy?"buy":isSell?"sell":"hold");
  const time = ev.dt ? new Date(ev.dt).toLocaleTimeString() : "—";
  const prob = ev.probability_pct ?? (ev.probability ? (ev.probability*100).toFixed(2) : "—");
  const conf = ev.confidence?.score_pct ?? "—";
  entry.innerHTML = `<span class="le-t">${time}</span>
    <div class="le-b">
      <b>${ev.ticker}</b> · ${ev.signal_label||"—"} · p=${prob}% · conf=${conf}%
      ${ev.confidence?.grade ? `<span class="muted"> (${ev.confidence.grade})</span>` : ""}
      <br><span class="muted" style="font-size:10px">$${ev.close?.toFixed(2)||"—"} ${ev.reason?'· '+ev.reason.slice(0,60):''}…</span>
    </div>`;
  log.insertBefore(entry, log.firstChild);
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

function probCol(p, tau) {
  if (p>=0.65)      return "var(--green)";
  if (p>tau)        return "#00cc66";
  if (p>=tau-0.04)  return "var(--yellow)";
  return "var(--red)";
}
function labelCls(l) {
  return {
    "STRONG BUY":"s-strong-buy","BUY":"s-buy","WATCH":"s-watch",
    "SELL":"s-sell","STRONG SELL":"s-sell",
  }[l] || "s-hold";
}

function showToast(msg, col="blue") {
  const c = document.getElementById("toast");
  const el = document.createElement("div"); el.className="toast";
  el.style.color = `var(--${col})`; el.textContent = msg;
  c.appendChild(el); setTimeout(()=>el.remove(), 6000);
}

async function startLive()    { const r=await fetch(`${BASE}/start_auto`,{method:"POST"}); const d=await r.json(); showToast(d.status==="started"?"✓ Live stream started":d.message, d.status==="started"?"green":"yellow"); }
async function startSim()     { const r=await fetch(`${BASE}/start_simulation`,{method:"POST"}); showToast("✓ Simulation started — warming buffers…","green"); }
async function gracefulStop() { const r=await fetch(`${BASE}/stop_graceful`,{method:"POST"}); const d=await r.json(); showToast(d.message||"Graceful stop initiated","yellow"); }
async function hardStop()     { await fetch(`${BASE}/stop`,{method:"POST"}); showToast("Engine force-stopped","red"); }
async function toggleAI(on) {
  const r = await fetch(`${BASE}/ai_trading`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({enable:on})});
  const d = await r.json();
  const txt = document.getElementById("ai-status-text");
  if (txt) txt.textContent = on
    ? `AI trading ENABLED — auto-buying/selling with SL=${d.stop_loss_pct}% / TP=${d.take_profit_pct}%`
    : "Disabled — AI will only predict, not execute trades";
  showToast(on ? "⚡ AI Auto-Trade ENABLED" : "AI Auto-Trade disabled", on?"green":"yellow");
}

window.addEventListener("resize", () => {
  TICKERS.forEach(t => {
    const c=S.charts[t]; const w=document.getElementById(`cw-${t}`);
    if(c&&w) c.chart.applyOptions({width:w.clientWidth});
  });
  const ew=document.getElementById("eq-wrap");
  if(S.equityChart&&ew) S.equityChart.applyOptions({width:ew.clientWidth});
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return DASHBOARD


# =============================================================================
# REST ENDPOINTS
# =============================================================================

@app.get("/market_status", summary="NYSE market hours status")
async def get_market_status():
    return market_status()


@app.post("/start_auto", summary="Start live Polygon WebSocket stream")
async def start_auto():
    engine = get_engine(); model = get_model()
    if not model.loaded:
        return {"status":"model_missing",
                "message":"Upload model_v2.pkl, scaler_v2.pkl, features_v2.json, threshold_v2.json"}
    ms = market_status()
    if engine._running:
        return {"status":"already_running", "market": ms}
    engine.start(simulate=False)
    return {"status":"started", "mode":"live", "tickers":TICKERS,
            "market": ms,
            "warning": "No data will arrive until NYSE opens (9:30 AM ET)" if not ms["open"] else None}


@app.post("/start_simulation", summary="Start synthetic bar stream (works 24/7)")
async def start_simulation():
    engine = get_engine()
    if engine._running:
        engine.stop_hard()
        await asyncio.sleep(0.3)
    engine.start(simulate=True)
    return {"status":"started", "mode":"simulation",
            "message":"Simulation started. Buffer warms in ~3s then predictions begin."}


@app.post("/stop", summary="Force stop (ignores open positions)")
async def stop_hard():
    get_engine().stop_hard()
    return {"status":"stopped"}


@app.post("/stop_graceful", summary="Graceful stop — waits for all AI positions to close with profit")
async def stop_graceful():
    engine = get_engine()
    engine.request_graceful_stop()
    open_p = {
        t: {"entry": s.position.entry_price, "current": s.last_bar.close if s.last_bar else None}
        for t, s in engine.states.items()
        if s.position and not s.position.closed
    }
    if open_p:
        return {"status":"draining",
                "message": f"Drain mode active. Waiting for {len(open_p)} position(s) to reach ≥0.5% profit.",
                "open_positions": open_p}
    return {"status":"stopped", "message":"No open positions — stopped immediately."}


@app.post("/ai_trading", summary="Enable or disable AI auto-trading")
async def set_ai_trading(body: Dict = Body(default={"enable": True})):
    enable = body.get("enable", True)
    get_engine().enable_ai_trading(enable)
    from trading_engine import STOP_LOSS_PCT, TAKE_PROFIT_PCT, MIN_CONFIDENCE
    return {
        "ai_trading":       enable,
        "stop_loss_pct":    round(STOP_LOSS_PCT * 100, 2),
        "take_profit_pct":  round(TAKE_PROFIT_PCT * 100, 2),
        "min_confidence":   MIN_CONFIDENCE,
        "message": f"AI auto-trading {'ENABLED' if enable else 'disabled'}",
    }


@app.get("/predict", summary="Current predictions for ALL tickers (cross-asset portfolio view)")
async def predict_all():
    """
    Returns current AI prediction for every ticker.
    Each prediction includes: probability, signal, confidence score, reason.
    Also returns a portfolio-level recommendation: which single stock is best to buy right now.
    """
    engine = get_engine()
    preds  = engine.get_all_predictions()

    # Portfolio-level recommendation
    ready   = {t: p for t, p in preds.items() if p.get("ready") and p.get("probability") is not None}
    best_buy = max(ready.items(), key=lambda x: x[1]["probability"], default=(None, {}))
    best_sell = min(ready.items(), key=lambda x: x[1]["probability"], default=(None, {}))

    return {
        "predictions":       preds,
        "portfolio_signal": {
            "best_buy":   best_buy[0],
            "best_sell":  best_sell[0],
            "best_buy_prob":   round(best_buy[1].get("probability",0)*100, 2) if best_buy[0] else None,
            "best_sell_prob":  round(best_sell[1].get("probability",0)*100, 2) if best_sell[0] else None,
            "recommendation":  (
                f"BUY {best_buy[0]} (p={best_buy[1].get('probability_pct',0):.1f}%)"
                if best_buy[0] and best_buy[1].get("signal",0)==1
                else "HOLD — no high-confidence BUY signal currently"
            ),
        },
        "model_version": "v2",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/predict/{ticker}", summary="Prediction for a single ticker")
async def predict_ticker(ticker: str):
    """Single-ticker prediction with probability, confidence, reason, and version."""
    ticker = ticker.upper()
    engine = get_engine(); model = get_model()

    if ticker not in TICKERS:
        raise HTTPException(400, f"Unknown ticker '{ticker}'. Valid: {TICKERS}")

    state = engine.states[ticker]
    if not state.last_features:
        return {"ready": False, "ticker": ticker,
                "message": f"Buffer still warming ({len(state.buf)}/62 bars). Try /start_simulation first."}
    if not model.loaded:
        return {"ready": False, "ticker": ticker, "message": "Model not loaded — upload artifacts."}

    result = model.predict(state.last_features)
    result["ticker"]      = ticker
    result["last_close"]  = state.last_bar.close if state.last_bar else None
    result["in_position"] = bool(state.position and not state.position.closed)
    result["timestamp"]   = datetime.utcnow().isoformat()
    return result


@app.get("/status")
async def status():
    return {"engine": get_engine().status(), "model": get_model().info(),
            "market": market_status(), "timestamp": datetime.utcnow().isoformat()}


@app.get("/portfolio")
async def portfolio():
    e = get_engine()
    return {"portfolio": asdict_safe(e.portfolio),
            "ticker_pnl": {
                t: {"pnl": round(s.realized_pnl,6), "n_trades": s.n_trades,
                    "last_signal": s.last_signal, "in_position": bool(s.position and not s.position.closed)}
                for t, s in e.states.items()
            }, "timestamp": datetime.utcnow().isoformat()}


@app.get("/history")
async def history(n: int = Query(100, le=500, ge=1)):
    e = get_engine()
    return {"count": min(n, len(e.log)), "events": e.log[-n:],
            "timestamp": datetime.utcnow().isoformat()}


@app.get("/model_info")
async def model_info():
    model = get_model()
    if not model.loaded:
        return {"loaded": False, "message": "Upload model_v2.pkl + scaler_v2.pkl + features_v2.json + threshold_v2.json"}
    info = model.info()
    info["feature_importances"] = model.feature_importances()
    return info


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = get_engine()
    await engine.register(websocket)
    try:
        await websocket.send_text(json.dumps({
            "event": "welcome", "tickers": TICKERS, "running": engine._running,
            "market": market_status(), "portfolio": asdict_safe(engine.portfolio),
            "model": get_model().info() if get_model().loaded else None,
            "message": "Connected. Use /start_simulation (24/7) or /start_auto (market hours only).",
            "timestamp": datetime.utcnow().isoformat(),
        }, default=str))
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25)
                cmd  = json.loads(data) if data.startswith("{") else {"action": data.strip().lower()}
                if cmd.get("action") in ("ping","status"):
                    await websocket.send_text(json.dumps({
                        "event":"pong","status":engine.status(),"timestamp":datetime.utcnow().isoformat()
                    }, default=str))
                elif cmd.get("action") == "ai_continue":
                    engine.enable_ai_trading(True)
                    await websocket.send_text(json.dumps({"event":"ai_resumed","message":"AI trading resumed"}))
                elif cmd.get("action") == "ai_stop":
                    engine.enable_ai_trading(False)
                    await websocket.send_text(json.dumps({"event":"ai_paused","message":"AI trading paused"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "event":"heartbeat","running":engine._running,
                    "equity":round(engine.portfolio.equity,2),"timestamp":datetime.utcnow().isoformat()
                }))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await engine.unregister(websocket)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
