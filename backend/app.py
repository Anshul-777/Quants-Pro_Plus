# =============================================================================
# backend/app.py  —  QuantVision Pro FastAPI Application
# =============================================================================
# Local:   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Render:  uvicorn app:app --host 0.0.0.0 --port $PORT
# Docs:    http://localhost:8000/docs
# Live UI: http://localhost:8000/
# =============================================================================

import os, json, logging, asyncio, time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from model import get_model
from trading_engine import get_engine, TICKERS

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="QuantVision Pro — AI Trading Intelligence",
    description="""
## QuantVision Pro  
**XGBoost + LightGBM Ensemble | Real-Time Polygon.io Data | 46 Econometric Features**

### What this system does
Every minute Polygon.io delivers a completed OHLCV bar for each stock. This
service feeds that bar into a rolling feature buffer, computes 46 financial
features (volatility, momentum, microstructure, order-flow, time-of-day), passes
them through a rank-averaged XGBoost + LightGBM ensemble, and broadcasts the
prediction probability and BUY/HOLD signal to every connected WebSocket client.

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Live HTML dashboard with candlestick charts |
| POST | `/start_auto` | Start live Polygon WebSocket stream |
| POST | `/start_simulation` | Start synthetic bar stream (for testing) |
| POST | `/stop` | Stop the engine |
| GET | `/status` | Full system + portfolio status |
| GET | `/portfolio` | Portfolio metrics |
| GET | `/history?n=100` | Last N prediction events |
| GET | `/model_info` | Model metadata and performance metrics |
| GET | `/features` | Live feature values per ticker |
| WS | `/ws` | Live JSON prediction stream |
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def on_startup():
    logger.info("QuantVision Pro starting up…")
    model = get_model()
    if model.loaded:
        m = model.info()
        logger.info(f"Model ready  τ={model.threshold:.4f}  features={m['n_features']}  type={m['model_type']}")
    else:
        logger.warning("No model loaded — upload model_v2.pkl to enable predictions")
    get_engine()
    logger.info("Engine initialised. POST /start_auto or /start_simulation to begin.")


@app.on_event("shutdown")
async def on_shutdown():
    get_engine().stop()


# =============================================================================
# HTML DASHBOARD
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantVision Pro — AI Trading Intelligence</title>

<!-- TradingView Lightweight Charts (free, open-source) -->
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>

<style>
/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:       #04060F;
  --card:     #0D1224;
  --card2:    #111830;
  --border:   #1A2040;
  --border-a: #00D4FF44;
  --txt:      #E8EDF8;
  --muted:    #8892AB;
  --blue:     #00D4FF;
  --green:    #00FF88;
  --red:      #FF4466;
  --yellow:   #FFB800;
  --purple:   #8B5CF6;
  --font-mono:"JetBrains Mono",monospace;
}
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;600&display=swap');
body{background:var(--bg);color:var(--txt);font-family:'Space Grotesk',sans-serif;font-size:13px;line-height:1.5;overflow-x:hidden}
h1,h2,h3{font-family:'Orbitron',sans-serif;letter-spacing:.05em}

/* ── Top bar ──────────────────────────────────────────────────────────────── */
#topbar{
  position:sticky;top:0;z-index:100;
  background:rgba(4,6,15,.95);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:24px;padding:10px 20px;
}
#topbar h1{font-size:16px;color:var(--blue);white-space:nowrap}
#topbar .spacer{flex:1}
.ws-dot{width:8px;height:8px;border-radius:50%;background:var(--red);display:inline-block;margin-right:6px;transition:background .4s}
.ws-dot.live{background:var(--green);box-shadow:0 0 8px var(--green)}
#equity-top{font-family:var(--font-mono);font-size:15px;font-weight:600;color:var(--green)}
#model-badge{background:#0a1535;border:1px solid var(--border-a);border-radius:6px;padding:3px 10px;font-size:11px;color:var(--blue)}

/* ── Layout ───────────────────────────────────────────────────────────────── */
#main{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px;max-width:1600px;margin:0 auto}
@media(max-width:1024px){#main{grid-template-columns:1fr}}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;transition:border-color .2s}
.card:hover{border-color:var(--border-a)}
.card-title{font-size:11px;font-family:'Orbitron',sans-serif;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.card-title .dot{width:6px;height:6px;border-radius:50%;background:var(--blue)}

/* ── Stock grid ────────────────────────────────────────────────────────────── */
#stock-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-bottom:16px}

/* ── Stock card ────────────────────────────────────────────────────────────── */
.sc{border-radius:12px;border:1px solid var(--border);background:var(--card);padding:14px;transition:border-color .3s,box-shadow .3s;cursor:default}
.sc.flash{border-color:var(--blue)!important;box-shadow:0 0 16px #00D4FF22}
.sc-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.sc-ticker{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;color:var(--txt)}
.signal-badge{font-size:10px;font-weight:700;padding:3px 8px;border-radius:20px;font-family:var(--font-mono);letter-spacing:.05em}
.sig-strong-buy{background:#00FF8833;color:var(--green);border:1px solid var(--green)}
.sig-buy       {background:#00cc6633;color:#00cc66;border:1px solid #00cc66}
.sig-watch     {background:#FFB80033;color:var(--yellow);border:1px solid var(--yellow)}
.sig-hold      {background:#1a204044;color:var(--muted);border:1px solid var(--border)}
.sc-price{font-family:var(--font-mono);font-size:22px;font-weight:600;color:var(--txt)}
.sc-change{font-size:12px;font-family:var(--font-mono);margin-left:8px}
.sc-change.up{color:var(--green)}.sc-change.dn{color:var(--red)}

/* Probability gauge */
.prob-row{display:flex;align-items:center;gap:10px;margin:10px 0 4px}
.prob-label{font-size:10px;color:var(--muted);width:60px;flex-shrink:0}
.prob-track{flex:1;height:8px;background:#1a2040;border-radius:4px;overflow:hidden}
.prob-fill{height:100%;border-radius:4px;transition:width .8s ease,background .8s ease}
.prob-val{font-family:var(--font-mono);font-size:12px;font-weight:600;width:44px;text-align:right}

/* Candlestick chart container */
.chart-wrap{height:160px;border-radius:8px;overflow:hidden;margin:10px 0;position:relative}
.chart-wrap .chart-loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px;background:var(--card)}

/* Mini stats row */
.sc-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px}
.stat-box{background:var(--card2);border-radius:6px;padding:6px 8px;text-align:center}
.stat-lbl{font-size:10px;color:var(--muted)}
.stat-val{font-family:var(--font-mono);font-size:12px;font-weight:600;margin-top:2px}

/* ── Right panel ────────────────────────────────────────────────────────────── */
#right{display:flex;flex-direction:column;gap:12px}

/* ── AI Prediction Panel ───────────────────────────────────────────────────── */
#ai-panel{background:linear-gradient(135deg,#050d1f,#0d1a3a);border:1px solid var(--border-a);border-radius:12px;padding:16px}
.ai-title{font-family:'Orbitron',sans-serif;font-size:13px;color:var(--blue);letter-spacing:.1em;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.ai-title-glow{text-shadow:0 0 12px var(--blue)}
.pulse-ring{width:10px;height:10px;border-radius:50%;background:var(--blue);box-shadow:0 0 0 0 var(--blue);animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 #00D4FF66}70%{box-shadow:0 0 0 8px transparent}100%{box-shadow:0 0 0 0 transparent}}

/* Bar chart for model features */
#feat-chart{margin-top:8px}
.feat-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.feat-name{font-size:10px;color:var(--muted);width:90px;flex-shrink:0;font-family:var(--font-mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.feat-bar-track{flex:1;height:6px;background:#1a2040;border-radius:3px;overflow:hidden}
.feat-bar-fill{height:100%;border-radius:3px;transition:width .6s ease}
.feat-num{font-size:10px;font-family:var(--font-mono);color:var(--muted);width:52px;text-align:right}

/* ── Trade log ──────────────────────────────────────────────────────────────── */
#trade-log-wrap{max-height:340px;overflow-y:auto;margin-top:8px}
#trade-log-wrap::-webkit-scrollbar{width:4px}
#trade-log-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.log-entry{display:flex;align-items:flex-start;gap:8px;padding:7px 0;border-bottom:1px solid var(--border);animation:slideIn .35s ease}
@keyframes slideIn{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
.log-time{font-family:var(--font-mono);font-size:10px;color:var(--muted);flex-shrink:0;padding-top:1px}
.log-body{flex:1}
.log-ticker-badge{font-size:10px;font-weight:700;padding:1px 6px;border-radius:3px;font-family:var(--font-mono)}
.log-entry.buy .log-ticker-badge{background:#00FF8820;color:var(--green);border:1px solid #00FF8850}
.log-entry.hold .log-ticker-badge{background:#1a204040;color:var(--muted);border:1px solid var(--border)}
.log-prob{font-family:var(--font-mono);font-size:11px;margin-top:2px}
.log-entry.buy .log-prob{color:var(--green)}
.log-entry.hold .log-prob{color:var(--muted)}
.log-prediction{font-size:10px;color:var(--muted);margin-top:1px}
.log-entry.buy .log-prediction{color:#00FF8888}

/* ── Portfolio stats ─────────────────────────────────────────────────────────── */
#port-stats{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.pstat{background:var(--card2);border-radius:8px;padding:10px 12px}
.pstat-lbl{font-size:10px;color:var(--muted);font-family:'Orbitron',sans-serif;letter-spacing:.06em}
.pstat-val{font-family:var(--font-mono);font-size:16px;font-weight:600;margin-top:4px}

/* ── Model metrics bar chart ─────────────────────────────────────────────────── */
.metric-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.metric-name{width:80px;font-size:11px;color:var(--muted);flex-shrink:0}
.metric-track{flex:1;height:12px;background:#1a2040;border-radius:6px;overflow:hidden;position:relative}
.metric-fill{height:100%;border-radius:6px;transition:width 1.2s ease}
.metric-val{width:44px;font-family:var(--font-mono);font-size:11px;font-weight:600;text-align:right}

/* ── Equity mini-chart ────────────────────────────────────────────────────────── */
#equity-wrap{height:120px;border-radius:8px;overflow:hidden;margin-top:8px}

/* ── Controls ─────────────────────────────────────────────────────────────────── */
#controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.btn{padding:8px 16px;border-radius:8px;border:none;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:600;transition:all .2s}
.btn-primary{background:var(--blue);color:#000}
.btn-primary:hover{filter:brightness(1.15)}
.btn-sim{background:#1a2040;color:var(--blue);border:1px solid var(--border-a)}
.btn-sim:hover{background:#1f2a55}
.btn-stop{background:#2a0a14;color:var(--red);border:1px solid #FF446630}
.btn-stop:hover{background:#3a0e1c}

/* ── Toasts ───────────────────────────────────────────────────────────────────── */
#toast{position:fixed;bottom:24px;right:24px;z-index:1000;display:flex;flex-direction:column;gap:8px}
.toast{background:#0d1224;border:1px solid var(--border-a);border-radius:8px;padding:10px 16px;font-size:12px;animation:toastIn .3s ease;max-width:300px}
@keyframes toastIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}

/* ── Util ─────────────────────────────────────────────────────────────────────── */
.green{color:var(--green)}.red{color:var(--red)}.blue{color:var(--blue)}.muted{color:var(--muted)}
.up{color:var(--green)}.dn{color:var(--red)}
canvas{display:block}
</style>
</head>
<body>

<!-- ── Top Bar ─────────────────────────────────────────────────────────────── -->
<div id="topbar">
  <h1>⚡ QUANTVISION PRO</h1>
  <div style="font-size:11px;color:var(--muted)">AI Trading Intelligence</div>
  <div class="spacer"></div>
  <div id="model-badge">V2 ENSEMBLE — XGBoost + LightGBM</div>
  <span><span class="ws-dot" id="ws-dot"></span><span id="ws-status" style="font-size:11px;color:var(--muted)">Connecting…</span></span>
  <div id="equity-top">—</div>
</div>

<div id="main">

<!-- ── Left Column ──────────────────────────────────────────────────────────── -->
<div id="left">

  <!-- Controls -->
  <div id="controls">
    <button class="btn btn-primary" onclick="startLive()">▶ Start Live Feed</button>
    <button class="btn btn-sim"     onclick="startSim()">⚙ Simulation Mode</button>
    <button class="btn btn-stop"    onclick="stopEngine()">■ Stop</button>
    <a href="/docs" target="_blank"><button class="btn btn-sim" style="text-decoration:none">📋 API Docs</button></a>
  </div>

  <!-- Stock cards grid -->
  <div id="stock-grid"></div>

  <!-- Model Performance metrics -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title"><span class="dot"></span>AI Model Performance Metrics</div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:12px">
      Computed on held-out test data (never seen during training). <span id="model-trained-at" class="muted"></span>
    </div>
    <div id="model-metrics">
      <div class="metric-row"><span class="metric-name muted">Loading…</span></div>
    </div>
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);font-size:11px;color:var(--muted)">
      <b class="blue">AUC-ROC</b> — probability of correctly ranking a true up-move above a non-move.
      <b class="green" style="margin-left:8px">Precision</b> — when the model says BUY, how often it's correct.
    </div>
  </div>

  <!-- Portfolio stats -->
  <div class="card">
    <div class="card-title"><span class="dot" style="background:var(--green)"></span>Portfolio — Demo Account</div>
    <div id="port-stats">
      <div class="pstat"><div class="pstat-lbl">Equity</div><div class="pstat-val" id="p-equity">$100,000.00</div></div>
      <div class="pstat"><div class="pstat-lbl">Total PnL</div><div class="pstat-val" id="p-pnl">—</div></div>
      <div class="pstat"><div class="pstat-lbl">Max Drawdown</div><div class="pstat-val red" id="p-dd">—</div></div>
      <div class="pstat"><div class="pstat-lbl">Bars Processed</div><div class="pstat-val" id="p-bars">0</div></div>
    </div>
    <!-- Equity mini chart using lightweight-charts -->
    <div class="card-title" style="margin-top:14px"><span class="dot" style="background:var(--purple)"></span>Equity Curve</div>
    <div id="equity-wrap"></div>
  </div>

</div><!-- /left -->

<!-- ── Right Panel ───────────────────────────────────────────────────────────── -->
<div id="right">

  <!-- AI Prediction Panel -->
  <div id="ai-panel">
    <div class="ai-title">
      <div class="pulse-ring"></div>
      <span class="ai-title-glow">AI PREDICTION ENGINE</span>
    </div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:14px">
      Live predictions from the XGBoost + LightGBM ensemble.<br>
      Threshold τ = <span id="tau-display" class="blue" style="font-family:var(--font-mono)">—</span>
      &nbsp;|&nbsp; Features = <span id="feat-count" class="blue" style="font-family:var(--font-mono)">—</span>
    </div>

    <!-- Per-ticker probability summary -->
    <div id="prob-summary"></div>

    <!-- Top feature importance bars -->
    <div class="card-title" style="margin-top:14px"><span class="dot" style="background:var(--purple)"></span>Top Model Features (SHAP-ranked)</div>
    <div id="feat-chart"></div>
  </div>

  <!-- AI Trade Signal Log -->
  <div class="card">
    <div class="card-title">
      <span class="dot" style="background:var(--green)"></span>
      AI Signal Log — Real-Time
      <span id="log-count" style="font-size:10px;color:var(--muted);margin-left:auto">0 events</span>
    </div>
    <div id="trade-log-wrap">
      <div id="trade-log"></div>
    </div>
  </div>

</div><!-- /right -->

</div><!-- /main -->

<!-- Toast container -->
<div id="toast"></div>

<script>
// =============================================================================
// STATE
// =============================================================================

const TICKERS     = ["AAPL","MSFT","NVDA","AMZN","TSLA"];
const BASE_URL    = window.location.origin;
const WS_URL      = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";

const state = {
  ws:         null,
  connected:  false,
  equityData: [],         // [{time, value}] for equity chart
  tickerData: {},         // {ticker: {price, open_price, prob, signal, label, rsi, bb, macd, vwap_dev, candles[]}}
  charts:     {},         // {ticker: {chart, series}} lightweight-chart instances
  equityChart:null,
  equitySeries:null,
  modelInfo:  null,
  tau:        0.5,
  logCount:   0,
  importances:{},
};

TICKERS.forEach(t => {
  state.tickerData[t] = {
    price: null, open_price: null,
    prob: null, signal: 0, label: "—",
    rsi: null, bb: null, macd: null, vwap_dev: null,
    candles: [],
  };
});

// =============================================================================
// INIT
// =============================================================================

window.addEventListener("DOMContentLoaded", async () => {
  buildStockCards();
  buildEquityChart();
  buildProbSummary();
  await loadModelInfo();
  await loadStatus();
  connectWS();
  // Poll status every 15s
  setInterval(loadStatus, 15000);
});

// =============================================================================
// STOCK CARDS + CHARTS
// =============================================================================

function buildStockCards() {
  const grid = document.getElementById("stock-grid");
  grid.innerHTML = "";
  TICKERS.forEach(t => {
    const div = document.createElement("div");
    div.className = "sc";
    div.id = `sc-${t}`;
    div.innerHTML = `
      <div class="sc-header">
        <span class="sc-ticker">${t}</span>
        <span class="signal-badge sig-hold" id="sig-${t}">LOADING</span>
      </div>
      <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:8px">
        <span class="sc-price" id="price-${t}">—</span>
        <span class="sc-change" id="chg-${t}"></span>
      </div>

      <!-- Candlestick chart -->
      <div class="chart-wrap" id="chart-wrap-${t}">
        <div class="chart-loading" id="chart-loading-${t}">Waiting for bars…</div>
      </div>

      <!-- Probability bar -->
      <div class="prob-row">
        <span class="prob-label">AI Prob</span>
        <div class="prob-track">
          <div class="prob-fill" id="prob-fill-${t}" style="width:0%;background:var(--muted)"></div>
        </div>
        <span class="prob-val" id="prob-val-${t}">—</span>
      </div>

      <!-- RSI bar -->
      <div class="prob-row" style="margin:3px 0">
        <span class="prob-label muted" style="font-size:9px">RSI-14</span>
        <div class="prob-track">
          <div class="prob-fill" id="rsi-fill-${t}" style="width:50%;background:var(--yellow)"></div>
        </div>
        <span class="prob-val muted" id="rsi-val-${t}" style="font-size:10px">—</span>
      </div>

      <!-- Mini stats -->
      <div class="sc-stats">
        <div class="stat-box">
          <div class="stat-lbl">BB Pos</div>
          <div class="stat-val muted" id="bb-${t}">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-lbl">MACD</div>
          <div class="stat-val muted" id="macd-${t}">—</div>
        </div>
        <div class="stat-box">
          <div class="stat-lbl">VWAP Δ</div>
          <div class="stat-val muted" id="vwap-${t}">—</div>
        </div>
      </div>`;
    grid.appendChild(div);
    initChart(t);
  });
}

function initChart(ticker) {
  const wrap = document.getElementById(`chart-wrap-${ticker}`);
  if (!wrap) return;

  const chart = LightweightCharts.createChart(wrap, {
    width:  wrap.clientWidth || 260,
    height: 160,
    layout: { background:{color:"#0D1224"}, textColor:"#8892AB" },
    grid:   { vertLines:{color:"#1A2040",style:1}, horzLines:{color:"#1A2040",style:1} },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor:"#1A2040", textColor:"#8892AB" },
    timeScale: { borderColor:"#1A2040", timeVisible:true, secondsVisible:false },
    handleScroll: false,
    handleScale:  false,
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor:       "#00FF88",
    downColor:     "#FF4466",
    borderVisible: false,
    wickUpColor:   "#00FF88",
    wickDownColor: "#FF4466",
  });

  // Volume histogram as separate series
  const volSeries = chart.addHistogramSeries({
    color:     "#00D4FF22",
    priceFormat:{ type:"volume" },
    priceScaleId:"volume",
    scaleMargins:{ top:0.85, bottom:0 },
  });

  state.charts[ticker] = { chart, candleSeries, volSeries };
}

function updateChart(ticker, bar) {
  const c = state.charts[ticker];
  if (!c) return;

  const ts = Math.floor(bar.ts / 1000);  // seconds
  const candle = { time:ts, open:bar.open, high:bar.high, low:bar.low, close:bar.close };
  const vol    = { time:ts, value:bar.volume, color: bar.close >= bar.open ? "#00FF8830" : "#FF446630" };

  try {
    c.candleSeries.update(candle);
    c.volSeries.update(vol);
  } catch(e) { /* ignore duplicate time errors */ }

  // Hide loading overlay on first real bar
  const loading = document.getElementById(`chart-loading-${ticker}`);
  if (loading) loading.style.display = "none";
}

// =============================================================================
// EQUITY CHART
// =============================================================================

function buildEquityChart() {
  const wrap = document.getElementById("equity-wrap");
  if (!wrap) return;

  const chart = LightweightCharts.createChart(wrap, {
    width:  wrap.clientWidth || 300,
    height: 120,
    layout: { background:{color:"transparent"}, textColor:"#8892AB" },
    grid:   { vertLines:{visible:false}, horzLines:{color:"#1A2040",style:1} },
    rightPriceScale:{ borderColor:"#1A2040" },
    timeScale:{ visible:false },
    handleScroll: false,
    handleScale:  false,
  });

  const areaSeries = chart.addAreaSeries({
    lineColor:  "#8B5CF6",
    topColor:   "#8B5CF644",
    bottomColor:"#8B5CF600",
    lineWidth:  2,
  });

  // Baseline reference
  areaSeries.applyOptions({ baseValue:{ type:"price", price:100000 } });
  state.equityChart  = chart;
  state.equitySeries = areaSeries;
}

function updateEquityChart(equity, ts) {
  if (!state.equitySeries) return;
  const t = Math.floor((ts || Date.now()) / 1000);
  try {
    state.equitySeries.update({ time:t, value: equity });
  } catch(e) {}
}

// =============================================================================
// PROB SUMMARY (right panel)
// =============================================================================

function buildProbSummary() {
  const el = document.getElementById("prob-summary");
  if (!el) return;
  el.innerHTML = TICKERS.map(t => `
    <div class="prob-row" style="margin-bottom:6px">
      <span class="prob-label" style="font-family:'Orbitron',sans-serif;font-size:10px;color:var(--txt)">${t}</span>
      <div class="prob-track">
        <div class="prob-fill" id="ps-fill-${t}" style="width:0%;background:var(--muted)"></div>
      </div>
      <span class="prob-val" id="ps-val-${t}">—</span>
      <span id="ps-sig-${t}" class="signal-badge sig-hold" style="margin-left:6px;font-size:9px">—</span>
    </div>`).join("");
}

// =============================================================================
// MODEL METRICS
// =============================================================================

async function loadModelInfo() {
  try {
    const r = await fetch(`${BASE_URL}/model_info`);
    if (!r.ok) return;
    const info = await r.json();
    state.modelInfo  = info;
    state.tau        = info.threshold || 0.5;
    state.importances = info.feature_importances || {};

    document.getElementById("tau-display").textContent  = state.tau.toFixed(4);
    document.getElementById("feat-count").textContent   = info.n_features || "—";
    if (info.trained_at) {
      document.getElementById("model-trained-at").textContent =
        "Trained: " + new Date(info.trained_at).toLocaleDateString();
    }

    // Render metrics bar chart
    const metrics = info.test_metrics || info.val_metrics || {};
    const container = document.getElementById("model-metrics");
    if (!container) return;

    const rows = [
      { name:"AUC-ROC",   key:"auc",       color:"#00D4FF", pct: true },
      { name:"Precision",  key:"precision",  color:"#00FF88", pct: true },
      { name:"Recall",     key:"recall",     color:"#FFB800", pct: true },
      { name:"F1 Score",   key:"f1",         color:"#8B5CF6", pct: true },
    ];

    container.innerHTML = rows.map(row => {
      const raw = metrics[row.key] ?? 0;
      const pct = Math.round(raw * 100);
      const disp = row.pct ? `${pct}%` : raw.toFixed(4);
      return `
        <div class="metric-row">
          <span class="metric-name" style="color:${row.color}">${row.name}</span>
          <div class="metric-track">
            <div class="metric-fill" style="width:${pct}%;background:${row.color}"></div>
          </div>
          <span class="metric-val" style="color:${row.color}">${disp}</span>
        </div>`;
    }).join("");

    // Feature importance bars
    buildFeatureBars(info.shap_top10 || []);

  } catch(e) {
    console.warn("loadModelInfo:", e);
  }
}

function buildFeatureBars(top10) {
  const el = document.getElementById("feat-chart");
  if (!el || !top10.length) return;

  // Assign equal placeholder importance since we only have names
  const colors = ["#00D4FF","#00FF88","#FFB800","#8B5CF6","#FF4466","#00D4FF","#00FF88","#FFB800","#8B5CF6","#FF4466"];
  el.innerHTML = top10.map((f, i) => {
    const pct  = Math.max(10, Math.round(100 - i * 8));
    return `
      <div class="feat-row">
        <span class="feat-name">${f}</span>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:${pct}%;background:${colors[i%colors.length]}"></div>
        </div>
        <span class="feat-num" style="color:${colors[i%colors.length]}">
          ${i === 0 ? "highest" : "#"+(i+1)}
        </span>
      </div>`;
  }).join("");
}

// =============================================================================
// STATUS POLL
// =============================================================================

async function loadStatus() {
  try {
    const r = await fetch(`${BASE_URL}/status`);
    if (!r.ok) return;
    const s = await r.json();

    updatePortfolio(s.engine?.portfolio || {});

    const ts = s.engine?.ticker_states || {};
    Object.entries(ts).forEach(([t, st]) => {
      if (st.last_close) updatePriceDisplay(t, st.last_close, null, null, null, null, null, null, st.last_prob, st.last_signal);
    });
  } catch(e) {}
}

// =============================================================================
// WEBSOCKET
// =============================================================================

let wsRetryDelay = 3000;

function connectWS() {
  if (state.ws && state.ws.readyState < 2) return;

  setWsStatus(false, "Connecting…");
  const ws = new WebSocket(WS_URL);
  state.ws = ws;

  ws.onopen = () => {
    setWsStatus(true, "Connected");
    wsRetryDelay = 3000;
  };

  ws.onmessage = e => {
    try { handleEvent(JSON.parse(e.data)); } catch(ex) {}
  };

  ws.onclose = ws.onerror = () => {
    setWsStatus(false, `Reconnecting in ${wsRetryDelay/1000}s…`);
    setTimeout(() => {
      wsRetryDelay = Math.min(wsRetryDelay * 1.5, 30000);
      connectWS();
    }, wsRetryDelay);
  };
}

function setWsStatus(live, text) {
  state.connected = live;
  document.getElementById("ws-dot").className    = "ws-dot" + (live ? " live" : "");
  document.getElementById("ws-status").textContent = text;
}

function handleEvent(ev) {
  switch(ev.event) {
    case "bar":       handleBar(ev);       break;
    case "warming":   handleWarming(ev);   break;
    case "connected": showToast("✓ " + ev.message, "blue"); break;
    case "disconnected": showToast("⚠ " + ev.error, "red"); break;
    case "welcome":
      if (ev.portfolio) updatePortfolio(ev.portfolio);
      showToast("Connected to QuantVision Pro", "blue");
      break;
  }
}

function handleBar(ev) {
  const t = ev.ticker;
  if (!TICKERS.includes(t)) return;

  const td = state.tickerData[t];
  if (!td.open_price && ev.close) td.open_price = ev.close;

  updateChart(t, ev);
  updatePriceDisplay(
    t, ev.close, ev.open, ev.high, ev.low,
    ev.rsi_14, ev.bb_pos, ev.macd, ev.vwap_dev,
    ev.probability, ev.signal, ev.signal_label
  );
  updatePortfolio(ev.portfolio);
  updateEquityChart(ev.portfolio?.equity, ev.ts);
  appendLogEntry(ev);

  // Flash the stock card
  const sc = document.getElementById(`sc-${t}`);
  if (sc) {
    sc.classList.add("flash");
    setTimeout(() => sc.classList.remove("flash"), 700);
  }
}

function handleWarming(ev) {
  const loading = document.getElementById(`chart-loading-${ev.ticker}`);
  if (loading) loading.textContent = `Warming buffer ${ev.buffer}/${ev.needed}…`;
}

// =============================================================================
// UI UPDATE FUNCTIONS
// =============================================================================

function updatePriceDisplay(t, close, open_, high, low, rsi, bb, macd, vwap_dev, prob, signal, label) {
  const td = state.tickerData[t];
  if (close) td.price = close;
  if (prob !== null && prob !== undefined) { td.prob = prob; td.signal = signal; td.label = label; }

  // Price
  const priceEl = document.getElementById(`price-${t}`);
  if (priceEl && close) priceEl.textContent = "$" + close.toFixed(2);

  // Change
  const chgEl = document.getElementById(`chg-${t}`);
  if (chgEl && td.open_price && close) {
    const pct = ((close - td.open_price) / td.open_price * 100).toFixed(2);
    chgEl.textContent = (pct >= 0 ? "+" : "") + pct + "%";
    chgEl.className = "sc-change " + (pct >= 0 ? "up" : "dn");
  }

  // Signal badge
  const sigEl = document.getElementById(`sig-${t}`);
  if (sigEl && label) {
    const cls = labelToClass(label);
    sigEl.className = "signal-badge " + cls;
    sigEl.textContent = label || "—";
  }

  // Probability bar
  if (prob !== null && prob !== undefined) {
    const pct  = Math.round(prob * 100);
    const fill = document.getElementById(`prob-fill-${t}`);
    const val  = document.getElementById(`prob-val-${t}`);
    if (fill) { fill.style.width = pct + "%"; fill.style.background = probColor(prob, state.tau); }
    if (val)  { val.textContent = pct + "%"; val.style.color = probColor(prob, state.tau); }

    // Prob summary (right panel)
    const psf = document.getElementById(`ps-fill-${t}`);
    const psv = document.getElementById(`ps-val-${t}`);
    const pss = document.getElementById(`ps-sig-${t}`);
    if (psf) { psf.style.width = pct + "%"; psf.style.background = probColor(prob, state.tau); }
    if (psv) { psv.textContent = pct + "%"; psv.style.color = probColor(prob, state.tau); }
    if (pss) { pss.className = "signal-badge " + labelToClass(label); pss.textContent = label || "—"; }
  }

  // RSI bar
  if (rsi !== null && rsi !== undefined) {
    const rf = document.getElementById(`rsi-fill-${t}`);
    const rv = document.getElementById(`rsi-val-${t}`);
    if (rf) {
      rf.style.width = rsi + "%";
      rf.style.background = rsi > 70 ? "var(--red)" : rsi < 30 ? "var(--green)" : "var(--yellow)";
    }
    if (rv) rv.textContent = rsi.toFixed(1);
  }

  // Mini stats
  if (bb !== null && bb !== undefined) {
    const el = document.getElementById(`bb-${t}`);
    if (el) { el.textContent = bb.toFixed(3); el.style.color = bb > 0 ? "var(--green)" : "var(--red)"; }
  }
  if (macd !== null && macd !== undefined) {
    const el = document.getElementById(`macd-${t}`);
    if (el) { el.textContent = macd.toFixed(4); el.style.color = macd > 0 ? "var(--green)" : "var(--red)"; }
  }
  if (vwap_dev !== null && vwap_dev !== undefined) {
    const el = document.getElementById(`vwap-${t}`);
    if (el) {
      el.textContent = (vwap_dev >= 0 ? "+" : "") + (vwap_dev * 100).toFixed(3) + "%";
      el.style.color = vwap_dev >= 0 ? "var(--green)" : "var(--red)";
    }
  }

  // Top-bar equity (first ticker only as proxy)
  const eq = state.tickerData[TICKERS[0]];
}

function updatePortfolio(p) {
  if (!p) return;
  const eq = p.equity ?? 100000;

  document.getElementById("equity-top").textContent = "$" + eq.toLocaleString("en-US", {maximumFractionDigits:2});

  const eqEl  = document.getElementById("p-equity");
  const pnlEl = document.getElementById("p-pnl");
  const ddEl  = document.getElementById("p-dd");
  const barEl = document.getElementById("p-bars");

  if (eqEl)  eqEl.textContent  = "$" + eq.toLocaleString("en-US", {maximumFractionDigits:2});
  if (barEl) barEl.textContent = (p.bars_processed || 0).toLocaleString();

  if (pnlEl && p.total_pnl !== undefined) {
    const pnl = p.total_pnl;
    pnlEl.textContent = (pnl >= 0 ? "+" : "") + (pnl * 100).toFixed(4) + "%";
    pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
  }
  if (ddEl && p.max_drawdown !== undefined) {
    ddEl.textContent = (p.max_drawdown * 100).toFixed(4) + "%";
  }
}

// =============================================================================
// AI SIGNAL LOG
// =============================================================================

function appendLogEntry(ev) {
  const log = document.getElementById("trade-log");
  if (!log) return;

  state.logCount++;
  document.getElementById("log-count").textContent = state.logCount + " events";

  const isBuy  = ev.signal === 1;
  const entry  = document.createElement("div");
  entry.className = "log-entry " + (isBuy ? "buy" : "hold");

  const time = ev.dt ? new Date(ev.dt).toLocaleTimeString() : "—";
  const prob = ev.probability_pct ?? (ev.probability ? (ev.probability * 100).toFixed(2) : "—");

  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <div class="log-body">
      <span class="log-ticker-badge">${ev.ticker}</span>
      <span style="font-size:10px;margin-left:6px;color:${isBuy ? "var(--green)" : "var(--muted)"}">
        ${isBuy ? "▲ AI PREDICTED UP" : "● HOLD"}
      </span>
      <div class="log-prob" style="margin-top:3px">
        p = ${typeof prob === 'number' ? prob.toFixed(2) : prob}%
        &nbsp;|&nbsp; $${ev.close ? ev.close.toFixed(2) : "—"}
        &nbsp;|&nbsp; <span style="color:${isBuy ? 'var(--green)' : 'var(--muted)'}">
          ${ev.signal_label || (isBuy ? "BUY" : "HOLD")}
        </span>
      </div>
      ${isBuy ? `<div class="log-prediction">Model confidence exceeds threshold τ = ${state.tau.toFixed(4)}</div>` : ""}
    </div>`;

  log.insertBefore(entry, log.firstChild);

  // Keep max 60 entries
  while (log.children.length > 60) log.removeChild(log.lastChild);
}

// =============================================================================
// UTILITY
// =============================================================================

function probColor(p, tau) {
  if (p >= 0.65)   return "var(--green)";
  if (p > tau)     return "#00cc66";
  if (p >= tau - 0.03) return "var(--yellow)";
  return "var(--red)";
}

function labelToClass(label) {
  switch(label) {
    case "STRONG BUY": return "sig-strong-buy";
    case "BUY":        return "sig-buy";
    case "WATCH":      return "sig-watch";
    default:           return "sig-hold";
  }
}

function showToast(msg, color="blue") {
  const container = document.getElementById("toast");
  const el = document.createElement("div");
  el.className = "toast";
  el.style.color = `var(--${color})`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// =============================================================================
// CONTROLS
// =============================================================================

async function startLive() {
  showToast("Starting live Polygon stream…", "blue");
  const r = await fetch(`${BASE_URL}/start_auto`, {method:"POST"});
  const d = await r.json();
  showToast(d.status === "started" ? "✓ Live stream active" : d.message, d.status === "started" ? "green" : "yellow");
}

async function startSim() {
  showToast("Starting simulation mode…", "blue");
  const r = await fetch(`${BASE_URL}/start_simulation`, {method:"POST"});
  const d = await r.json();
  showToast("✓ Simulation running — predictions will appear shortly", "green");
}

async function stopEngine() {
  await fetch(`${BASE_URL}/stop`, {method:"POST"});
  showToast("Engine stopped", "red");
}

// Resize charts on window resize
window.addEventListener("resize", () => {
  TICKERS.forEach(t => {
    const c = state.charts[t];
    const w = document.getElementById(`chart-wrap-${t}`);
    if (c && w) c.chart.applyOptions({width: w.clientWidth});
  });
  const eqW = document.getElementById("equity-wrap");
  if (state.equityChart && eqW) state.equityChart.applyOptions({width: eqW.clientWidth});
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return DASHBOARD_HTML


# =============================================================================
# REST ENDPOINTS
# =============================================================================

@app.post("/start_auto", summary="Start live Polygon WebSocket stream")
async def start_auto():
    """
    Connects to Polygon WebSocket, authenticates, subscribes to AM.AAPL,
    AM.MSFT, AM.NVDA, AM.AMZN, AM.TSLA, and begins processing minute bars.

    Each bar triggers:
    1. Rolling buffer update (80 bars per ticker)
    2. 49 raw features computed
    3. Features filtered to the 46 in features_v2.json
    4. RobustScaler transform
    5. Ensemble predict_proba (XGBoost × 0.7 + LightGBM × 0.3, rank-averaged)
    6. Signal emitted if probability > τ
    7. Demo PnL updated
    8. JSON broadcast to all /ws clients
    """
    engine = get_engine()
    model  = get_model()

    if engine._running:
        return {"status": "already_running", "message": "Engine is already streaming."}
    if not model.loaded:
        return {
            "status": "model_missing",
            "message": "Upload model_v2.pkl, scaler_v2.pkl, features_v2.json, threshold_v2.json to the backend directory."
        }
    engine.start(simulate=False)
    return {
        "status":    "started",
        "mode":      "live",
        "tickers":   TICKERS,
        "threshold": model.threshold,
        "n_features": len(model.features),
        "message":   f"Polygon WebSocket streaming started. Connect to /ws for live predictions.",
    }


@app.post("/start_simulation", summary="Start synthetic data stream for testing")
async def start_simulation():
    """
    Generates synthetic random-walk bars for all 5 tickers.
    Predictions, PnL logic, and WebSocket broadcast all work identically to live mode.
    Use this when the market is closed or for testing the prediction pipeline.
    """
    engine = get_engine()
    if engine._running:
        engine.stop()
        await asyncio.sleep(0.3)
    engine.start(simulate=True)
    return {
        "status":  "started",
        "mode":    "simulation",
        "tickers": TICKERS,
        "message": "Synthetic bars running. Connect to /ws for live predictions.",
    }


@app.post("/stop", summary="Stop the trading engine")
async def stop_engine():
    """Stops the Polygon WebSocket (or simulation) and all bar processing."""
    get_engine().stop()
    return {"status": "stopped"}


@app.get("/status", summary="Full system and portfolio status")
async def get_status():
    """Returns engine state, model info, per-ticker buffer state, and portfolio metrics."""
    engine = get_engine()
    model  = get_model()
    return {
        "engine":    engine.status(),
        "model":     model.info(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/portfolio", summary="Portfolio metrics")
async def portfolio():
    """Equity, total PnL, max drawdown, trade count, and per-ticker PnL breakdown."""
    e = get_engine()
    return {
        "portfolio": asdict_safe(e.portfolio),
        "ticker_pnl": {
            t: {"realized_pnl": round(s.realized_pnl, 6), "n_trades": s.n_trades,
                "last_signal": s.last_signal, "last_prob": s.last_prob}
            for t, s in e.states.items()
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/history", summary="Recent prediction log")
async def history(n: int = Query(default=100, le=500, ge=1,
                                  description="Number of recent bar events to return")):
    """
    Returns the last N bar events. Each event contains the full OHLCV bar,
    model prediction (probability, signal, signal_label), feature values (top 10),
    RSI-14, BB position, MACD, VWAP deviation, and portfolio snapshot.
    """
    engine = get_engine()
    recent = engine.log[-n:]
    return {
        "count":     len(recent),
        "events":    recent,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model_info", summary="Model metadata and performance metrics")
async def model_info():
    """
    Returns:
    - Model type (EnsembleModel or XGBClassifier)
    - Number of features and feature list
    - Optimized threshold τ
    - Validation and test set metrics (AUC, Precision, Recall, F1, Sharpe)
    - SHAP top-10 feature names
    - Training timestamp
    """
    model = get_model()
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Upload model artifacts.")
    info = model.info()
    info["feature_importances"] = model.feature_importances()
    return info


@app.get("/features", summary="Live feature values per ticker")
async def features():
    """
    Computes the current feature vector for each ticker from its rolling buffer.
    Shows all 49 raw features plus which ones the model actually uses.
    """
    from trading_engine import compute_features_raw
    engine = get_engine()
    model  = get_model()
    result = {}
    for ticker, state in engine.states.items():
        raw   = compute_features_raw(state)
        ready = raw is not None
        result[ticker] = {
            "ready":        ready,
            "buffer_size":  len(state.buf),
            "buffer_needed": 62,
            "features_raw": {k: round(v, 6) for k, v in raw.items()} if raw else {},
            "features_model": {
                k: round(raw.get(k, 0.0), 6)
                for k in model.features
            } if raw and model.loaded else {},
            "last_prob":    state.last_prob,
            "last_signal":  state.last_signal,
        }
    return {"tickers": result, "timestamp": datetime.utcnow().isoformat()}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    ## Live WebSocket Prediction Feed

    Connect to receive real-time JSON events every minute per ticker.

    ### Bar event (one per ticker per minute)
    ```json
    {
      "event":           "bar",
      "ticker":          "AAPL",
      "dt":              "2024-07-15T14:32:00+00:00",
      "open":            220.10,
      "high":            220.55,
      "low":             219.88,
      "close":           220.45,
      "volume":          182400,
      "probability":     0.6821,
      "probability_pct": 68.21,
      "signal":          1,
      "signal_label":    "STRONG BUY",
      "threshold":       0.45,
      "rsi_14":          62.4,
      "bb_pos":          0.34,
      "macd":            0.0241,
      "vwap_dev":        0.00123,
      "portfolio":       {"equity": 102341.20, "total_pnl": 0.02341, ...},
      "features":        {"vwap_dev": 0.00123, "bb_pos": 0.34, ...}
    }
    ```

    ### Warming event (buffer not yet full)
    ```json
    { "event": "warming", "ticker": "AAPL", "buffer": 45, "needed": 62 }
    ```

    Send "ping" to receive a status pong.
    """
    await websocket.accept()
    engine = get_engine()
    await engine.register(websocket)

    try:
        await websocket.send_text(json.dumps({
            "event":     "welcome",
            "tickers":   TICKERS,
            "running":   engine._running,
            "portfolio": asdict_safe(engine.portfolio),
            "model":     get_model().info() if get_model().loaded else None,
            "timestamp": datetime.utcnow().isoformat(),
            "message":   "Connected. POST /start_auto or /start_simulation to begin streaming.",
        }, default=str))

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25)
                if data.strip().lower() in ("ping", "status"):
                    await websocket.send_text(json.dumps({
                        "event":     "pong",
                        "status":    engine.status(),
                        "timestamp": datetime.utcnow().isoformat(),
                    }, default=str))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "event":     "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "running":   engine._running,
                    "equity":    round(engine.portfolio.equity, 2),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS client error: {e}")
    finally:
        await engine.unregister(websocket)


# =============================================================================
# HELPERS
# =============================================================================

def asdict_safe(obj):
    """Safe dataclass-to-dict that handles non-serialisable fields."""
    from dataclasses import asdict
    try:
        return asdict(obj)
    except Exception:
        return obj.__dict__


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
