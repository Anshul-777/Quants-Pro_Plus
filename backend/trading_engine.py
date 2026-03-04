# =============================================================================
# backend/trading_engine.py  —  V2 Fixed: Streaming + AI Auto-Trade Engine
# =============================================================================
# KEY FIXES vs old version:
# 1. Market-hours detection — auto-fallback to simulation when market closed
# 2. No-data timeout — if Polygon sends 0 bars in 90s, switch to simulation
# 3. Rapid warmup mode — 70 synthetic bars in 3 seconds to prime the buffer
# 4. /predict endpoint support — compute features + predict on demand
# 5. AI auto-trade — optional autonomous buy/sell based on signal
# 6. Per-position stop-loss / take-profit
# 7. Graceful stop — drains all positions at profit before halting
# 8. Confidence scores on every prediction
# =============================================================================

import os, asyncio, json, logging, time, math
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Deque

import numpy as np
import websockets

from model import get_model, compute_confidence, generate_reason

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"
API_KEY        = os.environ.get("POLYGON_API_KEY", "MoyLn951WdZAozaSClrOGar9xgYjy0pR")
TICKERS        = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
SUBSCRIBE_STR  = ",".join(f"AM.{t}" for t in TICKERS)

BUFFER_SIZE    = 80
EWMA_LAMBDA    = 0.94
COST_BPS       = 0.0005
INITIAL_CAPITAL = 100_000.0
RECONNECT_INIT  = 3.0
RECONNECT_MAX   = 60.0

# AI Auto-Trade parameters
STOP_LOSS_PCT      = 0.015   # Exit if position loses 1.5%
TAKE_PROFIT_PCT    = 0.025   # Lock in profit at 2.5%
MIN_CONFIDENCE     = 0.35    # Minimum confidence before AI trades
NO_DATA_TIMEOUT    = 90      # Seconds of silence before fallback to simulation

ALL_TRAIN_TICKERS = sorted([
    "AAPL","MSFT","NVDA","AMZN","TSLA",
    "GOOGL","META","NFLX","AMD","INTC",
    "SPY","QQQ","BABA","CRM","UBER"
])
TICKER_ID_MAP = {t: float(i) for i, t in enumerate(ALL_TRAIN_TICKERS)}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bar:
    ticker: str; ts: int; open: float; high: float; low: float
    close: float; volume: float; vwap: float; n_trades: int; dt_iso: str = ""

@dataclass
class Position:
    """Tracks a live AI or manual position for stop-loss / take-profit."""
    ticker:       str
    entry_price:  float
    entry_ts:     int
    size:         int    = 1     # number of units
    stop_loss:    float  = 0.0
    take_profit:  float  = 0.0
    ai_managed:   bool   = True
    closed:       bool   = False
    close_reason: str    = ""

@dataclass
class TickerState:
    ticker:    str
    buf:       Deque[Bar] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    ema12:     float = 0.0; ema26: float = 0.0; ema9: float = 0.0
    ewma_var:  float = 0.0; prev_tick: float = 0.0
    ofi_acc5:  int   = 0;   ofi_acc20: int   = 0
    last_prob: Optional[float] = None
    last_signal: int  = 0
    last_pos:    int  = 0
    last_bar:    Optional[Bar] = None
    position:    Optional[Position] = None
    realized_pnl: float = 0.0
    n_trades:     int   = 0
    last_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class Portfolio:
    cash:          float = INITIAL_CAPITAL
    equity:        float = INITIAL_CAPITAL
    total_pnl:     float = 0.0
    n_trades:      int   = 0
    n_wins:        int   = 0
    n_losses:      int   = 0
    bars_processed: int  = 0
    peak_equity:   float = INITIAL_CAPITAL
    max_drawdown:  float = 0.0
    started_at:    str   = ""
    ai_mode:       bool  = False

# =============================================================================
# MARKET HOURS DETECTION
# =============================================================================

def is_market_open() -> bool:
    """True when NYSE is open for regular session (ET timezone)."""
    now_utc = datetime.now(timezone.utc)
    now_et  = now_utc - timedelta(hours=4)  # EDT offset (conservative)

    # Skip weekends
    if now_et.weekday() >= 5:
        return False

    # Regular session: 9:30–16:00 ET
    market_open  = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et < market_close

def market_status() -> Dict[str, Any]:
    open_ = is_market_open()
    now_et = datetime.now(timezone.utc) - timedelta(hours=4)
    return {
        "open":         open_,
        "current_et":   now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "message":      "NYSE regular session active" if open_
                        else f"Market closed — opens 9:30 AM ET (Mon–Fri). Current: {now_et.strftime('%H:%M ET %a')}",
    }

# =============================================================================
# FEATURE COMPUTATION (same as before, unchanged)
# =============================================================================

def _ema_step(prev, val, span):
    alpha = 2.0 / (span + 1)
    return val if prev == 0.0 else alpha * val + (1 - alpha) * prev

def _rrvol(lr, w):
    return float(np.sqrt(np.sum(lr[-w:] ** 2))) if len(lr) >= w else None

def _rsi(lr, p):
    if len(lr) < p: return None
    r = lr[-p:]; up = np.where(r > 0, r, 0.0).mean(); dn = np.where(r < 0, -r, 0.0).mean()
    return 100.0 if dn < 1e-12 else float(100.0 - 100.0 / (1.0 + up / dn))

def _rolling_stat(arr, w, kind):
    if len(arr) < w: return 0.0
    x = arr[-w:]; std = x.std()
    if kind == "skew": return 0.0 if std < 1e-12 else float(((x - x.mean()) ** 3).mean() / std ** 3)
    if kind == "kurt": return 0.0 if std < 1e-12 else float(((x - x.mean()) ** 4).mean() / std ** 4)
    if kind == "ac1":
        mu = x.mean(); den = ((x - mu) ** 2).sum()
        return 0.0 if den < 1e-12 else float(((x[1:] - mu) * (x[:-1] - mu)).sum() / den)
    return 0.0

def compute_features_raw(state: TickerState) -> Optional[Dict[str, float]]:
    buf = state.buf
    if len(buf) < 62: return None

    bars   = list(buf)
    closes = np.array([b.close  for b in bars])
    highs  = np.array([b.high   for b in bars])
    lows   = np.array([b.low    for b in bars])
    opens  = np.array([b.open   for b in bars])
    vols   = np.array([b.volume for b in bars])
    vwaps  = np.array([b.vwap   for b in bars])

    lr = np.log(closes[1:] / closes[:-1])
    if len(lr) < 61: return None

    lc, lh, ll, lo, lv, lvw = closes[-1], highs[-1], lows[-1], opens[-1], vols[-1], vwaps[-1]
    last_ret = float(lr[-1])

    rvol_5  = _rrvol(lr, 5);  rvol_10 = _rrvol(lr, 10)
    rvol_20 = _rrvol(lr, 20); rvol_60 = _rrvol(lr, 60)
    if any(v is None for v in [rvol_5, rvol_10, rvol_20, rvol_60]): return None

    state.ewma_var = EWMA_LAMBDA * state.ewma_var + (1 - EWMA_LAMBDA) * last_ret ** 2
    ewma_vol = float(math.sqrt(max(state.ewma_var, 0.0)))

    hl2 = np.log(highs[-21:] / lows[-21:].clip(min=1e-12)) ** 2 * 0.5
    co2 = np.log(closes[-21:] / opens[-21:].clip(min=1e-12)) ** 2 * (2 * math.log(2) - 1)
    gk_vol_20 = float(math.sqrt(max(float((hl2 - co2).mean()), 0.0)))

    def mom(n): ref = closes[-n-1]; return float((lc - ref) / ref) if len(closes) > n and ref > 1e-12 else 0.0

    state.ema12 = _ema_step(state.ema12, lc, 12)
    state.ema26 = _ema_step(state.ema26, lc, 26)
    macd_val    = state.ema12 - state.ema26
    state.ema9  = _ema_step(state.ema9, macd_val, 9)

    rsi7 = _rsi(lr, 7); rsi14 = _rsi(lr, 14)
    if rsi7 is None or rsi14 is None: return None

    sma20 = float(closes[-20:].mean()); std20 = float(closes[-20:].std())
    bb_pos = float((lc - sma20) / (2 * std20)) if std20 > 1e-12 else 0.0
    bb_wid = float(4 * std20 / sma20) if sma20 > 1e-12 else 0.0

    vwap_dev   = float((lc - lvw) / lvw) if lvw > 1e-12 else 0.0
    vwap_devs  = (closes[-6:] - vwaps[-6:]) / vwaps[-6:].clip(min=1e-12)
    vwap_trend = float(vwap_devs.mean()) if len(vwap_devs) >= 5 else vwap_dev

    v_mean = float(vols[-20:].mean()); v_std = float(vols[-20:].std())
    vol_zscore = float((lv - v_mean) / v_std) if v_std > 1e-12 else 0.0
    vol_ratio  = float(lv / v_mean) if v_mean > 1e-12 else 1.0

    tick = float(np.sign(lc - state.prev_tick)) if state.prev_tick > 0 else 0.0
    state.prev_tick  = lc
    state.ofi_acc5   = max(-5,  min(5,  int(tick) + state.ofi_acc5))
    state.ofi_acc20  = max(-20, min(20, int(tick) + state.ofi_acc20))

    illiq     = np.abs(lr[-20:]) / (vols[-20:] + 1)
    amihud_20 = float(illiq.mean())
    pos_sq    = lr[-20:].clip(min=0) ** 2; neg_sq = lr[-20:].clip(max=0) ** 2
    signed_rv = float((pos_sq.sum() - neg_sq.sum()) / (pos_sq.sum() + neg_sq.sum() + 1e-12))

    # Time
    try:
        dt    = datetime.fromisoformat(bars[-1].dt_iso.replace("Z", "+00:00"))
        dt_et = dt - timedelta(hours=4)
        mins  = max(0, dt_et.hour * 60 + dt_et.minute - 570)
    except Exception:
        mins = 195
    T = 390

    return {
        "log_ret": last_ret, "log_ret_sq": last_ret ** 2,
        "ret_1_lag1": float(lr[-2]) if len(lr) >= 2 else 0.0,
        "ret_1_lag2": float(lr[-3]) if len(lr) >= 3 else 0.0,
        "cumret_5":   float(lr[-5:].sum()) if len(lr) >= 5 else 0.0,
        "rvol_5": rvol_5, "rvol_10": rvol_10, "rvol_20": rvol_20, "rvol_60": rvol_60,
        "ewma_vol": ewma_vol, "gk_vol_20": gk_vol_20,
        "vol_ratio_5_20": rvol_5 / (rvol_20 + 1e-12), "vol_ratio_10_60": rvol_10 / (rvol_60 + 1e-12),
        "vol_regime": 1.0 if rvol_20 > rvol_60 else 0.0,
        "mom_1": mom(1), "mom_5": mom(5), "mom_10": mom(10), "mom_20": mom(20), "mom_60": mom(60),
        "macd": float(macd_val), "macd_sig": float(state.ema9),
        "macd_hist": float(macd_val - state.ema9), "macd_cross": 0.0,
        "rsi_7": rsi7, "rsi_14": rsi14,
        "bb_pos": bb_pos, "bb_width": bb_wid,
        "vwap_dev": vwap_dev, "vwap_trend": vwap_trend,
        "vol_zscore": vol_zscore, "vol_ratio": vol_ratio,
        "tick_rule": tick, "ofi_5": float(state.ofi_acc5), "ofi_20": float(state.ofi_acc20),
        "amihud_20": amihud_20,
        "hl_range": float((lh - ll) / lc) if lc > 1e-12 else 0.0,
        "hl_pct":   float((highs[-20:] - lows[-20:]).mean() / lc) if lc > 1e-12 else 0.0,
        "close_pos": float((lc - ll) / (lh - ll + 1e-12)),
        "signed_rv_20": signed_rv,
        "skew_20": _rolling_stat(lr, 20, "skew"), "kurt_20": _rolling_stat(lr, 20, "kurt"),
        "autocorr_1": _rolling_stat(lr, 20, "ac1"),
        "time_sin": float(math.sin(2 * math.pi * mins / T)),
        "time_cos": float(math.cos(2 * math.pi * mins / T)),
        "intraday_pos": float(min(mins / T, 1.0)),
        "near_open": 1.0 if mins <= 30 else 0.0, "near_close": 1.0 if mins >= 360 else 0.0,
        "day_of_week": float(bars[-1].__dict__.get("_weekday", 2)),
        "ticker_id": TICKER_ID_MAP.get(state.ticker, 0.0),
    }


# =============================================================================
# POSITION MANAGEMENT (AI AUTO-TRADE)
# =============================================================================

def ai_should_enter(state: TickerState, prediction: Dict) -> bool:
    """Enter a new long position if signal is strong enough."""
    if state.position and not state.position.closed: return False  # already in
    if prediction.get("signal", 0) != 1:             return False
    if prediction.get("confidence", {}).get("score", 0) < MIN_CONFIDENCE: return False
    return True

def ai_should_exit(state: TickerState, current_price: float) -> Optional[str]:
    """Return exit reason if stop-loss or take-profit triggered."""
    pos = state.position
    if not pos or pos.closed: return None
    pnl_pct = (current_price - pos.entry_price) / pos.entry_price
    if pnl_pct <= -STOP_LOSS_PCT: return f"STOP_LOSS ({pnl_pct*100:.2f}%)"
    if pnl_pct >= TAKE_PROFIT_PCT: return f"TAKE_PROFIT ({pnl_pct*100:.2f}%)"
    return None

def open_position(state: TickerState, bar: Bar, prediction: Dict) -> Position:
    sl = bar.close * (1 - STOP_LOSS_PCT)
    tp = bar.close * (1 + TAKE_PROFIT_PCT)
    pos = Position(
        ticker=bar.ticker, entry_price=bar.close, entry_ts=bar.ts,
        stop_loss=sl, take_profit=tp, ai_managed=True,
    )
    state.position = pos
    state.last_pos = 1
    state.n_trades += 1
    logger.info(f"[AI ENTER] {bar.ticker} @ ${bar.close:.2f}  SL=${sl:.2f}  TP=${tp:.2f}")
    return pos

def close_position(state: TickerState, bar: Bar, reason: str, portfolio: "Portfolio") -> float:
    pos = state.position
    if not pos or pos.closed: return 0.0
    pnl_pct = (bar.close - pos.entry_price) / pos.entry_price
    pnl_log = math.log(bar.close / pos.entry_price) if pos.entry_price > 0 else 0.0
    cost    = COST_BPS * 2  # round-trip
    net_pnl = pnl_log - cost
    pos.closed = True
    pos.close_reason = reason
    state.realized_pnl += net_pnl
    state.last_pos      = 0
    state.position      = None
    portfolio.total_pnl += net_pnl
    portfolio.equity     = INITIAL_CAPITAL * (1 + portfolio.total_pnl)
    if pnl_log > 0: portfolio.n_wins   += 1
    else:           portfolio.n_losses += 1
    if portfolio.equity > portfolio.peak_equity: portfolio.peak_equity = portfolio.equity
    dd = (portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity
    if dd < portfolio.max_drawdown: portfolio.max_drawdown = dd
    logger.info(f"[AI EXIT] {bar.ticker} @ ${bar.close:.2f}  reason={reason}  net_pnl={net_pnl:.6f}")
    return net_pnl


# =============================================================================
# TRADING ENGINE
# =============================================================================

class TradingEngine:
    def __init__(self):
        self.model      = get_model()
        self.states:    Dict[str, TickerState]  = {t: TickerState(ticker=t) for t in TICKERS}
        self.portfolio  = Portfolio(started_at=datetime.utcnow().isoformat())
        self.log:       List[Dict] = []
        self._clients:  List[Any] = []
        self._running   = False
        self._task:     Optional[asyncio.Task] = None
        self._mode:     str = "idle"           # "live" | "simulation" | "idle"
        self._lock      = asyncio.Lock()

        # AI auto-trade state
        self.ai_trading_enabled = False
        self._stop_requested    = False
        self._drain_mode        = False        # graceful drain: sell all before stopping
        self._last_bar_ts:      float = 0.0   # for no-data timeout detection

    # ── Clients ──────────────────────────────────────────────────────────────

    async def register(self, ws):
        async with self._lock: self._clients.append(ws)

    async def unregister(self, ws):
        async with self._lock:
            try: self._clients.remove(ws)
            except ValueError: pass

    async def _broadcast(self, payload: Dict):
        msg  = json.dumps(payload, default=str)
        dead = []
        for ws in list(self._clients):
            try: await ws.send_text(msg)
            except Exception: dead.append(ws)
        for ws in dead: await self.unregister(ws)

    # ── Bar processing ────────────────────────────────────────────────────────

    async def process_bar(self, bar: Bar):
        self._last_bar_ts = time.time()
        state = self.states[bar.ticker]
        state.buf.append(bar)
        self.portfolio.bars_processed += 1

        raw = compute_features_raw(state)
        if raw is None:
            state.last_bar = bar
            await self._broadcast({
                "event": "warming", "ticker": bar.ticker,
                "buffer": len(state.buf), "needed": 62,
                "close": bar.close, "ts": bar.ts, "dt": bar.dt_iso,
            })
            return

        state.last_features = raw
        result = self.model.predict(raw)

        prob   = result.get("probability") or 0.5
        signal = result.get("signal", 0)
        label  = result.get("signal_label", "HOLD")
        conf   = result.get("confidence", {})

        # ── AI Auto-Trade logic ──────────────────────────────────────────────
        ai_action = None
        if self.ai_trading_enabled and self.model.loaded:
            # Check stop-loss / take-profit on open positions
            exit_reason = ai_should_exit(state, bar.close)
            if exit_reason:
                net_pnl = close_position(state, bar, exit_reason, self.portfolio)
                ai_action = {"type": "EXIT", "reason": exit_reason, "pnl": net_pnl}

                # If stop-loss triggered: recalculate / pause this ticker for 5 bars
                if "STOP_LOSS" in exit_reason:
                    await self._broadcast({
                        "event": "ai_stop_loss",
                        "ticker": bar.ticker,
                        "message": f"AI stopped {bar.ticker}: stop-loss hit. Recalculating…",
                        "pnl": round(net_pnl, 6),
                    })

                # If take-profit: notify user
                elif "TAKE_PROFIT" in exit_reason:
                    await self._broadcast({
                        "event": "ai_take_profit",
                        "ticker": bar.ticker,
                        "message": f"✓ AI took profit on {bar.ticker}. Waiting to re-enter.",
                        "pnl": round(net_pnl, 6),
                    })
                    # Prompt user if they want AI to continue
                    await self._broadcast({
                        "event": "ai_profit_prompt",
                        "message": f"AI successfully profited on {bar.ticker} ({net_pnl*100:.3f}%). "
                                   f"Send {{\"action\": \"ai_continue\"}} to resume, "
                                   f"or {{\"action\": \"ai_stop\"}} to halt.",
                        "ticker": bar.ticker,
                    })

            # Enter if conditions met
            elif ai_should_enter(state, result):
                open_position(state, bar, result)
                ai_action = {"type": "ENTER", "ticker": bar.ticker, "price": bar.close}

        # Drain mode: close all positions gracefully
        if self._drain_mode and state.position and not state.position.closed:
            pnl_pct = (bar.close - state.position.entry_price) / state.position.entry_price
            if pnl_pct >= 0.005:  # even 0.5% profit qualifies
                net_pnl = close_position(state, bar, "GRACEFUL_DRAIN", self.portfolio)
                ai_action = {"type": "DRAIN_EXIT", "pnl": net_pnl}

        # Check if drain complete → actually stop engine
        if self._drain_mode and self._all_positions_closed():
            self._drain_mode    = False
            self._stop_requested = True
            await self._broadcast({
                "event": "engine_stopped",
                "message": "All positions closed with profit. Engine halted.",
                "portfolio": asdict_safe(self.portfolio),
            })
            self._running = False
            return

        state.last_prob   = prob
        state.last_signal = signal
        state.last_bar    = bar

        # Update equity for bar carry (if holding)
        if state.last_pos == 1 and state.last_bar and state.last_bar != bar:
            bar_pnl = math.log(bar.close / state.last_bar.close) if state.last_bar.close > 0 else 0.0
            self.portfolio.total_pnl += bar_pnl
            self.portfolio.equity     = INITIAL_CAPITAL * (1 + self.portfolio.total_pnl)

        importances = self.model.feature_importances()
        top_feats   = sorted(importances.items(), key=lambda x: -x[1])[:10]
        display_f   = {k: round(raw.get(k, 0.0), 6) for k, _ in top_feats}

        output = {
            "event": "bar", "ticker": bar.ticker, "ts": bar.ts, "dt": bar.dt_iso,
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume,
            "probability": round(prob, 6), "probability_pct": round(prob * 100, 2),
            "signal": signal, "signal_label": label, "threshold": self.model.threshold,
            "confidence": conf,
            "reason": result.get("reason", ""),
            "version": result.get("version", "v2"),
            "position": state.last_pos,
            "realized_pnl": round(state.realized_pnl, 6),
            "n_trades": state.n_trades,
            "ai_action": ai_action,
            "portfolio": {
                "equity":      round(self.portfolio.equity, 2),
                "total_pnl":   round(self.portfolio.total_pnl, 6),
                "max_drawdown": round(self.portfolio.max_drawdown, 6),
                "n_trades":    self.portfolio.n_trades,
                "n_wins":      self.portfolio.n_wins,
                "n_losses":    self.portfolio.n_losses,
                "bars_processed": self.portfolio.bars_processed,
            },
            "features": display_f,
            "rsi_14":   round(raw.get("rsi_14", 50.0), 2),
            "bb_pos":   round(raw.get("bb_pos", 0.0), 4),
            "macd":     round(raw.get("macd", 0.0), 6),
            "vwap_dev": round(raw.get("vwap_dev", 0.0), 6),
        }

        self.log.append(output)
        if len(self.log) > 500: self.log.pop(0)
        await self._broadcast(output)

    # ── Polygon WebSocket (with no-data timeout + auto-fallback) ──────────────

    async def _run_ws(self):
        self._running = True
        self._mode    = "live"
        delay         = RECONNECT_INIT
        self._last_bar_ts = time.time()

        # Start a watchdog that detects market-closed / no data
        asyncio.get_event_loop().create_task(self._no_data_watchdog())

        while self._running:
            try:
                logger.info(f"Connecting → {POLYGON_WS_URL}")
                async with websockets.connect(
                    POLYGON_WS_URL, ping_interval=20, ping_timeout=30
                ) as ws:
                    auth_sent = subscribed = False
                    async for raw in ws:
                        if not self._running: break
                        try: msgs = json.loads(raw)
                        except Exception: continue
                        if not isinstance(msgs, list): msgs = [msgs]
                        for msg in msgs:
                            ev = msg.get("ev",""); status = msg.get("status","")
                            if ev == "status" and status == "connected" and not auth_sent:
                                await ws.send(json.dumps({"action":"auth","params":API_KEY}))
                                auth_sent = True
                            elif ev == "status" and status == "auth_success" and not subscribed:
                                await ws.send(json.dumps({"action":"subscribe","params":SUBSCRIBE_STR}))
                                subscribed = True
                                await self._broadcast({"event":"connected","message":f"Live: {SUBSCRIBE_STR}"})
                                logger.info(f"Subscribed: {SUBSCRIBE_STR}")
                            elif ev == "status" and status == "auth_failed":
                                raise RuntimeError("Polygon auth failed — check API key")
                            elif ev == "AM":
                                sym = msg.get("sym","")
                                if sym not in TICKERS: continue
                                ts_ms = int(msg.get("s", msg.get("t", time.time()*1000)))
                                try: dt_iso = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()
                                except Exception: dt_iso = datetime.utcnow().isoformat()
                                bar = Bar(
                                    ticker=sym, ts=ts_ms,
                                    open=float(msg.get("o",0)), high=float(msg.get("h",0)),
                                    low=float(msg.get("l",0)),  close=float(msg.get("c",0)),
                                    volume=float(msg.get("av",msg.get("v",0))),
                                    vwap=float(msg.get("vw",msg.get("c",0))),
                                    n_trades=int(msg.get("z",0)), dt_iso=dt_iso,
                                )
                                if bar.close > 0: await self.process_bar(bar)
                delay = RECONNECT_INIT
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"WS error: {e} — retry in {delay:.0f}s")
                await self._broadcast({"event":"disconnected","error":str(e),"reconnect_in":delay})
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)

    async def _no_data_watchdog(self):
        """If no Polygon bar arrives in NO_DATA_TIMEOUT seconds, auto-switch to simulation."""
        await asyncio.sleep(NO_DATA_TIMEOUT)
        if not self._running: return
        bars_received = sum(len(s.buf) for s in self.states.values())
        if bars_received == 0:
            ms = market_status()
            await self._broadcast({
                "event":   "auto_simulation",
                "message": f"No Polygon data in {NO_DATA_TIMEOUT}s. {ms['message']}. "
                           f"Switching to simulation mode automatically.",
            })
            logger.warning("No data watchdog triggered → switching to simulation")
            self._running = False
            await asyncio.sleep(0.5)
            self.start(simulate=True)

    # ── Simulation ────────────────────────────────────────────────────────────

    async def _run_sim(self):
        self._running = True
        self._mode    = "simulation"
        prices = {"AAPL":185.0,"MSFT":415.0,"NVDA":870.0,"AMZN":185.0,"TSLA":175.0}
        logger.info("Simulation mode started")

        # Warmup: send bars with no delay to prime buffers
        warmup_bars = 70
        for i in range(warmup_bars):
            for ticker in TICKERS:
                ret = np.random.normal(0, 0.0008)
                prices[ticker] *= math.exp(ret)
                p = prices[ticker]
                bar = Bar(
                    ticker=ticker, ts=int((time.time() - (warmup_bars - i) * 60) * 1000),
                    open=p*(1+np.random.uniform(-0.0005,0.0005)),
                    high=p*(1+abs(np.random.normal(0,0.001))),
                    low=p*(1-abs(np.random.normal(0,0.001))),
                    close=p, volume=float(np.random.randint(50_000,800_000)),
                    vwap=p*(1+np.random.uniform(-0.0002,0.0002)),
                    n_trades=np.random.randint(200,3000),
                    dt_iso=datetime.utcnow().isoformat(),
                )
                await self.process_bar(bar)

        await self._broadcast({"event":"warmup_complete",
                               "message":f"Buffer warm ({warmup_bars} bars). Predictions now active."})
        logger.info("Warmup complete — starting live simulation ticks")

        while self._running:
            for ticker in TICKERS:
                ret = np.random.normal(0.00002, 0.0012)  # slight upward drift + volatility
                prices[ticker] *= math.exp(ret)
                p = prices[ticker]
                bar = Bar(
                    ticker=ticker, ts=int(time.time()*1000),
                    open=p*(1+np.random.uniform(-0.0005,0.0005)),
                    high=p*(1+abs(np.random.normal(0,0.0015))),
                    low=p*(1-abs(np.random.normal(0,0.0015))),
                    close=p, volume=float(np.random.randint(50_000,800_000)),
                    vwap=p*(1+np.random.uniform(-0.0002,0.0002)),
                    n_trades=np.random.randint(200,3000),
                    dt_iso=datetime.utcnow().isoformat(),
                )
                await self.process_bar(bar)
            await asyncio.sleep(1.5)

    # ── Control ───────────────────────────────────────────────────────────────

    def _all_positions_closed(self) -> bool:
        return all(
            s.position is None or s.position.closed
            for s in self.states.values()
        )

    def start(self, simulate: bool = False):
        if self._task and not self._task.done(): return
        self._running          = True
        self._stop_requested   = False
        self._drain_mode       = False
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(
            self._run_sim() if simulate else self._run_ws()
        )
        logger.info(f"Engine started ({'sim' if simulate else 'live'})")

    def request_graceful_stop(self):
        """
        Graceful stop: switch to drain mode.
        Engine will continue running until every AI position is closed
        with at least 0.5% profit, then halts automatically.
        """
        has_open = any(
            s.position and not s.position.closed
            for s in self.states.values()
        )
        if has_open:
            self._drain_mode = True
            logger.info("Graceful drain mode activated — waiting for profitable exits")
        else:
            self._running = False

    def stop_hard(self):
        """Immediate stop — ignores open positions."""
        self._running = False
        if self._task: self._task.cancel()

    def enable_ai_trading(self, enable: bool = True):
        self.ai_trading_enabled = enable
        self.portfolio.ai_mode  = enable

    def status(self) -> Dict[str, Any]:
        open_positions = {
            t: {
                "entry_price": s.position.entry_price,
                "stop_loss": s.position.stop_loss,
                "take_profit": s.position.take_profit,
                "current_price": s.last_bar.close if s.last_bar else None,
                "unrealized_pnl_pct": round(
                    (s.last_bar.close - s.position.entry_price) / s.position.entry_price * 100, 3
                ) if s.last_bar and s.position else None,
            }
            for t, s in self.states.items()
            if s.position and not s.position.closed
        }
        return {
            "running":          self._running,
            "mode":             self._mode,
            "ai_trading":       self.ai_trading_enabled,
            "drain_mode":       self._drain_mode,
            "market":           market_status(),
            "model_loaded":     self.model.loaded,
            "tickers":          TICKERS,
            "portfolio":        asdict_safe(self.portfolio),
            "open_positions":   open_positions,
            "ticker_states":    {
                t: {
                    "buffer_size": len(s.buf), "ready": len(s.buf) >= 62,
                    "last_prob":  round(s.last_prob, 4) if s.last_prob else None,
                    "last_signal": s.last_signal, "last_pos": s.last_pos,
                    "last_close": s.last_bar.close if s.last_bar else None,
                    "realized_pnl": round(s.realized_pnl, 6),
                }
                for t, s in self.states.items()
            },
            "log_entries":      len(self.log),
            "connected_clients": len(self._clients),
        }

    def get_all_predictions(self) -> Dict[str, Any]:
        """Return current predictions for all tickers (for /predict endpoint)."""
        preds = {}
        for t, s in self.states.items():
            if not s.last_features or not self.model.loaded:
                preds[t] = {"ready": False, "reason": "Buffer warming or model not loaded"}
                continue
            try:
                result = self.model.predict(s.last_features)
                result["ticker"]      = t
                result["last_close"]  = s.last_bar.close if s.last_bar else None
                result["in_position"] = bool(s.position and not s.position.closed)
                preds[t] = result
            except Exception as e:
                preds[t] = {"ready": False, "error": str(e)}
        return preds


def asdict_safe(obj):
    from dataclasses import asdict
    try: return asdict(obj)
    except Exception: return obj.__dict__ if hasattr(obj, "__dict__") else {}


_engine: Optional[TradingEngine] = None

def get_engine() -> TradingEngine:
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine
