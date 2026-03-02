# =============================================================================
# backend/trading_engine.py  —  Live Streaming + V2 Feature Engine
# =============================================================================
# Computes all 49 raw features then filters to exactly the set in features.json.
# This means no code change is needed if the feature list changes — the engine
# auto-adapts at startup when it loads the model artifacts.
# =============================================================================

import os, asyncio, json, logging, time, math
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Deque

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedError

from model import get_model

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

POLYGON_WS_URL   = "wss://socket.polygon.io/stocks"
API_KEY          = os.environ.get("POLYGON_API_KEY", "MoyLn951WdZAozaSClrOGar9xgYjy0pR")
TICKERS          = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
SUBSCRIBE_STR    = ",".join(f"AM.{t}" for t in TICKERS)

BUFFER_SIZE      = 80       # keep 80 bars per ticker (covers rvol_60 + ema_26)
EWMA_LAMBDA      = 0.94
COST_BPS         = 0.0005   # transaction cost as fraction of return (5 bps)
INITIAL_CAPITAL  = 100_000.0

RECONNECT_INIT   = 3.0
RECONNECT_MAX    = 60.0

# All 15 tickers trained on — needed for ticker_id encoding
ALL_TRAIN_TICKERS = sorted([
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    "GOOGL", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ", "BABA", "CRM", "UBER"
])
TICKER_ID_MAP = {t: float(i) for i, t in enumerate(ALL_TRAIN_TICKERS)}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bar:
    ticker:   str
    ts:       int     # Unix ms
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   float
    vwap:     float
    n_trades: int
    dt_iso:   str = ""


@dataclass
class TickerState:
    ticker:    str
    buf:       Deque[Bar] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))

    # Recursive EMA state (avoids recomputing from scratch each bar)
    ema12:     float = 0.0
    ema26:     float = 0.0
    ema9:      float = 0.0    # MACD signal
    ewma_var:  float = 0.0    # EWMA variance for ewma_vol
    prev_tick: float = 0.0    # last close for tick rule
    ofi_acc5:  int   = 0      # order flow imbalance accumulator
    ofi_acc20: int   = 0

    # Live prediction state
    last_prob:   Optional[float] = None
    last_signal: int  = 0
    last_pos:    int  = 0       # actual demo position (0 or 1)
    last_bar:    Optional[Bar] = None

    # Per-ticker demo PnL
    realized_pnl: float = 0.0
    n_trades:      int   = 0


@dataclass
class Portfolio:
    cash:          float = INITIAL_CAPITAL
    equity:        float = INITIAL_CAPITAL
    total_pnl:     float = 0.0
    n_trades:      int   = 0
    bars_processed:int   = 0
    peak_equity:   float = INITIAL_CAPITAL
    max_drawdown:  float = 0.0
    started_at:    str   = ""


# =============================================================================
# FEATURE COMPUTATION — computes all 49 possible features
# Model uses only the subset stored in features.json (loaded via get_model())
# =============================================================================

def _ema_step(prev: float, val: float, span: int) -> float:
    alpha = 2.0 / (span + 1)
    if prev == 0.0:
        return val
    return alpha * val + (1 - alpha) * prev


def _rolling(buf: Deque[Bar], attr: str) -> np.ndarray:
    return np.array([getattr(b, attr) for b in buf])


def _log_rets(closes: np.ndarray) -> np.ndarray:
    if len(closes) < 2:
        return np.array([])
    return np.log(closes[1:] / closes[:-1])


def _rrvol(log_rets: np.ndarray, w: int) -> Optional[float]:
    if len(log_rets) < w:
        return None
    return float(np.sqrt(np.sum(log_rets[-w:] ** 2)))


def _rolling_skew(arr: np.ndarray, w: int) -> Optional[float]:
    if len(arr) < w:
        return None
    x = arr[-w:]
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / (std ** 3))


def _rolling_kurt(arr: np.ndarray, w: int) -> Optional[float]:
    if len(arr) < w:
        return None
    x = arr[-w:]
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(((x - x.mean()) ** 4).mean() / (std ** 4))


def _autocorr1(arr: np.ndarray, w: int) -> Optional[float]:
    if len(arr) < w:
        return None
    x = arr[-w:]
    if len(x) < 2:
        return 0.0
    mu  = x.mean()
    den = ((x - mu) ** 2).sum()
    if den < 1e-12:
        return 0.0
    num = ((x[1:] - mu) * (x[:-1] - mu)).sum()
    return float(num / den)


def _rsi(log_rets: np.ndarray, period: int) -> Optional[float]:
    if len(log_rets) < period:
        return None
    r  = log_rets[-period:]
    up = np.where(r > 0, r, 0.0).mean()
    dn = np.where(r < 0, -r, 0.0).mean()
    if dn < 1e-12:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + up / dn))


def compute_features_raw(state: TickerState) -> Optional[Dict[str, float]]:
    """
    Compute all 49 candidate features from the rolling buffer.
    Returns None if the buffer has too few bars to compute reliably.
    Minimum bars needed: 61 (covers rvol_60, ema_26, sma_20 with margin).
    """
    buf = state.buf
    if len(buf) < 62:
        return None

    bars    = list(buf)
    closes  = np.array([b.close  for b in bars])
    highs   = np.array([b.high   for b in bars])
    lows    = np.array([b.low    for b in bars])
    opens   = np.array([b.open   for b in bars])
    vols    = np.array([b.volume for b in bars])
    vwaps   = np.array([b.vwap   for b in bars])

    log_rets = _log_rets(closes)   # length = len(bars) - 1

    last_c   = closes[-1]
    last_h   = highs[-1]
    last_l   = lows[-1]
    last_o   = opens[-1]
    last_v   = vols[-1]
    last_vw  = vwaps[-1]
    last_ret = float(log_rets[-1]) if len(log_rets) > 0 else 0.0

    # ── Returns family ────────────────────────────────────────────────────────
    log_ret    = last_ret
    log_ret_sq = last_ret ** 2
    ret_1_lag1 = float(log_rets[-2]) if len(log_rets) >= 2 else 0.0
    ret_1_lag2 = float(log_rets[-3]) if len(log_rets) >= 3 else 0.0
    cumret_5   = float(log_rets[-5:].sum()) if len(log_rets) >= 5 else 0.0

    # ── Realized volatilities ─────────────────────────────────────────────────
    rvol_5  = _rrvol(log_rets, 5)
    rvol_10 = _rrvol(log_rets, 10)
    rvol_20 = _rrvol(log_rets, 20)
    rvol_60 = _rrvol(log_rets, 60)

    if any(v is None for v in [rvol_5, rvol_10, rvol_20, rvol_60]):
        return None

    # ── EWMA vol ──────────────────────────────────────────────────────────────
    state.ewma_var = EWMA_LAMBDA * state.ewma_var + (1 - EWMA_LAMBDA) * last_ret ** 2
    ewma_vol = float(math.sqrt(max(state.ewma_var, 0.0)))

    # ── Garman-Klass vol ──────────────────────────────────────────────────────
    # σ²_GK = mean(0.5(ln H/L)² - (2ln2-1)(ln C/O)²) over window 20
    hl2 = np.log(highs[-21:] / lows[-21:].clip(min=1e-12)) ** 2 * 0.5
    co2 = np.log(closes[-21:] / opens[-21:].clip(min=1e-12)) ** 2 * (2 * math.log(2) - 1)
    gk_val = float((hl2 - co2).mean())
    gk_vol_20 = float(math.sqrt(max(gk_val, 0.0)))

    # ── Vol regime (removed at training as constant — keep for completeness) ──
    # vol_regime dropped during prepare() so will be filtered out anyway

    # ── Vol ratios ────────────────────────────────────────────────────────────
    vol_ratio_5_20  = rvol_5  / (rvol_20 + 1e-12)
    vol_ratio_10_60 = rvol_10 / (rvol_60 + 1e-12)
    vol_regime      = 1.0 if rvol_20 > rvol_60 else 0.0

    # ── Momentum ──────────────────────────────────────────────────────────────
    def mom(n):
        if len(closes) <= n:
            return 0.0
        ref = closes[-n-1]
        return float((last_c - ref) / ref) if ref > 1e-12 else 0.0

    mom_1  = mom(1)
    mom_5  = mom(5)
    mom_10 = mom(10)
    mom_20 = mom(20)
    mom_60 = mom(60)

    # ── MACD family ───────────────────────────────────────────────────────────
    state.ema12 = _ema_step(state.ema12, last_c, 12)
    state.ema26 = _ema_step(state.ema26, last_c, 26)
    macd_val    = state.ema12 - state.ema26
    state.ema9  = _ema_step(state.ema9, macd_val, 9)
    macd_sig_val  = state.ema9
    macd_hist_val = macd_val - macd_sig_val

    # macd_cross: sign change of macd_hist (requires previous bar's hist)
    # We approximate as 0 unless we track previous hist — set to 0.0 safely
    macd_cross = 0.0

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_7_val  = _rsi(log_rets, 7)
    rsi_14_val = _rsi(log_rets, 14)
    if rsi_7_val is None or rsi_14_val is None:
        return None

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    sma20  = float(closes[-20:].mean())
    std20  = float(closes[-20:].std())
    bb_pos = float((last_c - sma20) / (2 * std20)) if std20 > 1e-12 else 0.0
    bb_wid = float(4 * std20 / sma20) if sma20 > 1e-12 else 0.0

    # ── VWAP deviation ────────────────────────────────────────────────────────
    vwap_dev    = float((last_c - last_vw) / last_vw) if last_vw > 1e-12 else 0.0
    vwap_devs   = (closes[-6:] - vwaps[-6:]) / vwaps[-6:].clip(min=1e-12)
    vwap_trend  = float(vwap_devs.mean()) if len(vwap_devs) >= 5 else vwap_dev

    # ── Volume z-score ────────────────────────────────────────────────────────
    v_mean     = float(vols[-20:].mean())
    v_std      = float(vols[-20:].std())
    vol_zscore = float((last_v - v_mean) / v_std) if v_std > 1e-12 else 0.0
    vol_ratio  = float(last_v / v_mean) if v_mean > 1e-12 else 1.0

    # ── Tick rule / order flow ────────────────────────────────────────────────
    tick = np.sign(last_c - state.prev_tick) if state.prev_tick > 0 else 0.0
    tick_rule = float(tick)
    state.prev_tick = last_c

    # Accumulate OFI in state (deque-like rolling sum approximation)
    state.ofi_acc5  = int(tick) + state.ofi_acc5
    state.ofi_acc20 = int(tick) + state.ofi_acc20
    ofi_5  = float(state.ofi_acc5)
    ofi_20 = float(state.ofi_acc20)
    # Cap to reasonable range
    state.ofi_acc5  = max(-5,  min(5,  state.ofi_acc5))
    state.ofi_acc20 = max(-20, min(20, state.ofi_acc20))

    # ── Amihud illiquidity ────────────────────────────────────────────────────
    illiq = np.abs(log_rets[-20:]) / (vols[-20:] + 1)
    amihud_20 = float(illiq.mean())

    # ── Bar microstructure ────────────────────────────────────────────────────
    hl_range  = float((last_h - last_l) / last_c) if last_c > 1e-12 else 0.0
    hl_pct    = float((highs[-20:] - lows[-20:]).mean() / last_c) if last_c > 1e-12 else 0.0
    close_pos = float((last_c - last_l) / (last_h - last_l + 1e-12))

    # Signed realized variance ratio
    pos_sq = log_rets[-20:].clip(min=0) ** 2
    neg_sq = log_rets[-20:].clip(max=0) ** 2
    signed_rv_20 = float((pos_sq.sum() - neg_sq.sum()) / (pos_sq.sum() + neg_sq.sum() + 1e-12))

    # ── Higher moments ────────────────────────────────────────────────────────
    skew_20_val  = _rolling_skew(log_rets, 20) or 0.0
    kurt_20_val  = _rolling_kurt(log_rets, 20) or 0.0
    autocorr_1   = _autocorr1(log_rets, 20) or 0.0

    # ── Time features ─────────────────────────────────────────────────────────
    last_bar  = bars[-1]
    try:
        dt = datetime.fromisoformat(last_bar.dt_iso.replace("Z", "+00:00"))
        # Convert to ET for correct market-hour calculation
        from datetime import timedelta
        dt_et = dt - timedelta(hours=4)  # EDT offset; close enough for intraday
        mins  = max(0, dt_et.hour * 60 + dt_et.minute - 570)  # 9:30 AM = 570 min
    except Exception:
        mins = 195  # mid-day default

    T = 390
    time_sin    = float(math.sin(2 * math.pi * mins / T))
    time_cos    = float(math.cos(2 * math.pi * mins / T))
    intraday_pos = float(min(mins / T, 1.0))
    near_open   = 1.0 if mins <= 30 else 0.0
    near_close  = 1.0 if mins >= 360 else 0.0
    day_of_week = float(dt.weekday()) if 'dt' in dir() else 2.0

    # ── Ticker id ─────────────────────────────────────────────────────────────
    ticker_id = TICKER_ID_MAP.get(state.ticker, 0.0)

    return {
        # Returns
        "log_ret": log_ret, "log_ret_sq": log_ret_sq,
        "ret_1_lag1": ret_1_lag1, "ret_1_lag2": ret_1_lag2, "cumret_5": cumret_5,
        # Volatilities
        "rvol_5": rvol_5, "rvol_10": rvol_10, "rvol_20": rvol_20, "rvol_60": rvol_60,
        "ewma_vol": ewma_vol, "gk_vol_20": gk_vol_20,
        # Vol regime
        "vol_ratio_5_20": vol_ratio_5_20, "vol_ratio_10_60": vol_ratio_10_60,
        "vol_regime": vol_regime,
        # Momentum
        "mom_1": mom_1, "mom_5": mom_5, "mom_10": mom_10, "mom_20": mom_20, "mom_60": mom_60,
        # MACD
        "macd": float(macd_val), "macd_sig": float(macd_sig_val),
        "macd_hist": float(macd_hist_val), "macd_cross": macd_cross,
        # Oscillators
        "rsi_7": rsi_7_val, "rsi_14": rsi_14_val,
        # Bollinger
        "bb_pos": bb_pos, "bb_width": bb_wid,
        # VWAP
        "vwap_dev": vwap_dev, "vwap_trend": vwap_trend,
        # Volume
        "vol_zscore": vol_zscore, "vol_ratio": vol_ratio,
        # Order flow
        "tick_rule": tick_rule, "ofi_5": ofi_5, "ofi_20": ofi_20,
        # Illiquidity
        "amihud_20": amihud_20,
        # Microstructure
        "hl_range": hl_range, "hl_pct": hl_pct, "close_pos": close_pos,
        "signed_rv_20": signed_rv_20,
        # Higher moments
        "skew_20": skew_20_val, "kurt_20": kurt_20_val, "autocorr_1": autocorr_1,
        # Time
        "time_sin": time_sin, "time_cos": time_cos, "intraday_pos": intraday_pos,
        "near_open": near_open, "near_close": near_close, "day_of_week": day_of_week,
        # Encoding
        "ticker_id": ticker_id,
    }


# =============================================================================
# DEMO TRADING
# =============================================================================

def update_position(state: TickerState, portfolio: Portfolio,
                    new_signal: int, bar: Bar):
    """
    Fixed PnL logic: cost = COST_BPS * |Δpos| (dimensionally consistent).
    Handles the long → flat → long transitions.
    """
    close    = bar.close
    prev_pos = state.last_pos
    new_pos  = new_signal

    # PnL from carrying previous position through this bar
    if prev_pos == 1 and state.last_bar is not None:
        price_chg = close - state.last_bar.close
        bar_pnl   = math.log(close / state.last_bar.close) if state.last_bar.close > 0 else 0.0
    else:
        bar_pnl = 0.0

    # Transaction cost (in log-return basis)
    trade   = abs(new_pos - prev_pos)
    cost    = COST_BPS * trade
    net_pnl = bar_pnl - cost

    state.realized_pnl += net_pnl
    if trade:
        state.n_trades += 1

    portfolio.total_pnl      += net_pnl
    portfolio.equity          = INITIAL_CAPITAL + portfolio.total_pnl * INITIAL_CAPITAL
    portfolio.n_trades        += trade
    portfolio.bars_processed  += 1

    if portfolio.equity > portfolio.peak_equity:
        portfolio.peak_equity = portfolio.equity
    dd = (portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity
    if dd < portfolio.max_drawdown:
        portfolio.max_drawdown = dd

    state.last_pos  = new_pos
    state.last_bar  = bar


# =============================================================================
# TRADING ENGINE
# =============================================================================

class TradingEngine:
    def __init__(self):
        self.model    = get_model()
        self.states:  Dict[str, TickerState]  = {t: TickerState(ticker=t) for t in TICKERS}
        self.portfolio = Portfolio(started_at=datetime.utcnow().isoformat())
        self.log:     List[Dict[str, Any]] = []   # rolling prediction log (500 entries)
        self._clients: List[Any] = []
        self._running = False
        self._task:   Optional[asyncio.Task] = None
        self._lock    = asyncio.Lock()

    # ── Client management ─────────────────────────────────────────────────────

    async def register(self, ws):
        async with self._lock:
            self._clients.append(ws)

    async def unregister(self, ws):
        async with self._lock:
            try: self._clients.remove(ws)
            except ValueError: pass

    async def _broadcast(self, payload: Dict):
        msg  = json.dumps(payload, default=str)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)

    # ── Bar processing ────────────────────────────────────────────────────────

    async def process_bar(self, bar: Bar):
        state = self.states[bar.ticker]
        state.buf.append(bar)

        # Compute all raw features
        raw_features = compute_features_raw(state)
        if raw_features is None:
            await self._broadcast({
                "event":   "warming",
                "ticker":  bar.ticker,
                "buffer":  len(state.buf),
                "needed":  62,
                "close":   bar.close,
                "ts":      bar.ts,
                "dt":      bar.dt_iso,
            })
            return

        # Predict (model.predict filters to its own feature subset)
        result = self.model.predict(raw_features)
        prob   = result["probability"] or 0.5
        signal = result["signal"]
        label  = result["signal_label"]

        # Update demo position
        update_position(state, self.portfolio, signal, bar)
        state.last_prob   = prob
        state.last_signal = signal

        # Feature importance subset for display (top 10 by model importance)
        importances  = self.model.feature_importances()
        top_feats    = sorted(importances.items(), key=lambda x: -x[1])[:10]
        display_feats = {k: round(raw_features.get(k, 0.0), 6) for k, _ in top_feats}

        output = {
            "event":      "bar",
            "ticker":     bar.ticker,
            "ts":         bar.ts,
            "dt":         bar.dt_iso,
            # OHLCV for candlestick
            "open":       bar.open,
            "high":       bar.high,
            "low":        bar.low,
            "close":      bar.close,
            "volume":     bar.volume,
            # Model output
            "probability":     round(prob, 6),
            "probability_pct": round(prob * 100, 2),
            "signal":          signal,
            "signal_label":    label,
            "threshold":       self.model.threshold,
            "threshold_pct":   round(self.model.threshold * 100, 2),
            # Position / PnL
            "position":      state.last_pos,
            "realized_pnl":  round(state.realized_pnl, 6),
            "n_trades":      state.n_trades,
            # Portfolio snapshot
            "portfolio": {
                "equity":        round(self.portfolio.equity, 2),
                "total_pnl":     round(self.portfolio.total_pnl, 6),
                "max_drawdown":  round(self.portfolio.max_drawdown, 6),
                "n_trades":      self.portfolio.n_trades,
                "bars_processed": self.portfolio.bars_processed,
            },
            # Feature values (top 10 for display)
            "features": display_feats,
            # Key display features
            "rsi_14":    round(raw_features.get("rsi_14", 50.0), 2),
            "bb_pos":    round(raw_features.get("bb_pos", 0.0), 4),
            "macd":      round(raw_features.get("macd", 0.0), 6),
            "vwap_dev":  round(raw_features.get("vwap_dev", 0.0), 6),
            "rvol_20":   round(raw_features.get("rvol_20", 0.0), 6),
        }

        # Append to rolling log
        self.log.append(output)
        if len(self.log) > 500:
            self.log.pop(0)

        await self._broadcast(output)

        logger.info(
            f"[{bar.ticker}] close={bar.close:.2f} "
            f"p={prob:.4f} ({label}) "
            f"equity=${self.portfolio.equity:,.2f}"
        )

    # ── Polygon WebSocket ─────────────────────────────────────────────────────

    async def _run_ws(self):
        self._running = True
        delay = RECONNECT_INIT

        while self._running:
            try:
                logger.info(f"Connecting → {POLYGON_WS_URL}")
                async with websockets.connect(
                    POLYGON_WS_URL, ping_interval=20, ping_timeout=30
                ) as ws:
                    auth_sent   = False
                    subscribed  = False

                    async for raw in ws:
                        try:
                            msgs = json.loads(raw)
                        except Exception:
                            continue
                        if not isinstance(msgs, list):
                            msgs = [msgs]

                        for msg in msgs:
                            ev     = msg.get("ev")
                            status = msg.get("status", "")

                            if ev == "status" and status == "connected" and not auth_sent:
                                await ws.send(json.dumps({"action": "auth", "params": API_KEY}))
                                auth_sent = True

                            elif ev == "status" and status == "auth_success" and not subscribed:
                                await ws.send(json.dumps({"action": "subscribe", "params": SUBSCRIBE_STR}))
                                subscribed = True
                                logger.info(f"Subscribed: {SUBSCRIBE_STR}")
                                await self._broadcast({
                                    "event": "connected",
                                    "message": f"Live stream active: {SUBSCRIBE_STR}",
                                    "ts": datetime.utcnow().isoformat()
                                })

                            elif ev == "status" and status == "auth_failed":
                                raise RuntimeError("Polygon auth failed")

                            elif ev == "AM":
                                sym = msg.get("sym", "")
                                if sym not in TICKERS:
                                    continue
                                ts_ms = int(msg.get("s", msg.get("t", time.time() * 1000)))
                                try:
                                    dt_iso = datetime.fromtimestamp(
                                        ts_ms / 1000, tz=timezone.utc
                                    ).isoformat()
                                except Exception:
                                    dt_iso = datetime.utcnow().isoformat()

                                bar = Bar(
                                    ticker   = sym,
                                    ts       = ts_ms,
                                    open     = float(msg.get("o", 0)),
                                    high     = float(msg.get("h", 0)),
                                    low      = float(msg.get("l", 0)),
                                    close    = float(msg.get("c", 0)),
                                    volume   = float(msg.get("av", msg.get("v", 0))),
                                    vwap     = float(msg.get("vw", msg.get("c", 0))),
                                    n_trades = int(msg.get("z", 0)),
                                    dt_iso   = dt_iso,
                                )
                                if bar.close > 0:
                                    await self.process_bar(bar)

                delay = RECONNECT_INIT  # reset on clean disconnect

            except asyncio.CancelledError:
                logger.info("Engine cancelled.")
                break
            except Exception as e:
                logger.error(f"WS error: {e} — reconnecting in {delay:.0f}s")
                await self._broadcast({
                    "event": "disconnected", "error": str(e),
                    "reconnect_in": delay, "ts": datetime.utcnow().isoformat()
                })
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)

    # ── Simulation mode ───────────────────────────────────────────────────────

    async def _run_sim(self):
        self._running = True
        prices = {"AAPL": 185.0, "MSFT": 415.0, "NVDA": 870.0,
                  "AMZN": 185.0, "TSLA": 175.0}
        logger.info("Simulation mode started")
        while self._running:
            for ticker in TICKERS:
                ret = np.random.normal(0, 0.0008)
                prices[ticker] *= math.exp(ret)
                p = prices[ticker]
                noise = lambda f: p * (1 + np.random.uniform(-f, f))
                bar = Bar(
                    ticker   = ticker,
                    ts       = int(time.time() * 1000),
                    open     = noise(0.0005),
                    high     = p * (1 + abs(np.random.normal(0, 0.001))),
                    low      = p * (1 - abs(np.random.normal(0, 0.001))),
                    close    = p,
                    volume   = float(np.random.randint(50_000, 800_000)),
                    vwap     = noise(0.0002),
                    n_trades = np.random.randint(200, 3000),
                    dt_iso   = datetime.utcnow().isoformat(),
                )
                await self.process_bar(bar)
            await asyncio.sleep(1.5)

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, simulate: bool = False):
        if self._task and not self._task.done():
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(
            self._run_sim() if simulate else self._run_ws()
        )
        logger.info(f"Engine started ({'simulation' if simulate else 'live'})")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    def status(self) -> Dict[str, Any]:
        return {
            "running":     self._running,
            "model_loaded": self.model.loaded,
            "model_version": self.model.model_version,
            "tickers":     TICKERS,
            "portfolio":   asdict(self.portfolio),
            "ticker_states": {
                t: {
                    "buffer_size":  len(s.buf),
                    "ready":        len(s.buf) >= 62,
                    "last_prob":    round(s.last_prob, 4) if s.last_prob else None,
                    "signal_label": self.model.predict({"ticker_id": 0})["signal_label"]
                                    if s.last_prob and self.model.loaded else "—",
                    "last_signal":  s.last_signal,
                    "last_pos":     s.last_pos,
                    "last_close":   s.last_bar.close if s.last_bar else None,
                    "realized_pnl": round(s.realized_pnl, 6),
                    "n_trades":     s.n_trades,
                }
                for t, s in self.states.items()
            },
            "connected_clients": len(self._clients),
            "log_entries":       len(self.log),
        }


_engine: Optional[TradingEngine] = None

def get_engine() -> TradingEngine:
    global _engine
    if _engine is None:
        _engine = TradingEngine()
    return _engine
