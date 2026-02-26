"""
src/api/server.py — FastAPI Control & Telemetry Server
=======================================================
Exposes two endpoints to the Android app (via Tailscale):

  POST /api/command  — Receive biometric kill, pause, resume, bias commands.
  WS   /ws/telemetry — 1-second broadcast of live PnL, margin, market regime,
                       sentiment feed, and active trades.

Startup injection:
  Call set_simulator_ref(simulator) before starting uvicorn so the WebSocket
  handler can read live data without circular imports.

Author: Quant & Execution Architecture Team
"""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from src.live.control_state import CONTROL_STATE, _async_lock
from src.live.upstox_simulator import UpstoxSimulator

# Optional HybridNewsEngine Import
try:
    from src.core.hybrid_news import HybridNewsEngine
    news_engine = HybridNewsEngine()
except ImportError:
    news_engine = None

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trading Engine Control API",
    description="Android ↔ FastAPI bridge for the XGBoost+Renko intraday engine.",
    version="1.0.0",
)

# Allow Android app to call from any origin (Tailscale private network is safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the server boots."""
    logger.info("Starting background tasks...")
    if news_engine:
        # Import here to avoid circular dependencies if any, since it's defined later in the file
        from src.api.server import automated_news_spooler
        asyncio.create_task(automated_news_spooler())
    else:
        logger.warning("HybridNewsEngine not initialized; skipping background spooler.")

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR REFERENCE (injected by server_main.py at startup)
# ─────────────────────────────────────────────────────────────────────────────
_simulator_ref: Optional[UpstoxSimulator] = None


def set_simulator_ref(simulator: UpstoxSimulator) -> None:
    """
    Called once by server_main.py after the PaperPortfolio is created.
    Gives the WebSocket handler access to live margin and PnL data.
    """
    global _simulator_ref
    _simulator_ref = simulator
    logger.info("Simulator reference registered with API server.")


# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
# Maintains a rolling window of brick directions to compute regime on the fly.
# No external data dependency — pure from the Renko engine's output signal.
_regime_buffer: deque = deque(maxlen=40)   # last 40 brick direction signals


def register_brick_signal(direction: int, conviction: float) -> None:
    """
    Called from the trading loop (via server_main.py) each time a new brick
    fires so the regime buffer stays fresh. direction: +1 or -1.
    """
    _regime_buffer.append({"dir": direction, "conv": conviction})


def compute_market_regime() -> str:
    """
    Derive market regime from the rolling brick direction buffer.

    Rules (from 40 most recent cross-stock brick signals):
      SIDEWAYS  — fewer than 15 signals, or net_bias < 40 %
      VOLATILE  — net_bias 40–60 % AND avg conviction < 45
      TRENDING  — net_bias > 60 % OR avg conviction > 60
    """
    if _simulator_ref is None:
        return _read_live_state().get("market_regime", "SIDEWAYS")

    if len(_regime_buffer) < 10:
        return "SIDEWAYS"

    directions  = [b["dir"] for b in _regime_buffer]
    convictions = [b["conv"] for b in _regime_buffer]
    longs  = directions.count(1)
    shorts = directions.count(-1)
    total  = len(directions)
    net_bias    = abs(longs - shorts) / total * 100   # 0–100 %
    avg_conv    = sum(convictions) / len(convictions)

    if net_bias > 60 or avg_conv > 60:
        return "TRENDING"
    if net_bias >= 40 and avg_conv < 45:
        return "VOLATILE"
    return "SIDEWAYS"


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT FEED 
# ─────────────────────────────────────────────────────────────────────────────
# This cache is populated directly by the automated_news_spooler background task.
# No placeholder or dummy data is ever inserted here.
_latest_sentiment_cache: list[dict] = []

def _get_sentiment_feed() -> list[dict]:
    """
    Returns only live, real-world per-ticker FinBERT scores gathered from yfinance and RSS.
    Returns an empty list if no news has been parsed yet.
    """
    global _latest_sentiment_cache
    return _latest_sentiment_cache

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE TRADES SNAPSHOT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _read_live_state() -> dict:
    """Read the live_state.json written by the live engine (separate process)."""
    try:
        p = Path(config.LIVE_STATE_FILE)
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _get_active_trades() -> list[dict]:
    """Return active trades — from in-process simulator OR live_state.json fallback."""
    if _simulator_ref is not None:
        trades = []
        for sym, order in _simulator_ref.active_trades.items():
            trades.append({
                "symbol":         sym,
                "side":           order.side,
                "qty":            order.qty,
                "entry_price":    round(order.entry_price, 2),
                "last_price":     round(order.last_price, 2),
                "unrealized_pnl": round(order.unrealized_pnl, 2),
                "locked_margin":  round(order.locked_margin, 2),
                "entry_time":     order.filled_at.isoformat() if order.filled_at else None,
            })
        return trades

    # Fallback: live engine wrote trades to live_state.json (separate process)
    return _read_live_state().get("active_trades", [])


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE 3A — POST /api/command
# ─────────────────────────────────────────────────────────────────────────────

class CommandPayload(BaseModel):
    """
    JSON body for the Android command endpoint.

    3-Tier Control Hierarchy:
      Tier 1 — Engine-level:  {"command": "KILL"}
                               {"command": "GLOBAL_PAUSE"}
                               {"command": "GLOBAL_RESUME"}
      Tier 2 — Ticker-level:  {"command": "PAUSE_TICKER",  "ticker": "RELIANCE"}
                               {"command": "RESUME_TICKER", "ticker": "RELIANCE"}
      Tier 3 — Hunter Mode:   {"command": "BIAS",       "ticker": "LT", "direction": "LONG"}
                               {"command": "CLEAR_BIAS", "ticker": "LT"}
      Info:                    {"command": "STATUS"}
    """
    command:   str
    ticker:    Optional[str] = None
    direction: Optional[str] = None   # "LONG" or "SHORT"


@app.post("/api/command")
async def handle_command(payload: CommandPayload):
    """
    Receives a command from the Android app and atomically mutates CONTROL_STATE.
    """
    cmd = payload.command.upper()

    async with _async_lock:

        # ── TIER 1: Engine-Level Controls ──────────────────────────────
        if cmd == "KILL":
            CONTROL_STATE["GLOBAL_KILL"] = True
            logger.critical("ANDROID -> GLOBAL KILL SWITCH ACTIVATED.")
            return {"status": "ok", "detail": "GLOBAL_KILL set to True — squaring off all positions"}

        elif cmd == "GLOBAL_PAUSE":
            CONTROL_STATE["GLOBAL_PAUSE"] = True
            logger.warning("ANDROID -> GLOBAL_PAUSE: all new entries suppressed.")
            return {"status": "ok", "detail": "GLOBAL_PAUSE = True — all entries suspended"}

        elif cmd == "GLOBAL_RESUME":
            CONTROL_STATE["GLOBAL_PAUSE"] = False
            logger.info("ANDROID -> GLOBAL_RESUME: entries re-enabled.")
            return {"status": "ok", "detail": "GLOBAL_PAUSE = False — entries resumed"}

        # ── TIER 2: Per-Ticker Controls ───────────────────────────────
            ticker = payload.ticker.upper() if payload.ticker else ""
            if not ticker:
                return {"status": "error", "detail": "ticker is required for PAUSE_TICKER"}
            CONTROL_STATE["PAUSED_TICKERS"].add(ticker)
            logger.warning(f"ANDROID -> PAUSE_TICKER: {ticker} — entries suppressed, exits active")
            return {"status": "ok", "detail": f"{ticker} added to PAUSED_TICKERS"}

            ticker = payload.ticker.upper() if payload.ticker else ""
            if not ticker:
                return {"status": "error", "detail": "ticker is required for RESUME_TICKER"}
            CONTROL_STATE["PAUSED_TICKERS"].discard(ticker)
            logger.info(f"ANDROID -> RESUME_TICKER: {ticker} — entries re-enabled")
            return {"status": "ok", "detail": f"{ticker} removed from PAUSED_TICKERS"}

        # ── TIER 3: Soft Bias / Hunter Mode ────────────────────────────
            ticker    = payload.ticker.upper() if payload.ticker else ""
            direction = payload.direction.upper() if payload.direction else ""
            if not ticker or not direction:
                return {"status": "error", "detail": "ticker and direction required for BIAS"}
            if direction not in ("LONG", "SHORT", "CLEAR"):
                return {"status": "error", "detail": "direction must be LONG, SHORT, or CLEAR"}

            if direction == "CLEAR":
                CONTROL_STATE["BIAS"].pop(ticker, None)
                logger.info(f"ANDROID -> BIAS CLEAR {ticker} — Soft Bias removed")
                return {"status": "ok", "detail": f"{ticker} bias cleared"}

            CONTROL_STATE["BIAS"][ticker] = direction
            logger.info(
                f"ANDROID -> SOFT BIAS {ticker} = {direction} "
                f"(base_threshold=0.75 -> bias_threshold=0.65, opposing signals blocked)"
            )
            return {"status": "ok", "detail": f"{ticker} soft bias set to {direction}"}

        # ── INFO ──────────────────────────────────────────────────────────
        elif cmd == "STATUS":
            return {
                "status": "ok",
                "control_state": {
                    "GLOBAL_KILL":    CONTROL_STATE["GLOBAL_KILL"],
                    "GLOBAL_PAUSE":   CONTROL_STATE["GLOBAL_PAUSE"],
                    "PAUSED_TICKERS": list(CONTROL_STATE["PAUSED_TICKERS"]),
                    "BIAS":           dict(CONTROL_STATE["BIAS"]),
                },
                "timestamp": datetime.now().isoformat(),
            }

        else:
            return {"status": "error", "detail": f"Unknown command: {cmd}"}


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE 3B — WS /ws/telemetry & NEWS SPOOLER
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Convert dict to JSON string for transmission
        message_str = json.dumps(message, default=str)
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception:
                pass

manager = ConnectionManager()

async def automated_news_spooler():
    """
    Background task that polls news every 5 minutes (300s) and broadcasts
    significant sentiment shifts (> 0.50 absolute) to connected WebSockets.
    """
    if not news_engine:
        logger.warning("HybridNewsEngine not found. Spooler disabled.")
        return

    while True:
        try:
            # 1. Get truly active tickers (for Yahoo Finance to avoid rate limits)
            trades = _get_active_trades()
            active_tickers = list(set([t.get("symbol") for t in trades])) if trades else ["RELIANCE", "TCS", "INFY"]

            # 2. Get full 139 watch list (for broad RSS scraping)
            symbols_file = Path("assets/symbols.txt")
            if symbols_file.exists():
                with open(symbols_file, "r") as f:
                    watch_tickers = [line.strip().replace("NSE:", "") for line in f if line.strip()]
            else:
                watch_tickers = active_tickers

            # Use to_thread to prevent blocking the main asyncio event loop
            news_results = await asyncio.to_thread(news_engine.poll_all_news, active_tickers=active_tickers, watch_tickers=watch_tickers)
            
            # Update the cache for the 1s websocket telemetry telemetry
            global _latest_sentiment_cache
            new_cache = []
            
            for item in news_results:
                sentiment = item.get("sentiment_score", 0.0)
                
                # Add to cache for regular telemetry updates
                new_cache.append({
                    "ticker": item.get("ticker", "UNKNOWN"),
                    "headline": item.get("headline", ""),
                    "finbert_score": sentiment
                })
                
                # Also broadcast massive sentiment shifts instantly
                if abs(sentiment) > 0.50:
                    payload = {
                        "type": "NEWS_UPDATE",
                        "ticker": item.get("ticker", "UNKNOWN"),
                        "headline": item.get("headline", ""),
                        "sentiment_score": sentiment
                    }
                    await manager.broadcast(payload)
                    
            if new_cache:
                _latest_sentiment_cache = new_cache

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in automated_news_spooler: {e}")
            
        await asyncio.sleep(300) # Sleep for exactly 5 minutes


@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    """
    WebSocket endpoint — broadcasts a full engine telemetry payload every 1 second.

    Payload schema:
    {
      "timestamp":     "2026-02-25T09:32:00.123456",
      "live_pnl":      -230.45,
      "margin_usage":  { total_capital, available_margin, locked_margin, margin_usage_pct },
      "market_regime": "TRENDING",
      "sentiment_feed": [ { ticker, finbert_score }, ... ],
      "active_trades": [ { symbol, side, qty, entry_price, last_price, unrealized_pnl, locked_margin, entry_time }, ... ],
      "control_state": { GLOBAL_KILL, PAUSED, BIAS }
    }
    """
    await manager.connect(websocket)
    logger.info(f"WebSocket client connected: {websocket.client}")

    try:
        while True:
            # ── Collect telemetry ──────────────────────────────────────────
            if _simulator_ref is not None:
                live_pnl     = _simulator_ref.get_live_pnl()
                margin_usage = _simulator_ref.get_margin_usage()
            else:
                # Fallback: read from live_state.json written by the live engine
                _ls = _read_live_state()
                live_pnl     = _ls.get("live_pnl", 0.0)
                margin_usage = _ls.get("margin_usage", {
                    "total_capital":    0.0,
                    "available_margin": 0.0,
                    "locked_margin":    0.0,
                    "margin_usage_pct": 0.0,
                })

            # Safe snapshot of CONTROL_STATE (no async_lock needed for reads
            # of a dict — GIL makes simple dict access atomic in CPython)
            state_snapshot = {
                "GLOBAL_KILL":    CONTROL_STATE["GLOBAL_KILL"],
                "GLOBAL_PAUSE":   CONTROL_STATE["GLOBAL_PAUSE"],
                "PAUSED_TICKERS": list(CONTROL_STATE["PAUSED_TICKERS"]),
                "BIAS":           dict(CONTROL_STATE["BIAS"]),
            }

            payload = {
                "timestamp":      datetime.now().isoformat(),
                "live_pnl":       live_pnl,
                "margin_usage":   margin_usage,
                "market_regime":  compute_market_regime(),
                "sentiment_feed": _get_sentiment_feed(),
                "active_trades":  _get_active_trades(),
                "control_state":  state_snapshot,
            }

            await websocket.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(1)   # Zero-latency: non-blocking 1-second cadence

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        manager.disconnect(websocket)
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE 3C — GET /api/history
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Returns historical trades from paper_trades.csv with optional date filters.
    Format: YYYY-MM-DD
    """
    trades_log = config.LOGS_DIR / "paper_trades.csv"
    
    if not trades_log.exists():
        return {"status": "error", "detail": "No trade history found (CSV missing)"}

    try:
        df = pd.read_csv(trades_log)
        if df.empty:
            return []

        # Convert entry_time to datetime for filtering
        df['dt'] = pd.to_datetime(df['entry_time'])
        
        if start_date:
            try:
                sd = pd.to_datetime(start_date)
                df = df[df['dt'].dt.date >= sd.date()]
            except Exception as e:
                return {"status": "error", "detail": f"Invalid start_date format: {e}"}

        if end_date:
            try:
                ed = pd.to_datetime(end_date)
                df = df[df['dt'].dt.date <= ed.date()]
            except Exception as e:
                return {"status": "error", "detail": f"Invalid end_date format: {e}"}

        # Drop the temporary datetime column and handle NaNs for JSON
        df = df.drop(columns=['dt']).fillna("")
        
        return df.to_dict(orient="records")

    except Exception as e:
        logger.error(f"Failed to read trade history: {e}")
        return {"status": "error", "detail": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple liveness probe for the Android app."""
    return {
        "status":         "online",
        "engine":         "XGBoost+Renko Paper Trader",
        "simulator_live": _simulator_ref is not None,
        "timestamp":      datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE 3D — GET /api/news/refresh
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/news/refresh")
async def manual_news_refresh():
    """
    Manual override endpoint to immediately poll and broadcast news.
    """
    if not news_engine:
        return {"status": "error", "message": "HybridNewsEngine not configured."}
        
    try:
        # Clear cache so we actually rebroadcast recent news for the UI
        news_engine.processed_headlines.clear()

        # 1. Get truly active tickers (for Yahoo Finance)
        active_trades = _get_active_trades()
        active_tickers = list(set([t.get("symbol") for t in active_trades])) if active_trades else ["RELIANCE", "TCS", "INFY"]

        # 2. Get full 139 watch list (for broad RSS scraping)
        symbols_file = Path("assets/symbols.txt")
        if symbols_file.exists():
            with open(symbols_file, "r") as f:
                watch_tickers = [line.strip().replace("NSE:", "") for line in f if line.strip()]
        else:
            watch_tickers = active_tickers

        # Polling runs in a non-blocking thread
        news_results = await asyncio.to_thread(news_engine.poll_all_news, active_tickers=active_tickers, watch_tickers=watch_tickers)
        
        global _latest_sentiment_cache
        new_cache = []
        
        broadcast_count = 0
        for item in news_results:
            sentiment = item.get("sentiment_score", 0.0)
            
            new_cache.append({
                "ticker": item.get("ticker", "UNKNOWN"),
                "headline": item.get("headline", ""),
                "finbert_score": sentiment
            })
            
            if abs(sentiment) > 0.50:
                payload = {
                    "type": "NEWS_UPDATE",
                    "ticker": item.get("ticker", "UNKNOWN"),
                    "headline": item.get("headline", ""),
                    "sentiment_score": sentiment
                }
                await manager.broadcast(payload)
                broadcast_count += 1
                
        if new_cache:
            _latest_sentiment_cache = new_cache

        return {
            "status": "success", 
            "message": "News refreshed and broadcasted",
            "broadcast_count": broadcast_count
        }
        
    except Exception as e:
        logger.error(f"Manual news refresh failed: {e}")
        return {"status": "error", "message": str(e)}
