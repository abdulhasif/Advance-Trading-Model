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
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.live.control_state import CONTROL_STATE, _async_lock
from src.live.upstox_simulator import UpstoxSimulator

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
# SENTIMENT FEED SCAFFOLD
# ─────────────────────────────────────────────────────────────────────────────
# FinBERT requires ~4 GB GPU memory. In Paper Trading mode the engine returns
# placeholder scores of 0.5.  When you add a GPU node, replace the body of
# _get_sentiment_feed() with actual HuggingFace inference.
_STATIC_TICKERS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "LT", "SBIN", "WIPRO", "AXISBANK", "BAJFINANCE",
]


def _get_sentiment_feed() -> list[dict]:
    """
    Returns per-ticker FinBERT scores.
    Scaffold: returns neutral 0.50 for each ticker in the watchlist.

    Future: replace with:
        from transformers import pipeline
        pipe = pipeline("text-classification", model="ProsusAI/finbert")
        ... fetch headlines ... run inference ...
    """
    return [{"ticker": t, "finbert_score": 0.50} for t in _STATIC_TICKERS]


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE TRADES SNAPSHOT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _get_active_trades() -> list[dict]:
    """Serializes UpstoxSimulator.active_trades for the telemetry broadcast."""
    if _simulator_ref is None:
        return []
    trades = []
    for sym, order in _simulator_ref.active_trades.items():
        trades.append({
            "symbol":         sym,
            "side":           order.side,           # "BUY" or "SELL"
            "qty":            order.qty,
            "entry_price":    round(order.entry_price, 2),
            "last_price":     round(order.last_price, 2),
            "unrealized_pnl": round(order.unrealized_pnl, 2),
            "locked_margin":  round(order.locked_margin, 2),
            "entry_time":     order.filled_at.isoformat() if order.filled_at else None,
        })
    return trades


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
            logger.critical("ANDROID → GLOBAL KILL SWITCH ACTIVATED.")
            return {"status": "ok", "detail": "GLOBAL_KILL set to True — squaring off all positions"}

        elif cmd == "GLOBAL_PAUSE":
            CONTROL_STATE["GLOBAL_PAUSE"] = True
            logger.warning("ANDROID → GLOBAL_PAUSE: all new entries suppressed.")
            return {"status": "ok", "detail": "GLOBAL_PAUSE = True — all entries suspended"}

        elif cmd == "GLOBAL_RESUME":
            CONTROL_STATE["GLOBAL_PAUSE"] = False
            logger.info("ANDROID → GLOBAL_RESUME: entries re-enabled.")
            return {"status": "ok", "detail": "GLOBAL_PAUSE = False — entries resumed"}

        # ── TIER 2: Per-Ticker Controls ───────────────────────────────
        elif cmd == "PAUSE_TICKER":
            if not payload.ticker:
                return {"status": "error", "detail": "ticker is required for PAUSE_TICKER"}
            ticker = payload.ticker.upper()
            CONTROL_STATE["PAUSED_TICKERS"].add(ticker)
            logger.warning(f"ANDROID → PAUSE_TICKER: {ticker} — entries suppressed, exits active")
            return {"status": "ok", "detail": f"{ticker} added to PAUSED_TICKERS"}

        elif cmd == "RESUME_TICKER":
            if not payload.ticker:
                return {"status": "error", "detail": "ticker is required for RESUME_TICKER"}
            ticker = payload.ticker.upper()
            CONTROL_STATE["PAUSED_TICKERS"].discard(ticker)
            logger.info(f"ANDROID → RESUME_TICKER: {ticker} — entries re-enabled")
            return {"status": "ok", "detail": f"{ticker} removed from PAUSED_TICKERS"}

        # ── TIER 3: Soft Bias / Hunter Mode ────────────────────────────
        elif cmd == "BIAS":
            if not payload.ticker or not payload.direction:
                return {"status": "error", "detail": "ticker and direction required for BIAS"}
            ticker    = payload.ticker.upper()
            direction = payload.direction.upper()
            if direction not in ("LONG", "SHORT", "CLEAR"):
                return {"status": "error", "detail": "direction must be LONG, SHORT, or CLEAR"}

            if direction == "CLEAR":
                CONTROL_STATE["BIAS"].pop(ticker, None)
                logger.info(f"ANDROID → BIAS CLEAR {ticker} — Soft Bias removed")
                return {"status": "ok", "detail": f"{ticker} bias cleared"}

            CONTROL_STATE["BIAS"][ticker] = direction
            logger.info(
                f"ANDROID → SOFT BIAS {ticker} = {direction} "
                f"(base_threshold=0.75 → bias_threshold=0.65, opposing signals blocked)"
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
# DELIVERABLE 3B — WS /ws/telemetry
# ─────────────────────────────────────────────────────────────────────────────

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
    await websocket.accept()
    logger.info(f"WebSocket client connected: {websocket.client}")

    try:
        while True:
            # ── Collect telemetry ──────────────────────────────────────────
            if _simulator_ref is not None:
                live_pnl     = _simulator_ref.get_live_pnl()
                margin_usage = _simulator_ref.get_margin_usage()
            else:
                live_pnl     = 0.0
                margin_usage = {
                    "total_capital":    0.0,
                    "available_margin": 0.0,
                    "locked_margin":    0.0,
                    "margin_usage_pct": 0.0,
                }

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
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


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
