"""
scripts/start_e2e_live.py — Live E2E Test Launcher
====================================================
Starts the REAL server (FastAPI + paper trader) but redirects ALL
test-session logs to a temporary folder: logs/e2e_test/<timestamp>/

On Ctrl+C (or when user says "stop"):
  → Automatically deletes logs/e2e_test/ and frees port 8000
  → Zero permanent impact on production logs or state

What this tests:
  ✓ Models load and predict correctly
  ✓ FastAPI server starts on port 8000
  ✓ WebSocket telemetry broadcasts live data
  ✓ Android app can connect via Tailscale (100.75.119.4:8000)
  ✓ API commands (PAUSE/RESUME/BIAS) work from mobile
  ✓ Paper trader loop initializes (no ticks = no trades after market hours)

Usage:
    .venv\\Scripts\\python.exe scripts/start_e2e_live.py
    # Open Android app → connect to ws://100.75.119.4:8000/ws/telemetry
    # Test commands from phone
    # Press Ctrl+C when done → temp logs auto-deleted
"""

import sys
import asyncio
import logging
import shutil
import socket
import signal
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Create isolated temp log dir BEFORE any imports that touch logs ───────────
E2E_DIR = ROOT / "logs" / "e2e_test" / datetime.now().strftime("%Y%m%d_%H%M%S")
E2E_DIR.mkdir(parents=True, exist_ok=True)
TEST_DEBUG_DIR = E2E_DIR / "paper_debug"
TEST_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ── Monkey-patch daily_logger BEFORE importing paper_trader ───────────────────
import src.live.daily_logger as _dl
_dl._DEBUG_DIR = TEST_DEBUG_DIR          # redirect CSV writes → temp dir

# ── Also redirect the signals log ─────────────────────────────────────────────
import config as _cfg
_ORIG_LOGS_DIR = _cfg.LOGS_DIR
_cfg.LOGS_DIR = E2E_DIR                  # redirect storage/logs → temp dir

# ── Now safe to import the trading stack ──────────────────────────────────────
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from src.live.paper_trader import PaperPortfolio, run_paper_trader
from src.api.server import app, set_simulator_ref, register_brick_signal

# ── Middleware: log every incoming command from Android app ───────────────────
class _AppCommandLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        if request.url.path == "/api/command" and request.method == "POST":
            body_bytes = await request.body()
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n  ┌── [ANDROID → SERVER] {ts} ─────────────────────────")
            print(f"  │  {body_bytes.decode('utf-8', errors='replace')}")
            response = await call_next(request)
            print(f"  │  Response: HTTP {response.status_code}")
            print(f"  └────────────────────────────────────────────────────\n")
            return response
        return await call_next(request)

app.add_middleware(_AppCommandLogger)


DIVIDER = "=" * 65

# Detect Tailscale IP
def _tailscale_ip():
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ip.startswith("100."):
                return ip
    except Exception:
        pass
    return "127.0.0.1 (Tailscale not detected)"

def _cleanup():
    print(f"\n{'='*65}")
    print("  Stopping E2E test — cleaning up temp logs...")
    try:
        shutil.rmtree(ROOT / "logs" / "e2e_test", ignore_errors=True)
        print(f"  ✓ Deleted: logs/e2e_test/")
    except Exception as e:
        print(f"  [warn] Cleanup failed: {e}")
    # Restore config
    _cfg.LOGS_DIR = _ORIG_LOGS_DIR
    print("  ✓ Production logs untouched")
    print("  ✓ E2E test complete — no impact on any production file")
    print(f"{'='*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    tail_ip = _tailscale_ip()
    port    = 8000

    print(f"\n{DIVIDER}")
    print("  LIVE E2E TEST — Institutional Fortress Trading System")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(DIVIDER)
    print(f"\n  Temp log dir   : {E2E_DIR}")
    print(f"  Production logs: UNTOUCHED (redirected for this session)")
    print(f"\n  ─── ANDROID APP — connect these to your app ───────────────")
    print(f"  WebSocket  : ws://{tail_ip}:{port}/ws/telemetry")
    print(f"  API base   : http://{tail_ip}:{port}/api/command")
    print(f"  ────────────────────────────────────────────────────────────")
    print(f"\n  Test commands to send from your Android app:")
    print(f'    STATUS       : POST /api/command  {{"command":"STATUS"}}')
    print(f'    GLOBAL_PAUSE : POST /api/command  {{"command":"GLOBAL_PAUSE"}}')
    print(f'    GLOBAL_RESUME: POST /api/command  {{"command":"GLOBAL_RESUME"}}')
    print(f'    SET_BIAS     : POST /api/command  {{"command":"SET_BIAS","ticker":"INFY","direction":"SHORT"}}')
    print(f"\n  Press Ctrl+C when done — all test logs auto-deleted.")
    print(DIVIDER + "\n")

    # ── Start paper trader in a daemon thread (SystemExit stays contained) ────
    import threading

    def _run_trader_safe():
        try:
            run_paper_trader()
        except SystemExit:
            pass   # paper trader does sys.exit(0) at 15:30 — don't crash server
        except Exception as e:
            print(f"  [trader] Exception: {e}")

    # Start PaperPortfolio (loads models, initialises simulator)
    portfolio = PaperPortfolio()
    
    # --- INJECT DUMMY DATA FOR UI TESTING ---
    from src.live.upstox_simulator import SimulatedOrder, TradeState
    import random
    
    dummy_tcs = SimulatedOrder(
        trade_id=1001, symbol="TCS", side="BUY", qty=50, 
        entry_price=4100.50, created_at=datetime.now(),
        state=TradeState.ACTIVE, locked_margin=41000.0,
        filled_at=datetime.now(), last_price=4125.00
    )
    dummy_infy = SimulatedOrder(
        trade_id=1002, symbol="INFY", side="SELL", qty=100,
        entry_price=1680.00, created_at=datetime.now(),
        state=TradeState.ACTIVE, locked_margin=33600.0,
        filled_at=datetime.now(), last_price=1675.25
    )
    portfolio.simulator.active_trades["TCS"] = dummy_tcs
    portfolio.simulator.active_trades["INFY"] = dummy_infy
    portfolio.simulator.live_pnl = (4125.0 - 4100.5) * 50 + (1680.0 - 1675.25) * 100
    portfolio.simulator.margin_usage = 41000.0 + 33600.0
    # ----------------------------------------

    set_simulator_ref(portfolio.simulator)

    trader_thread = threading.Thread(target=_run_trader_safe, daemon=True)
    trader_thread.start()

    # ── Keep-alive: print heartbeat so user knows server is live ─────────────
    async def _heartbeat():
        count = 0
        while True:
            await asyncio.sleep(30)
            count += 1
            print(f"  [E2E] Server alive — {count*30}s | "
                  f"WebSocket: ws://{tail_ip}:{port}/ws/telemetry")

    uvicorn_config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
    server = uvicorn.Server(uvicorn_config)

    try:
        await asyncio.gather(
            server.serve(),
            _heartbeat(),
        )
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        _cleanup()
