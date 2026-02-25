"""
scripts/test_app_connectivity.py — Full API + WebSocket Connectivity Test
=========================================================================
Starts the FastAPI server in a background thread (NO trading loop),
fires every endpoint the Android app uses, then shuts down cleanly.

Zero side-effects:
  • Uses a throw-away in-memory simulator ref (no real portfolio)
  • No writes to production logs
  • Server auto-shuts after tests complete

Usage:
    .venv\\Scripts\\python.exe scripts/test_app_connectivity.py
"""

import sys
import json
import time
import threading
import asyncio
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Suppress noisy logs from uvicorn during test
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)

import uvicorn
import requests
import websockets

DIVIDER = "=" * 60
HOST    = "127.0.0.1"
PORT    = 8000
BASE    = f"http://{HOST}:{PORT}"
WS_URL  = f"ws://{HOST}:{PORT}/ws/telemetry"

results = []

def check(label, ok, detail=""):
    tag = "  [OK] " if ok else "  [!!] "
    print(f"{tag} {label:<42} {detail}")
    results.append((ok, label))


# ── 1. Start server in background thread ─────────────────────────────────────

def _start_server():
    """Run uvicorn with just the FastAPI app — no trading loop, no portfolio."""
    from src.api.server import app, set_simulator_ref

    # Inject a minimal stub simulator so the WebSocket doesn't crash
    class _StubSimulator:
        live_pnl     = 0.0
        margin_usage = 0.0
        realized_pnl = 0.0
        peak_capital = 100_000
        active_trades = {}
        daily_trades  = []
        def record_daily_summary(self): pass
        def get_live_pnl(self): return 0.0
        def get_margin_usage(self): return 0.0
        def get_realized_pnl(self): return 0.0

    set_simulator_ref(_StubSimulator())

    config = uvicorn.Config(app=app, host=HOST, port=PORT,
                            log_level="error", loop="asyncio")
    server = uvicorn.Server(config)
    # Run in this thread's own event loop
    asyncio.run(server.serve())


print(f"\n{DIVIDER}")
print("  APP CONNECTIVITY TEST — FastAPI + WebSocket")
print(DIVIDER)
print("\n  Starting FastAPI server in background thread...")

server_thread = threading.Thread(target=_start_server, daemon=True)
server_thread.start()

# Wait for server to be ready
for _ in range(20):
    try:
        r = requests.get(f"{BASE}/docs", timeout=1)
        if r.status_code < 500:
            break
    except Exception:
        time.sleep(0.3)
else:
    print("  [!!] Server failed to start in 6 seconds. Check port 8000 is free.")
    sys.exit(1)

print("  Server ready.\n")


# ── 2. Test REST endpoints ────────────────────────────────────────────────────

print("[ REST ENDPOINTS ]")

# GET /docs (OpenAPI — confirms FastAPI is alive)
try:
    r = requests.get(f"{BASE}/docs", timeout=3)
    check("/docs (FastAPI UI)", r.status_code == 200, f"HTTP {r.status_code}")
except Exception as e:
    check("/docs (FastAPI UI)", False, str(e)[:50])

# GET /health (if exists)
try:
    r = requests.get(f"{BASE}/health", timeout=3)
    check("/health", r.status_code in (200, 404),
          f"HTTP {r.status_code}" + (" (endpoint present)" if r.status_code == 200 else " (not implemented — OK)"))
except Exception as e:
    check("/health", False, str(e)[:50])

# POST /api/command — STATUS
try:
    r = requests.post(f"{BASE}/api/command",
                      json={"command": "STATUS"}, timeout=3)
    ok   = r.status_code == 200
    body = r.json() if ok else {}
    check("POST /api/command STATUS", ok,
          f"HTTP {r.status_code} | keys: {list(body.keys())[:4]}" if ok else str(r.status_code))
except Exception as e:
    check("POST /api/command STATUS", False, str(e)[:50])

# POST /api/command — GLOBAL_PAUSE
try:
    r = requests.post(f"{BASE}/api/command",
                      json={"command": "GLOBAL_PAUSE"}, timeout=3)
    check("POST /api/command GLOBAL_PAUSE", r.status_code == 200,
          f"HTTP {r.status_code}")
except Exception as e:
    check("POST /api/command GLOBAL_PAUSE", False, str(e)[:50])

# POST /api/command — GLOBAL_RESUME
try:
    r = requests.post(f"{BASE}/api/command",
                      json={"command": "GLOBAL_RESUME"}, timeout=3)
    check("POST /api/command GLOBAL_RESUME", r.status_code == 200,
          f"HTTP {r.status_code}")
except Exception as e:
    check("POST /api/command GLOBAL_RESUME", False, str(e)[:50])

# POST /api/command — SET_BIAS
try:
    r = requests.post(f"{BASE}/api/command",
                      json={"command": "SET_BIAS", "ticker": "INFY", "direction": "SHORT"},
                      timeout=3)
    check("POST /api/command SET_BIAS", r.status_code == 200,
          f"HTTP {r.status_code}")
except Exception as e:
    check("POST /api/command SET_BIAS", False, str(e)[:50])

# POST /api/command — CLEAR_BIAS
try:
    r = requests.post(f"{BASE}/api/command",
                      json={"command": "CLEAR_BIAS", "ticker": "INFY"},
                      timeout=3)
    check("POST /api/command CLEAR_BIAS", r.status_code == 200,
          f"HTTP {r.status_code}")
except Exception as e:
    check("POST /api/command CLEAR_BIAS", False, str(e)[:50])


# ── 3. Test WebSocket /ws/telemetry ──────────────────────────────────────────

print("\n[ WEBSOCKET /ws/telemetry ]")

async def test_websocket():
    try:
        async with websockets.connect(WS_URL, open_timeout=5) as ws:
            check("WebSocket connect", True, WS_URL)
            # Receive first telemetry frame
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            payload = json.loads(msg)
            has_ts   = "timestamp"     in payload
            has_pnl  = "live_pnl"     in payload
            has_ctrl = "control_state" in payload
            has_trd  = "active_trades" in payload
            check("Telemetry frame received", True,
                  f"{len(msg)} bytes")
            check("timestamp field present",     has_ts,   "OK" if has_ts else "MISSING")
            check("live_pnl field present",      has_pnl,  "OK" if has_pnl else "MISSING")
            check("control_state field present", has_ctrl, "OK" if has_ctrl else "MISSING")
            check("active_trades field present", has_trd,  "OK" if has_trd else "MISSING")

            # Verify control_state reflects our GLOBAL_RESUME command from above
            ctrl = payload.get("control_state", {})
            check("GLOBAL_KILL = False (engine running)", not ctrl.get("GLOBAL_KILL", True),
                  str(ctrl))

            print(f"\n  Telemetry snapshot:")
            print(f"    timestamp    : {payload.get('timestamp','?')}")
            print(f"    live_pnl     : {payload.get('live_pnl', '?')}")
            print(f"    margin_usage : {payload.get('margin_usage', '?')}")
            print(f"    regime       : {payload.get('regime', '?')}")
            print(f"    control_state: {ctrl}")

    except Exception as e:
        check("WebSocket connect", False, str(e)[:60])

asyncio.run(test_websocket())


# ── 4. Tailscale IP detection ─────────────────────────────────────────────────

print("\n[ TAILSCALE / NETWORK ]")
import socket
try:
    tailscale_ip = None
    for iface_info in socket.getaddrinfo(socket.gethostname(), None):
        ip = iface_info[4][0]
        if ip.startswith("100."):    # Tailscale range
            tailscale_ip = ip
            break
    if tailscale_ip:
        check("Tailscale IP detected", True, f"Android app URL: ws://{tailscale_ip}:{PORT}/ws/telemetry")
        print(f"\n  ┌─ ANDROID APP SETTINGS ──────────────────────────────────┐")
        print(f"  │  WebSocket : ws://{tailscale_ip}:{PORT}/ws/telemetry   │")
        print(f"  │  API base  : http://{tailscale_ip}:{PORT}/api/command  │")
        print(f"  └──────────────────────────────────────────────────────────┘")
    else:
        check("Tailscale IP detected", False,
              "Not found — connect Tailscale VPN then retest", )
        print("  Hint: Start Tailscale on this PC → Android app will find it at 100.x.x.x:8000")
except Exception as e:
    check("Tailscale IP detection", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────

passed = sum(1 for ok, _ in results if ok)
failed = len(results) - passed

print(f"\n{DIVIDER}")
if failed == 0:
    print(f"  ✅  ALL {passed} CONNECTIVITY CHECKS PASSED")
    print(f"  Android app can connect. Start server_main.py tomorrow morning.")
else:
    print(f"  ❌  {failed} CHECK(S) FAILED")
    for ok, label in results:
        if not ok:
            print(f"       → {label}")
print(f"{DIVIDER}\n")
sys.exit(0 if failed == 0 else 1)
