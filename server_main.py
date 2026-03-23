"""
server_main.py — Zero-Latency Async Entry Point
================================================
Runs the FastAPI server (uvicorn) and the paper trading loop concurrently
using asyncio.gather(). Neither blocks the other.

Architecture:
  • asyncio event loop  -> uvicorn serves the FastAPI app
  • ThreadPool executor -> blocking while True trading loop runs in a thread
                          (asyncio.to_thread wraps it, preserving GIL safety)

Tailscale binding: host="0.0.0.0" makes the server reachable from any
network interface, including the Tailscale virtual NIC at 100.x.x.x.

Run:
  python server_main.py
"""

import asyncio
import logging
import sys
import uvicorn

from src.api.server import app, set_simulator_ref

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ─────────────────────────────────────────────────────────────────────────────
# ASYNC MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Bootstraps both coroutines and runs them concurrently, OR
    runs just the API if --api-only is passed.
    """
    api_only = "--api-only" in sys.argv

    logger.info("=" * 65)
    if api_only:
        logger.info(" SERVER MAIN — API Server ONLY (Reading from live_state.json)")
    else:
        logger.info(" SERVER MAIN — XGBoost+Renko Async Engine Starting")
    logger.info("=" * 65)

    if not api_only:
        # ── Step 1: Create simulator (replaces legacy PaperPortfolio) ────────
        from src.live.upstox_simulator import UpstoxSimulator
        import config
        
        portfolio = UpstoxSimulator(starting_capital=config.STARTING_CAPITAL)

        # ── Step 2: Inject simulator reference into the API server ───────────
        set_simulator_ref(portfolio)
        logger.info("Simulator reference injected into API server.")

    # ── Step 3: Build uvicorn config ─────────────────────────────────────
    uvicorn_config = uvicorn.Config(
        app=app,
        host="0.0.0.0",     # Reachable via Tailscale + LAN
        port=8000,
        log_level="info",
        ws_ping_interval=20,   # Keep WebSocket alive through Tailscale NAT
        ws_ping_timeout=30,
    )
    server = uvicorn.Server(uvicorn_config)

    logger.info("Launching: uvicorn on 0.0.0.0:8000")
    logger.info("Android app: connect WebSocket to ws://<tailscale-ip>:8000/ws/telemetry")
    logger.info("Android app: POST commands to http://<tailscale-ip>:8000/api/command")

    if api_only:
        # Run just the server (avoids second Upstox WS connection)
        await server.serve()
    else:
        # ── Step 4: Run both concurrently ────────────────────────────────────
        # asyncio.to_thread pushes the blocking while True loop into a
        # ThreadPoolExecutor so the event loop remains free for uvicorn.
        from src.live.engine import run_live_engine
        await asyncio.gather(
            server.serve(),                              # FastAPI (async, non-blocking)
            asyncio.to_thread(run_live_engine),         # Trading loop (sync -> thread)
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("server_main.py — Shutting down gracefully.")
