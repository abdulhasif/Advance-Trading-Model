import json
import logging
import asyncio
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from trading_api import config
from trading_api.src.core.state import manager, _simulator_ref
from trading_api.src.services.trade_service import get_active_trades, read_live_state
from trading_api.src.services.market_service import compute_market_regime
from trading_api.src.services.news_service import get_sentiment_feed
from trading_engine.src.control_state import CONTROL_STATE

logger = logging.getLogger(__name__)

async def telemetry_ws(websocket: WebSocket):
    """
    WebSocket endpoint - broadcasts full engine telemetry every 1 second.
    """
    await manager.connect(websocket)
    try:
        while True:
            if _simulator_ref is not None:
                live_pnl     = _simulator_ref.get_live_pnl()
                margin_usage = _simulator_ref.get_margin_usage()
            else:
                ls = read_live_state()
                live_pnl     = ls.get("live_pnl", 0.0)
                margin_usage = ls.get("margin_usage", {
                    "total_capital": 0.0, "available_margin": 0.0, "locked_margin": 0.0, "margin_usage_pct": 0.0
                })

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
                "market_regime":  compute_market_regime(_simulator_ref),
                "sentiment_feed": get_sentiment_feed(),
                "active_trades":  get_active_trades(_simulator_ref),
                "control_state":  state_snapshot,
            }
            await websocket.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
        logger.error(f"WebSocket error: {e}")

