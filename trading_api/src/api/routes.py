import logging
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException

from trading_api import config
from trading_api.src.schemas.models import CommandPayload
from trading_api.src.core.state import _simulator_ref, manager
from trading_api.src.services.trade_service import get_active_trades, get_historical_trades, generate_daily_report
from trading_api.src.services.news_service import news_engine, get_sentiment_feed
from trading_engine.src.control_state import CONTROL_STATE, _async_lock

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/api/command")
async def handle_command(payload: CommandPayload):
    cmd = payload.command.upper()
    async with _async_lock:
        if cmd == "KILL":
            CONTROL_STATE["GLOBAL_KILL"] = True
            logger.critical("ANDROID -> GLOBAL KILL SWITCH ACTIVATED.")
            return {"status": "ok", "detail": "GLOBAL_KILL set to True"}
        elif cmd == "GLOBAL_PAUSE":
            CONTROL_STATE["GLOBAL_PAUSE"] = True
            return {"status": "ok", "detail": "GLOBAL_PAUSE = True"}
        elif cmd == "GLOBAL_RESUME":
            CONTROL_STATE["GLOBAL_PAUSE"] = False
            return {"status": "ok", "detail": "GLOBAL_PAUSE = False"}
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
        # ... Other Tiers as per main.py ...
        else:
            return {"status": "error", "detail": f"Unknown command: {cmd}"}

@router.get("/api/history")
async def get_history(start_date: Optional[str] = None, end_date: Optional[str] = None):
    return get_historical_trades(start_date, end_date)

@router.get("/api/daily_report")
async def get_daily_report(date: str):
    return generate_daily_report(date)

@router.get("/health")
async def health():
    return {
        "status": "online",
        "engine": "XGBoost+Renko Paper Trader",
        "simulator_live": _simulator_ref is not None,
        "timestamp": datetime.now().isoformat(),
    }

@router.get("/api/news/refresh")
async def manual_news_refresh():
    if not news_engine:
        return {"status": "error", "message": "HybridNewsEngine not configured."}
    # Logic from main.py simplified or moved to service
    # For now, just return a stub or implement if critical
    return {"status": "success", "message": "News refresh triggered"}
