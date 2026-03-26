import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trading_api import config
from trading_api.src.api.routes import router
from trading_api.src.api.websocket import telemetry_ws
from trading_api.src.services.news_service import automated_news_spooler
from trading_api.src.services.trade_service import get_active_trades
from trading_api.src.core.state import set_simulator_ref

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading Engine Control API",
    description="Modularized FastAPI bridge for the Institutional Trading System.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.add_api_websocket_route("/ws/telemetry", telemetry_ws)

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the server boots."""
    logger.info("Initializing background tasks...")
    # Inject lambda to get active trades to avoid circular imports
    asyncio.create_task(automated_news_spooler(get_active_trades))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
