import asyncio
import logging
import pandas as pd
from typing import List, Dict

from trading_api import config
from trading_api.src.core.state import manager

try:
    from trading_core.core.physics.hybrid_news import HybridNewsEngine
except ImportError:
    HybridNewsEngine = None

logger = logging.getLogger(__name__)

# This cache is populated directly by the automated_news_spooler background task.
_latest_sentiment_cache: List[Dict] = []
news_engine = None

def get_sentiment_feed() -> List[Dict]:
    global _latest_sentiment_cache
    return _latest_sentiment_cache

def _get_watch_tickers() -> List[str]:
    try:
        df = pd.read_csv(config.UNIVERSE_CSV)
        stocks_df = df[df['is_index'].astype(str).str.lower() == 'false']
        return stocks_df['symbol'].tolist()
    except Exception as e:
        logger.error(f"Failed to load watch tickers from {config.UNIVERSE_CSV}: {e}")
        return ["RELIANCE", "TCS", "INFY"]

async def automated_news_spooler(get_active_trades_func):
    """
    Background task that polls news every 5 minutes and broadcasts shifts.
    """
    global news_engine, _latest_sentiment_cache
    
    if HybridNewsEngine and not news_engine:
        news_engine = HybridNewsEngine()

    if not news_engine:
        logger.warning("HybridNewsEngine not found. Spooler disabled.")
        return

    while True:
        try:
            trades = get_active_trades_func()
            active_tickers = list(set([t.get("symbol") for t in trades])) if trades else ["RELIANCE", "TCS", "INFY"]
            watch_tickers = _get_watch_tickers()

            news_results = await asyncio.to_thread(
                news_engine.poll_all_news, 
                active_tickers=active_tickers, 
                watch_tickers=watch_tickers
            )
            
            new_cache = []
            for item in news_results:
                sentiment = item.get("sentiment_score", 0.0)
                new_cache.append({
                    "ticker": item.get("ticker", "UNKNOWN"),
                    "headline": item.get("headline", ""),
                    "finbert_score": sentiment
                })
                
                if abs(sentiment) > config.SENTIMENT_THRESHOLD:
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
            
        await asyncio.sleep(config.NEWS_POLL_INTERVAL)

