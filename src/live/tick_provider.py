"""
src/live/tick_provider.py — WebSocket Tick Feed (Placeholder)
==============================================================
Replace get_latest_ticks() with real Upstox WebSocket parsing.
"""

import logging
import random
from datetime import datetime

import config

logger = logging.getLogger(__name__)


class TickProvider:
    """
    Placeholder for Upstox WebSocket tick feed.

    Expected output from get_latest_ticks():
      { "SBIN": {"ltp": 625.5, "high": 626.0, "low": 624.8, "timestamp": ...}, ... }
    """

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self._connected = False

    def connect(self):
        """[PLACEHOLDER] Connect to Upstox WebSocket."""
        logger.info("WebSocket PLACEHOLDER — simulating connection")
        self._connected = True

    def disconnect(self):
        self._connected = False
        logger.info("WebSocket disconnected")

    def get_latest_ticks(self) -> dict:
        """[PLACEHOLDER] Returns simulated ticks."""
        now = datetime.now()
        ticks = {}
        for sym in self.symbols:
            base = random.uniform(100, 5000)
            ticks[sym] = {
                "ltp": base,
                "high": base + random.uniform(0, base * 0.002),
                "low": base - random.uniform(0, base * 0.002),
                "timestamp": now,
            }
        return ticks
