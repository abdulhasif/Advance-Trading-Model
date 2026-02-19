"""
src/live/tick_provider.py -- Real-Time Tick Feed via Upstox WebSocket
======================================================================
Connects to Upstox Market Data WebSocket for live tick prices.
Falls back to simulated random ticks if UPSTOX_ACCESS_TOKEN is not set.

Required:  pip install upstox-python-sdk
Or:        pip install websockets protobuf requests

The provider maps trading symbols (e.g. "SBIN") to Upstox instrument keys
(e.g. "NSE_EQ|INE062A01020") using the sector_universe.csv file.
"""

import os
import sys
import json
import time
import logging
import random
import threading
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


# =============================================================================
# UPSTOX WEBSOCKET TICK PROVIDER
# =============================================================================
class TickProvider:
    """
    Real-time tick provider for the trading engine.

    Behavior:
      - If UPSTOX_ACCESS_TOKEN is set -> connects to Upstox WebSocket
      - Otherwise -> falls back to simulated random ticks (paper testing)

    Output from get_latest_ticks():
      { "SBIN": {"ltp": 625.5, "high": 626.0, "low": 624.8, "timestamp": ...}, ... }
    """

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self._connected = False
        self._ticks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

        # Check if we have a real access token
        self._access_token = config.UPSTOX_ACCESS_TOKEN
        self._use_live = bool(self._access_token and self._access_token.strip())

        # Build symbol <-> instrument_key mapping from universe CSV
        self._sym_to_ikey: dict[str, str] = {}
        self._ikey_to_sym: dict[str, str] = {}
        self._load_instrument_map()

    def _load_instrument_map(self):
        """Load symbol to instrument_key mapping from sector_universe.csv."""
        import pandas as pd
        try:
            df = pd.read_csv(config.UNIVERSE_CSV)
            for _, row in df.iterrows():
                sym = row["symbol"]
                ikey = row["instrument_key"]
                if sym in self.symbols:
                    self._sym_to_ikey[sym] = ikey
                    self._ikey_to_sym[ikey] = sym
            logger.info(f"Mapped {len(self._sym_to_ikey)}/{len(self.symbols)} "
                        f"symbols to instrument keys")
        except Exception as e:
            logger.warning(f"Could not load instrument map: {e}")

    # ── Connection ──────────────────────────────────────────────────────────
    def connect(self):
        """Connect to data source."""
        if self._use_live:
            self._connect_upstox()
        else:
            logger.info("No UPSTOX_ACCESS_TOKEN found -- using SIMULATED ticks")
            logger.info("Set env var UPSTOX_ACCESS_TOKEN for live market data")
            self._connected = True

    def _connect_upstox(self):
        """Connect to Upstox WebSocket using the official SDK."""
        try:
            import upstox_client
            from upstox_client.feeder import MarketDataStreamerV3

            # Configure API client
            configuration = upstox_client.Configuration()
            configuration.access_token = self._access_token
            api_client = upstox_client.ApiClient(configuration)

            # Get instrument keys for our symbols
            instrument_keys = [
                self._sym_to_ikey[sym] for sym in self.symbols
                if sym in self._sym_to_ikey
            ]

            if not instrument_keys:
                logger.error("No instrument keys mapped. Check sector_universe.csv")
                self._use_live = False
                self._connected = True
                return

            logger.info(f"Connecting to Upstox WebSocket with "
                        f"{len(instrument_keys)} instruments...")

            # Create streamer with LTPC mode (lightest: last price + close)
            self._streamer = MarketDataStreamerV3(
                api_client,
                instrumentKeys=instrument_keys,
                mode="ltpc"
            )

            # Register event handlers
            self._streamer.on("message", self._on_message)
            self._streamer.on("open", self._on_open)
            self._streamer.on("error", self._on_error)
            self._streamer.on("close", self._on_close)

            # Connect in background thread
            self._running = True
            self._ws_thread = threading.Thread(
                target=self._streamer.connect,
                daemon=True,
                name="UpstoxWebSocket"
            )
            self._ws_thread.start()

            # Wait for connection (up to 10 seconds)
            for _ in range(100):
                if self._connected:
                    break
                time.sleep(0.1)

            if not self._connected:
                logger.warning("WebSocket connection timed out. Using simulated ticks.")
                self._use_live = False
                self._connected = True

        except ImportError:
            logger.warning("upstox-python-sdk not installed. "
                          "Run: pip install upstox-python-sdk")
            logger.info("Falling back to SIMULATED ticks")
            self._use_live = False
            self._connected = True

        except Exception as e:
            logger.error(f"Upstox WebSocket connection failed: {e}")
            logger.info("Falling back to SIMULATED ticks")
            self._use_live = False
            self._connected = True

    # ── WebSocket Event Handlers ────────────────────────────────────────────
    def _on_open(self, *args, **kwargs):
        logger.info("Upstox WebSocket CONNECTED -- receiving live ticks")
        self._connected = True

    def _on_error(self, *args, **kwargs):
        logger.error(f"Upstox WebSocket error: {args}")

    def _on_close(self, *args, **kwargs):
        logger.info("Upstox WebSocket connection closed")
        self._connected = False

    def _on_message(self, message):
        """Process incoming market data message from Upstox.
        
        Upstox SDK v3 sends messages as plain dicts:
        {
            'type': 'live_feed',
            'feeds': {
                'NSE_EQ|INE062A01020': {
                    'ltpc': {'ltp': 625.5, 'cp': 620.0, 'ltt': '...', 'ltq': '7'}
                }
            }
        }
        """
        try:
            # SDK v3 sends plain dicts, not protobuf objects
            feeds = None
            if isinstance(message, dict):
                feeds = message.get("feeds")
            elif hasattr(message, "feeds"):
                feeds = message.feeds

            if not feeds:
                return

            now = datetime.now()
            count = 0
            with self._lock:
                for ikey, feed in feeds.items():
                    sym = self._ikey_to_sym.get(ikey)
                    if not sym:
                        continue

                    # Dict-based feed (SDK v3)
                    if isinstance(feed, dict):
                        # LTPC mode
                        ltpc = feed.get("ltpc")
                        if ltpc:
                            ltp = float(ltpc.get("ltp", 0))
                            if ltp > 0:
                                self._ticks[sym] = {
                                    "ltp": ltp,
                                    "high": ltp,
                                    "low": ltp,
                                    "close": float(ltpc.get("cp", ltp)),
                                    "timestamp": now,
                                }
                                count += 1

                        # Full-feed mode (if subscribed to full)
                        ff = feed.get("ff")
                        if ff:
                            mff = ff.get("marketFF", {})
                            ltpc_data = mff.get("ltpc", {})
                            ohlc_list = mff.get("marketOHLC", {}).get("ohlc", [])
                            ohlc = ohlc_list[0] if ohlc_list else {}

                            ltp = float(ltpc_data.get("ltp", 0))
                            if ltp > 0 and ohlc:
                                self._ticks[sym] = {
                                    "ltp": ltp,
                                    "high": float(ohlc.get("high", ltp)),
                                    "low": float(ohlc.get("low", ltp)),
                                    "close": float(ohlc.get("close", ltp)),
                                    "timestamp": now,
                                }
                                count += 1

                    # Protobuf-based feed (legacy SDK)
                    else:
                        ltpc = getattr(feed, "ltpc", None)
                        if ltpc:
                            ltp = float(ltpc.ltp) if ltpc.ltp else 0.0
                            self._ticks[sym] = {
                                "ltp": ltp,
                                "high": ltp,
                                "low": ltp,
                                "close": float(ltpc.cp) if ltpc.cp else ltp,
                                "timestamp": now,
                            }
                            count += 1

            # Log periodically
            if not hasattr(self, "_msg_count"):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count <= 3 or self._msg_count % 500 == 0:
                logger.info(f"WS ticks #{self._msg_count}: {count} symbols updated, "
                            f"total tracked: {len(self._ticks)}")
        except Exception as e:
            logger.warning(f"Tick parse error: {e}")

    # ── Public Interface ────────────────────────────────────────────────────
    def disconnect(self):
        """Disconnect from data source."""
        self._running = False
        if self._use_live and hasattr(self, "_streamer"):
            try:
                self._streamer.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Tick provider disconnected")

    def get_latest_ticks(self) -> dict:
        """
        Returns latest tick data for all symbols.

        Returns:
            { "SBIN": {"ltp": 625.5, "high": 626.0, "low": 624.8,
                        "timestamp": datetime}, ... }
        """
        if self._use_live:
            return self._get_live_ticks()
        else:
            return self._get_simulated_ticks()

    def _get_live_ticks(self) -> dict:
        """Return latest ticks from WebSocket buffer."""
        with self._lock:
            return dict(self._ticks)  # Return copy

    def _get_simulated_ticks(self) -> dict:
        """Generate simulated random ticks (placeholder mode)."""
        now = datetime.now()
        ticks = {}

        for sym in self.symbols:
            # Use persistent base price per symbol for realistic simulation
            if sym not in self._ticks:
                self._ticks[sym] = {
                    "_base": random.uniform(100, 5000),
                    "ltp": 0, "high": 0, "low": 0, "timestamp": now,
                }

            base = self._ticks[sym].get("_base", random.uniform(100, 5000))
            # Random walk: +-0.3% per tick
            change = random.gauss(0, base * 0.003)
            base += change
            self._ticks[sym]["_base"] = base

            ticks[sym] = {
                "ltp": base,
                "high": base + random.uniform(0, base * 0.002),
                "low": base - random.uniform(0, base * 0.002),
                "timestamp": now,
            }
        return ticks

    @property
    def is_live(self) -> bool:
        """Whether we're connected to real market data."""
        return self._use_live and self._connected
