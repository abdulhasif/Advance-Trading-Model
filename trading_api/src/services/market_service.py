import time
import logging
from collections import deque
from typing import Optional, Dict

from trading_api import config
from trading_engine.src.upstox_simulator import UpstoxSimulator

logger = logging.getLogger(__name__)

# Maintains a rolling window of brick directions to compute regime on the fly.
_regime_buffer: deque = deque(maxlen=config.REGIME_WINDOW)   # last X brick direction signals

def register_brick_signal(direction: int, conviction: float) -> None:
    """
    Called from the trading loop each time a new brick fires.
    direction: +1 or -1.
    """
    _regime_buffer.append({
        "dir": direction, 
        "conv": conviction,
        "ts": time.time()
    })

def compute_market_regime(simulator_ref: Optional[UpstoxSimulator] = None, live_state: Optional[Dict] = None) -> str:
    """
    Derive market regime from the rolling brick direction buffer.
    """
    if simulator_ref is None and live_state:
        return live_state.get("market_regime", "SIDEWAYS")

    now_ts = time.time()
    valid_signals = [b for b in _regime_buffer if now_ts - b["ts"] < 300]

    if len(valid_signals) < config.REGIME_MIN_SIGNALS:
        return "SIDEWAYS"

    directions  = [b["dir"] for b in valid_signals]
    convictions = [b["conv"] for b in valid_signals]
    longs  = directions.count(1)
    shorts = directions.count(-1)
    total  = len(directions)
    net_bias    = abs(longs - shorts) / total * 100
    avg_conv    = sum(convictions) / len(convictions)

    if net_bias > config.REGIME_BIAS_TRENDING or avg_conv > config.REGIME_CONV_TRENDING:
        return "TRENDING"
    if net_bias >= config.REGIME_BIAS_VOLATILE and avg_conv < config.REGIME_CONV_VOLATILE:
        return "VOLATILE"
    return "SIDEWAYS"
