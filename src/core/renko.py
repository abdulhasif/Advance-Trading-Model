"""
src/core/renko.py — Renko Brick Builder v3.0 (Institutional Physics Engine)
=============================================================================
Converts 1-minute OHLC data into Renko bricks using NATR brick size (0.15% of price).
Implements 4 Sim2Real fixes for brutal real-world accuracy:

  Fix 1: Volume Passthrough — proportional volume distribution per brick.
          Required for VWAP computation in features.py.

  Fix 2: Brownian Bridge Sub-Grid — when candle_range >= 2x brick_size,
          simulate sub-second intra-minute ticks via a pinned Brownian Bridge.
          Eliminates "Backtest Mirage" (1 candle != 1 tick in live).

  Fix 3: True Wick Carry-Forward — overshoot beyond brick boundary feeds
          the NEXT brick's wick tracking (not discarded). Prevents the
          "Phantom Wick" problem where bricks show 0 shadow.

  Fix 4: Path-Conflict Resolution — if both target (N bricks up) and stop
          loss are hit within the same 1-minute interpolated candle path,
          record the outcome as a LOSS (conservative/pessimistic).

Used by:
  • src/data/batch_factory.py  (historical bulk transform)
  • src/live/engine.py         (real-time incremental bricks)
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Optional

import config


# =============================================================================
# BROWNIAN BRIDGE SUB-TICK GENERATOR
# =============================================================================

def _brownian_bridge(
    p_start: float,
    p_end: float,
    n_steps: int,
    sigma: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Brownian Bridge pinned at p_start and p_end with n_steps
    intermediate points. Sigma is derived from the candle's High-Low range.

    The bridge ensures:
      - path[0]  = p_start (anchored to Open)
      - path[-1] = p_end   (anchored to Close)
      - intra-path volatility ∝ sigma (the candle's H-L range)

    This correctly simulates the WebSocket tick stream a live engine sees
    within a single 1-minute candle — eliminating the Backtest Mirage.

    Args:
        p_start:  Starting price (candle Open or intermediate waypoint).
        p_end:    Ending price   (candle Close or intermediate waypoint).
        n_steps:  Number of sub-tick points to generate.
        sigma:    Volatility scale = (candle_high - candle_low) * 0.3.
        seed:     Optional RNG seed for reproducibility.

    Returns:
        np.ndarray of shape (n_steps,): intermediate prices from start to end.
    """
    if n_steps <= 1:
        return np.array([p_end])

    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_steps + 1)[1:]  # time steps: (0, 1]

    # Standard Brownian Bridge: B(t) = t·W(1) + (W(t) - t·W(1))
    # For our pinned bridge from 0 to (p_end - p_start):
    drift = p_end - p_start
    # Gaussian noise scaled by sqrt(t*(1-t)) — max variance at t=0.5
    noise = sigma * rng.standard_normal(n_steps) * np.sqrt(t * (1 - t) + 1e-9)
    bridge = p_start + drift * t + noise
    # Hard-pin the last value to guarantee exact Close price
    bridge[-1] = p_end
    return bridge


# =============================================================================
# RENKO BRICK BUILDER v3.0
# =============================================================================

class RenkoBrickBuilder:
    """
    Converts 1-minute OHLC data into Renko bricks using a Normalised ATR
    (NATR) brick size = 0.15% of price.

    Key logic
    ---------
    • Brick size recalculated per trading day using the previous day's close.
    • 9:15 AM Gap Filter:  if |open − prev_renko_level| > 2 × brick_size,
      teleport the Renko base (do NOT create intermediate "fake" bricks);
      mark the resulting brick as is_reset = True.
    • Volume is tracked cumulatively per brick for VWAP computation.
    • Brownian Bridge generates sub-ticks for large candles.
    • Wick overshoot is carried forward into the next brick.
    """

    # Minimum candle range multiplier to trigger Brownian Bridge generation
    BRIDGE_TRIGGER_MULTIPLIER = 2.0  # candle_range >= 2x brick_size → BB
    BRIDGE_STEPS = 10                 # Number of sub-tick points in bridge

    def __init__(self, natr_pct: float = config.NATR_BRICK_PERCENT):
        self.natr_pct = natr_pct
        
        # Phase 1: State Memory Logic
        self._prev_candle_close: float | None = None
        self._last_brick_timestamp = None
        self._current_candle_gap_pct: float = 0.0

    def transform(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a 1-min OHLC DataFrame into Renko bricks.

        Parameters
        ----------
        ohlc_df : DataFrame
            columns=[timestamp, open, high, low, close, volume]
            volume column is optional but required for VWAP features.

        Returns
        -------
        DataFrame of bricks with columns:
            [brick_timestamp, brick_open, brick_close, brick_high, brick_low,
             brick_size, direction, is_reset, duration_seconds,
             volume, cum_volume, typical_price]
        """
        if ohlc_df.empty:
            return pd.DataFrame()

        has_volume = "volume" in ohlc_df.columns

        ohlc = ohlc_df.copy()
        ohlc["date"] = ohlc["timestamp"].dt.date
        trading_days = sorted(ohlc["date"].unique())

        all_bricks: list[dict] = []
        renko_level: float | None = None
        brick_size: float | None = None
        brick_start_time = None

        # Wick tracking — carries overshoot between bricks (Fix 3)
        current_actual_high: float = 0.0
        current_actual_low: float = float("inf")
        current_brick_volume: float = 0.0  # accumulated volume per brick (Fix 1)

        for day_idx, day in enumerate(trading_days):
            day_data = ohlc[ohlc["date"] == day].sort_values("timestamp")
            if day_data.empty:
                continue

            # ── Calculate brick size from previous day's close ──────────────
            if day_idx == 0:
                brick_size = day_data.iloc[0]["close"] * self.natr_pct
            else:
                prev_day = trading_days[day_idx - 1]
                prev_close = ohlc[ohlc["date"] == prev_day]["close"].iloc[-1]
                brick_size = prev_close * self.natr_pct

            if brick_size <= 0:
                brick_size = 1.0  # safety floor

            is_first_tick_of_day = True

            for _, candle in day_data.iterrows():
                ts = candle["timestamp"]
                c_open  = candle["open"]
                c_high  = candle["high"]
                c_low   = candle["low"]
                c_close = candle["close"]
                c_vol   = float(candle["volume"]) if has_volume else 0.0
                c_range = c_high - c_low

                # Phase 1: Gap Math calculating gap at the top of 1-minute candle ingestion loop
                if self._prev_candle_close is not None and self._prev_candle_close > 0:
                    self._current_candle_gap_pct = ((c_open - self._prev_candle_close) / self._prev_candle_close) * 100.0
                else:
                    self._current_candle_gap_pct = 0.0

                # ── Initialize on very first candle ──────────────────────────
                if renko_level is None:
                    renko_level = c_open
                    brick_start_time = ts
                    current_actual_high = c_high
                    current_actual_low  = c_low
                    current_brick_volume = c_vol
                    is_first_tick_of_day = False
                    continue

                # ── 9:15 AM Gap Filter ────────────────────────────────────────
                if is_first_tick_of_day:
                    is_first_tick_of_day = False
                    gap = abs(c_open - renko_level)
                    if gap > config.GAP_FILTER_MULTIPLIER * brick_size:
                        direction = 1 if c_open > renko_level else -1
                        brick = self._make_brick(
                            ts=ts,
                            open_price=renko_level,
                            close_price=c_open,
                            high=max(renko_level, c_open),
                            low=min(renko_level, c_open),
                            brick_size=brick_size,
                            direction=direction,
                            is_reset=True,
                            start_time=brick_start_time or ts,
                            end_time=ts,
                            volume=current_brick_volume,
                            true_gap_pct=self._current_candle_gap_pct,
                            time_to_form_seconds=0.0,
                            volume_intensity_per_sec=0.0,
                            is_opening_drive=1 if time(9, 15) <= ts.time() <= time(10, 0) else 0
                        )
                        all_bricks.append(brick)
                        renko_level = c_open
                        brick_start_time = ts
                        # Reset tracking — gap brick carries no wick overshoot
                        current_actual_high = c_open
                        current_actual_low  = c_open
                        current_brick_volume = 0.0

                # ── Build the tick path for this candle ──────────────────────
                # Direction determines OHLC traversal order to reflect
                # how price moves within a 1-minute candle:
                #   Bullish:  O → L → H → C  (dip first, then rally)
                #   Bearish:  O → H → L → C  (fake spike, then dump)
                if c_close >= c_open:
                    waypoints = [c_open, c_low, c_high, c_close]
                else:
                    waypoints = [c_open, c_high, c_low, c_close]

                # ── Fix 2: Brownian Bridge sub-grid for large candles ─────────
                # When the candle range is large enough to theoretically cross
                # the brick size multiple times, simulate the intra-minute path
                # to generate additional bricks that the live engine would see.
                if c_range >= self.BRIDGE_TRIGGER_MULTIPLIER * brick_size:
                    tick_prices = self._expand_waypoints_with_bridge(
                        waypoints, c_range, brick_size
                    )
                else:
                    tick_prices = waypoints

                # ── Process the expanded tick path ────────────────────────────
                bricks_in_candle: list[dict] = []
                # Distribute candle volume proportionally (Fix 1)
                vol_per_segment = c_vol / max(len(tick_prices) - 1, 1)

                for i, price in enumerate(tick_prices):
                    # Accumulate volume for current brick
                    current_brick_volume += vol_per_segment

                    # Fix 3: Update actual price extremes seen since last brick close
                    current_actual_high = max(current_actual_high, price)
                    current_actual_low  = min(current_actual_low, price)

                    move = price - renko_level

                    while abs(move) >= brick_size:
                        direction = 1 if move > 0 else -1
                        new_level = renko_level + direction * brick_size

                        # Fix 3 (Patched for Gaps): True Wick — use carried-forward actual extremes
                        # CRITICAL: Do not apply the gap's terminal peak to intermediate bricks!
                        is_last_brick_in_gap = abs(price - new_level) < brick_size
                        
                        if is_last_brick_in_gap:
                            b_high = max(renko_level, new_level, current_actual_high)
                            b_low  = min(renko_level, new_level, current_actual_low)
                        else:
                            b_high = max(renko_level, new_level)
                            b_low  = min(renko_level, new_level)

                        brick = self._make_brick(
                            ts=ts,
                            open_price=renko_level,
                            close_price=new_level,
                            high=b_high,
                            low=b_low,
                            brick_size=brick_size,
                            direction=direction,
                            is_reset=False,
                            start_time=brick_start_time or ts,
                            end_time=ts,
                            volume=current_brick_volume,
                            true_gap_pct=self._current_candle_gap_pct,
                            time_to_form_seconds=min(3600.0, (ts - self._last_brick_timestamp).total_seconds()) if self._last_brick_timestamp else 0.0,
                            volume_intensity_per_sec=current_brick_volume / max(1.0, min(3600.0, (ts - self._last_brick_timestamp).total_seconds()) if self._last_brick_timestamp else 0.0),
                            is_opening_drive=1 if time(9, 15) <= ts.time() <= time(10, 0) else 0
                        )
                        self._last_brick_timestamp = ts
                        bricks_in_candle.append(brick)

                        renko_level = new_level
                        move = price - renko_level

                        # Fix 3: Carry forward — the overshoot (price - new_level)
                        # becomes the starting extreme for the next brick.
                        # Do NOT reset to renko_level; carry the actual price.
                        current_actual_high = max(new_level, price)
                        current_actual_low  = min(new_level, price)
                        current_brick_volume = 0.0  # brick consumed, reset accumulator

                        brick_start_time = ts
                        
                # Phase 1: Memory Reset
                self._prev_candle_close = c_close

                if bricks_in_candle:
                    # Distribute candle time evenly across all bricks formed
                    for b in bricks_in_candle:
                        b["bricks_in_this_candle"] = len(bricks_in_candle)
                    all_bricks.extend(bricks_in_candle)

        if not all_bricks:
            return pd.DataFrame()

        bricks_df = pd.DataFrame(all_bricks)

        # ── Duration calculation ─────────────────────────────────────────────
        raw_dur = (
            bricks_df["brick_end_time"] - bricks_df["brick_start_time"]
        ).dt.total_seconds()
        bricks_count = bricks_df.get(
            "bricks_in_this_candle", pd.Series(1, index=bricks_df.index)
        ).fillna(1)
        bricks_df["duration_seconds"] = (raw_dur / bricks_count).clip(lower=1, upper=300)

        if "bricks_in_this_candle" in bricks_df.columns:
            bricks_df.drop(columns=["bricks_in_this_candle"], inplace=True)

        # ── Volume-derived columns (Fix 1) ────────────────────────────────────
        # typical_price and cum_volume are required for VWAP Z-Score computation
        bricks_df["typical_price"] = (
            bricks_df["brick_high"] + bricks_df["brick_low"] + bricks_df["brick_close"]
        ) / 3.0
        bricks_df["cum_volume"] = bricks_df["volume"].cumsum()

        return bricks_df

    def _expand_waypoints_with_bridge(
        self,
        waypoints: list,
        c_range: float,
        brick_size: float,
    ) -> list:
        """
        Expand the 4 OHLC waypoints into a dense sub-tick path using
        Brownian Bridges between each consecutive waypoint pair.

        The number of bridge steps scales with the candle range / brick_size
        ratio, ensuring larger candles get proportionally more sub-ticks.
        """
        # Scale bridge density with range magnitude
        scale = max(2, int(c_range / brick_size))
        n_steps = min(scale * 3, 30)  # cap at 30 sub-ticks per segment
        sigma = c_range * 0.3         # volatility = 30% of the candle range

        expanded = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            segment = _brownian_bridge(
                p_start=waypoints[i],
                p_end=waypoints[i + 1],
                n_steps=n_steps,
                sigma=sigma,
            )
            expanded.extend(segment.tolist())
        return expanded

    @staticmethod
    def _make_brick(
        ts, open_price, close_price, high, low,
        brick_size, direction, is_reset, start_time, end_time,
        volume: float = 0.0, **kwargs
    ) -> dict:
        return {
            "brick_timestamp":  ts,
            "brick_start_time": start_time,
            "brick_end_time":   end_time,
            "brick_open":       open_price,
            "brick_close":      close_price,
            "brick_high":       high,
            "brick_low":        low,
            "brick_size":       brick_size,
            "direction":        direction,
            "is_reset":         is_reset,
            "volume":           volume,
            "true_gap_pct":     kwargs.get("true_gap_pct", 0.0),
            "time_to_form_seconds": kwargs.get("time_to_form_seconds", 0.0),
            "volume_intensity_per_sec": kwargs.get("volume_intensity_per_sec", 0.0),
            "is_opening_drive": kwargs.get("is_opening_drive", 0)
        }


# =============================================================================
# PATH-CONFLICT RESOLUTION (Fix 4)
# =============================================================================

def check_path_conflict(
    tick_path: np.ndarray,
    entry_price: float,
    target_price: float,
    stop_price: float,
) -> str:
    """
    Pessimistic Path-Conflict Resolution.

    Scans an interpolated tick path (from Brownian Bridge or OHLC waypoints)
    to determine the trade outcome. If BOTH target AND stop are touched
    within the same path, the outcome defaults to LOSS (conservative).

    This replicates the harsh reality of HFT market microstructure where
    a long wick fills a stop before the ultimate target is reached.

    Args:
        tick_path:     Array of prices from the candle's interpolated path.
        entry_price:   Trade entry price.
        target_price:  Profit target price (e.g., entry + 5 x brick_size for LONG).
        stop_price:    Stop loss price    (e.g., entry - 2 x brick_size for LONG).

    Returns:
        "WIN"   — if target hit before stop, and no conflict.
        "LOSS"  — if stop hit first, OR if both hit in the same path (conflict).
        "OPEN"  — if neither barrier was hit in this candle.
    """
    hit_target = False
    hit_stop   = False
    result     = "OPEN"

    for price in tick_path:
        if not hit_target and not hit_stop:
            # First hit determines primary outcome
            if price >= target_price:
                hit_target = True
                result = "WIN"
            elif price <= stop_price:
                hit_stop = True
                result = "LOSS"
        elif hit_target and price <= stop_price:
            # Target was hit first, but later in same path stop also hit
            # → Path Conflict: record LOSS (pessimistic)
            result = "LOSS"
            break
        elif hit_stop and price >= target_price:
            # Stop was hit, path conflict irrelevant (already LOSS)
            break

    return result


# =============================================================================
# LIVE RENKO STATE (real-time tick-by-tick)
# =============================================================================

class LiveRenkoState:
    """
    Maintains the incremental Renko state for a single symbol during
    the live trading session. Processes one tick at a time.
    Includes Fix 1 (volume tracking) and Fix 3 (wick carry-forward).
    """

    def __init__(self, symbol: str, sector: str, brick_size: float):
        self.symbol     = symbol
        self.sector     = sector
        self.brick_size = brick_size
        self.renko_level: float | None = None
        self.brick_start_time          = None
        self.bricks: list[dict]        = []
        self.is_first_tick             = True

        # Fix 3: Wick carry-forward state
        self._actual_high: float = 0.0
        self._actual_low:  float = float("inf")
        # Fix 1: Volume tracking
        self._brick_volume: float = 0.0

        # Phase 1: State Memory
        self._prev_candle_close: float | None = None
        self._last_brick_timestamp = None
        self._current_candle_gap_pct: float = 0.0
        self._last_tick_minute = None

    def load_history(self, limit: int = 100):
        """Pre-load historical bricks to prevent cold-start math errors."""
        import pandas as pd
        stock_dir = config.DATA_DIR / self.sector / self.symbol
        if not stock_dir.exists():
            return
        pqs = sorted(stock_dir.glob("*.parquet"))
        if not pqs:
            return

        try:
            df = pd.read_parquet(pqs[-1])
            if df.empty:
                return

            if df["brick_timestamp"].dt.tz is not None:
                df["brick_timestamp"] = df["brick_timestamp"].dt.tz_localize(None)

            df = df.tail(limit).copy()
            hist_bricks = df.to_dict("records")
            self.bricks.extend(hist_bricks)

            last_brick = self.bricks[-1]
            self.renko_level      = last_brick["brick_close"]
            self.brick_start_time = last_brick["brick_timestamp"]
            # Seed wick tracking from last known close
            self._actual_high = self.renko_level
            self._actual_low  = self.renko_level

        except Exception as e:
            print(f"Warning: Failed to load history for {self.symbol}: {e}")

    def process_tick(
        self, price: float, high: float, low: float, timestamp: datetime
    ) -> list[dict]:
        """Process a single price tick — generate bricks if needed. Returns new bricks formed."""
        new_bricks: list[dict] = []

        if self.renko_level is None:
            self.renko_level      = price
            self.brick_start_time = timestamp
            self._actual_high     = high
            self._actual_low      = low
            self._prev_candle_close = price
            self._last_tick_minute = timestamp.minute
            return new_bricks

        # Approximation of 1-minute candle ingestion gap math for live ticks
        if self._last_tick_minute is not None and self._last_tick_minute != timestamp.minute:
            if self._prev_candle_close is not None and self._prev_candle_close > 0:
                self._current_candle_gap_pct = ((price - self._prev_candle_close) / self._prev_candle_close) * 100.0
            self._last_tick_minute = timestamp.minute
            self._prev_candle_close = price # Simple tick fallback, close of last min

        # Fix 3: Update carried wick extremes with this tick's data
        self._actual_high = max(self._actual_high, high)
        self._actual_low  = min(self._actual_low,  low)

        # 9:15 Gap Filter on first tick of day
        if self.is_first_tick:
            self.is_first_tick = False
            gap = abs(price - self.renko_level)
            if gap > config.GAP_FILTER_MULTIPLIER * self.brick_size:
                direction = 1 if price > self.renko_level else -1
                brick = {
                    "brick_timestamp":  timestamp,
                    "brick_start_time": self.brick_start_time or timestamp,
                    "brick_end_time":   timestamp,
                    "brick_open":       self.renko_level,
                    "brick_close":      price,
                    "brick_high":       max(self.renko_level, high),
                    "brick_low":        min(self.renko_level, low),
                    "brick_size":       self.brick_size,
                    "direction":        direction,
                    "is_reset":         True,
                    "duration_seconds": min(300.0, max(1.0, (
                        timestamp - (self.brick_start_time or timestamp)
                    ).total_seconds())),
                    "volume": self._brick_volume,
                    "typical_price": (max(self.renko_level, high) + min(self.renko_level, low) + price) / 3.0,
                    "cum_volume": 0.0,  # filled by features.py at batch time
                    "true_gap_pct": self._current_candle_gap_pct,
                    "time_to_form_seconds": min(3600.0, (timestamp - self._last_brick_timestamp).total_seconds()) if self._last_brick_timestamp else 0.0,
                    "volume_intensity_per_sec": 0.0,
                    "is_opening_drive": 1 if time(9, 15) <= timestamp.time() <= time(10, 0) else 0
                }
                self.bricks.append(brick)
                new_bricks.append(brick)
                self.renko_level      = price
                self.brick_start_time = timestamp
                self._last_brick_timestamp = timestamp
                self._actual_high     = price
                self._actual_low      = price
                self._brick_volume    = 0.0

        move = price - self.renko_level
        if abs(move) >= self.brick_size:
            bricks_to_generate = int(abs(move) // self.brick_size)
            total_dur = max(1.0, (
                timestamp - (self.brick_start_time or timestamp)
            ).total_seconds())
            dur_per_brick = min(300.0, max(1.0, total_dur / bricks_to_generate))

            while abs(move) >= self.brick_size:
                direction = 1 if move > 0 else -1
                new_level = self.renko_level + direction * self.brick_size

                # Fix 3 (Patched for Gaps): True wick using carried extremes.
                # CRITICAL: Intermediate "gap" bricks crossed instantaneously do not have wicks.
                # If we apply the terminal gap high to the first intermediate brick, it creates 
                # a massive fake rejection wick that trips the Wick Trap gate and blocks breakouts.
                is_last_brick_in_gap = abs(price - new_level) < self.brick_size
                
                if is_last_brick_in_gap:
                    b_high = max(self.renko_level, new_level, self._actual_high)
                    b_low  = min(self.renko_level, new_level, self._actual_low)
                else:
                    b_high = max(self.renko_level, new_level)
                    b_low  = min(self.renko_level, new_level)

                typical = (b_high + b_low + new_level) / 3.0

                brick = {
                    "brick_timestamp":  timestamp,
                    "brick_start_time": self.brick_start_time or timestamp,
                    "brick_end_time":   timestamp,
                    "brick_open":       self.renko_level,
                    "brick_close":      new_level,
                    "brick_high":       b_high,
                    "brick_low":        b_low,
                    "brick_size":       self.brick_size,
                    "direction":        direction,
                    "is_reset":         False,
                    "duration_seconds": dur_per_brick,
                    "volume":           self._brick_volume,
                    "typical_price":    typical,
                    "cum_volume":       0.0,
                    "true_gap_pct":     self._current_candle_gap_pct,
                    "time_to_form_seconds": min(3600.0, (timestamp - self._last_brick_timestamp).total_seconds()) if self._last_brick_timestamp else 0.0,
                    "volume_intensity_per_sec": self._brick_volume / max(1.0, min(3600.0, (timestamp - self._last_brick_timestamp).total_seconds()) if self._last_brick_timestamp else 0.0),
                    "is_opening_drive": 1 if time(9, 15) <= timestamp.time() <= time(10, 0) else 0
                }
                self.bricks.append(brick)
                new_bricks.append(brick)
                
                self._last_brick_timestamp = timestamp

                self.renko_level = new_level
                move = price - self.renko_level

                # Carry-forward overshoot into next brick formation
                self._actual_high  = max(new_level, price)
                self._actual_low   = min(new_level, price)
                self._brick_volume = 0.0

            self.brick_start_time = timestamp

        # Memory leak prevention
        if len(self.bricks) > 1000:
            self.bricks = self.bricks[-500:]

        return new_bricks

    def to_dataframe(self) -> pd.DataFrame:
        if not self.bricks:
            return pd.DataFrame()
        return pd.DataFrame(self.bricks)
