"""
src/core/renko.py — Renko Brick Builder
=========================================
Converts 1-minute OHLC data into Renko bricks using NATR brick size
(0.15% of price).  Implements the 9:15 AM Gap Filter.

Used by:
  • src/data/batch_factory.py  (historical bulk transform)
  • src/live/engine.py         (real-time incremental bricks)
"""

import numpy as np
import pandas as pd
from datetime import datetime

import config


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
    """

    def __init__(self, natr_pct: float = config.NATR_BRICK_PERCENT):
        self.natr_pct = natr_pct

    def transform(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a 1-min OHLC DataFrame into Renko bricks.

        Parameters
        ----------
        ohlc_df : DataFrame  columns=[timestamp, open, high, low, close, volume]

        Returns
        -------
        DataFrame of bricks with columns:
            [brick_timestamp, brick_open, brick_close, brick_high, brick_low,
             brick_size, direction, is_reset, duration_seconds]
        """
        if ohlc_df.empty:
            return pd.DataFrame()

        ohlc = ohlc_df.copy()
        ohlc["date"] = ohlc["timestamp"].dt.date
        trading_days = sorted(ohlc["date"].unique())

        all_bricks: list[dict] = []
        renko_level: float | None = None
        brick_size: float | None = None
        brick_start_time = None

        for day_idx, day in enumerate(trading_days):
            day_data = ohlc[ohlc["date"] == day].sort_values("timestamp")
            if day_data.empty:
                continue

            # ── Calculate brick size from previous day's close ──────────
            if day_idx == 0:
                brick_size = day_data.iloc[0]["close"] * self.natr_pct
            else:
                prev_day = trading_days[day_idx - 1]
                prev_close = ohlc[ohlc["date"] == prev_day]["close"].iloc[-1]
                brick_size = prev_close * self.natr_pct

            if brick_size <= 0:
                brick_size = 1.0  # safety floor

            # ── 9:15 AM Gap Filter ──────────────────────────────────────
            is_first_tick_of_day = True

            for _, candle in day_data.iterrows():
                price = candle["close"]
                ts = candle["timestamp"]

                if renko_level is None:
                    renko_level = price
                    brick_start_time = ts
                    continue

                # Gap filter on the day's first candle (market open)
                if is_first_tick_of_day:
                    is_first_tick_of_day = False
                    gap = abs(candle["open"] - renko_level)
                    if gap > config.GAP_FILTER_MULTIPLIER * brick_size:
                        direction = 1 if candle["open"] > renko_level else -1
                        brick = self._make_brick(
                            ts=ts,
                            open_price=renko_level,
                            close_price=candle["open"],
                            high=max(renko_level, candle["open"]),
                            low=min(renko_level, candle["open"]),
                            brick_size=brick_size,
                            direction=direction,
                            is_reset=True,
                            start_time=brick_start_time or ts,
                            end_time=ts,
                        )
                        all_bricks.append(brick)
                        renko_level = candle["open"]
                        brick_start_time = ts

                # ── Normal Renko logic ──────────────────────────────────
                move = price - renko_level
                bricks_in_candle = []
                while abs(move) >= brick_size:
                    direction = 1 if move > 0 else -1
                    new_level = renko_level + direction * brick_size

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
                    )
                    bricks_in_candle.append(brick)
                    renko_level = new_level
                    brick_start_time = ts
                    move = price - renko_level
                
                if bricks_in_candle:
                    # Wick Pressure Match: The live engine only sees ticks sequentially 
                    # and sets brick_high/low strictly to mathematical body limits immediately.
                    # We remove the 1-min candle historical extreme to prevent train-serve mismatch
                    # and eliminate micro-lookahead biases within the minute.
                    
                    # Fix Duration Leak: If multiple bricks formed in a single candle,
                    # they all instantly formed at timestamp `ts`. This causes duration=0.
                    # In live trading, time is continuous. We approximate this by
                    # dividing the total time of the candle evenly among the generated bricks.
                    # We will override the duration_seconds later, but we need a special marker.
                    for b in bricks_in_candle:
                        b["bricks_in_this_candle"] = len(bricks_in_candle)
                    
                    all_bricks.extend(bricks_in_candle)

        if not all_bricks:
            return pd.DataFrame()

        bricks_df = pd.DataFrame(all_bricks)
        
        # Calculate raw duration
        raw_dur = (bricks_df["brick_end_time"] - bricks_df["brick_start_time"]).dt.total_seconds()
        
        # Determine average duration per brick for intra-candle synthetic bricks
        bricks_count = bricks_df.get("bricks_in_this_candle", pd.Series(1, index=bricks_df.index))
        bricks_count = bricks_count.fillna(1)
        
        # Distribute the time evenly
        bricks_df["duration_seconds"] = (raw_dur / bricks_count).clip(lower=1)
        
        # Drop the temporary column
        if "bricks_in_this_candle" in bricks_df.columns:
            bricks_df.drop(columns=["bricks_in_this_candle"], inplace=True)
            
        return bricks_df

    @staticmethod
    def _make_brick(
        ts, open_price, close_price, high, low,
        brick_size, direction, is_reset, start_time, end_time,
    ) -> dict:
        return {
            "brick_timestamp": ts,
            "brick_start_time": start_time,
            "brick_end_time": end_time,
            "brick_open": open_price,
            "brick_close": close_price,
            "brick_high": high,
            "brick_low": low,
            "brick_size": brick_size,
            "direction": direction,
            "is_reset": is_reset,
        }


class LiveRenkoState:
    """
    Maintains the incremental Renko state for a single symbol during
    the live trading session.  Processes one tick at a time.
    """

    def __init__(self, symbol: str, sector: str, brick_size: float):
        self.symbol = symbol
        self.sector = sector
        self.brick_size = brick_size
        self.renko_level: float | None = None
        self.brick_start_time = None
        self.bricks: list[dict] = []
        self.is_first_tick = True

    def load_history(self, limit: int = 100):
        """Pre-load historical bricks to prevent cold-start math errors."""
        import pandas as pd
        import config
        stock_dir = config.DATA_DIR / self.sector / self.symbol
        if not stock_dir.exists():
            return
        pqs = sorted(stock_dir.glob("*.parquet"))
        if not pqs:
            return
        
        try:
            # Only need to load the most recent file for 100 bricks
            df = pd.read_parquet(pqs[-1])
            if df.empty:
                return
            
            # Normalize timestamp to naive local time to precisely match live datetime.now() ticks
            if df["brick_timestamp"].dt.tz is not None:
                df["brick_timestamp"] = df["brick_timestamp"].dt.tz_localize(None)
            
            # Keep only the last N bricks
            df = df.tail(limit).copy()
            
            # Convert to dict records matching live structure
            hist_bricks = df.to_dict("records")
            self.bricks.extend(hist_bricks)
            
            # Set the baseline for the next live tick based on the last historical brick
            last_brick = self.bricks[-1]
            self.renko_level = last_brick["brick_close"]
            self.brick_start_time = last_brick["brick_timestamp"]
            
        except Exception as e:
            # Don't let a read error stop the engine, but log it if possible.
            print(f"Warning: Failed to load history for {self.symbol}: {e}")

    def process_tick(self, price: float, high: float, low: float, timestamp: datetime):
        """Process a single price tick — generate bricks if needed."""
        if self.renko_level is None:
            self.renko_level = price
            self.brick_start_time = timestamp
            return

        # 9:15 Gap Filter on first tick of day
        if self.is_first_tick:
            self.is_first_tick = False
            gap = abs(price - self.renko_level)
            if gap > config.GAP_FILTER_MULTIPLIER * self.brick_size:
                direction = 1 if price > self.renko_level else -1
                self.bricks.append({
                    "brick_timestamp": timestamp,
                    "brick_start_time": self.brick_start_time or timestamp,
                    "brick_end_time": timestamp,
                    "brick_open": self.renko_level,
                    "brick_close": price,
                    "brick_high": max(self.renko_level, high),
                    "brick_low": min(self.renko_level, low),
                    "brick_size": self.brick_size,
                    "direction": direction,
                    "is_reset": True,
                    "duration_seconds": max(1, (timestamp - (self.brick_start_time or timestamp)).total_seconds()),
                })
                self.renko_level = price
                self.brick_start_time = timestamp

        move = price - self.renko_level
        if abs(move) >= self.brick_size:
            bricks_to_generate = int(abs(move) // self.brick_size)
            total_dur = max(1.0, (timestamp - (self.brick_start_time or timestamp)).total_seconds())
            dur_per_brick = max(1.0, total_dur / bricks_to_generate)

            while abs(move) >= self.brick_size:
                direction = 1 if move > 0 else -1
                new_level = self.renko_level + direction * self.brick_size

                self.bricks.append({
                    "brick_timestamp": timestamp,
                    "brick_start_time": self.brick_start_time or timestamp,
                    "brick_end_time": timestamp,
                    "brick_open": self.renko_level,
                    "brick_close": new_level,
                    "brick_high": max(self.renko_level, new_level),
                    "brick_low": min(self.renko_level, new_level),
                    "brick_size": self.brick_size,
                    "direction": direction,
                    "is_reset": False,
                    "duration_seconds": dur_per_brick,
                })
                self.renko_level = new_level
                move = price - self.renko_level

            self.brick_start_time = timestamp

    def to_dataframe(self) -> pd.DataFrame:
        if not self.bricks:
            return pd.DataFrame()
        return pd.DataFrame(self.bricks)
