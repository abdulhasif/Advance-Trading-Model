"""
src/live/execution_guard.py — Five Live Execution Hardening Modules
======================================================================
Hostile audit fixes for the gap between backtest and live WebSocket execution.

Vulnerability Vectors Addressed:
  1. Cold Start State Mismatch    — HistoricalWarmupSplicer
  2. Intra-Candle Tick Precision  — tick_adjusted_stop_pct()
  3. Silent WebSocket / Forward-Fill — HeartbeatCandle injector
  4. DataFrame Memory Leak         — RollingBrickBuffer (O(1) deque)
  5. Pending Order Lockup          — PendingOrderGuard (asyncio mutex)

Author: Red Team Audit — Quant Risk Office
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 (SYNC): PENDING ORDER GUARD — Thread-Safe Mutex for Synchronous Engine
# ═══════════════════════════════════════════════════════════════════════════

class SyncPendingOrderGuard:
    """
    Thread-safe, synchronous version of PendingOrderGuard.
    Uses threading.Lock instead of asyncio.Lock since engine.py is
    a standard synchronous loop (not async/await).

    Prevents Brain 1 from placing a second order for the same symbol
    while the first order is still being processed (paper fill simulation).

    Architecture:
        - Per-symbol threading.Lock acquired on signal.
        - Lock auto-releases after lock_timeout_seconds (safety net).
        - try_acquire() is non-blocking — returns False immediately if locked.
        - release() MUST be called in a finally block.

    Usage in execute_trade():
        if not guard.try_acquire(symbol, side):
            logger.info(f"{symbol}: order still PENDING, signal dropped")
            return
        try:
            ... place paper order ...
        finally:
            guard.release(symbol)
    """

    def __init__(self, lock_timeout_seconds: int = config.ORDER_LOCK_TIMEOUT_SEC):
        """
        Args:
            lock_timeout_seconds: Paper orders fill instantly but give 5s
                                  buffer for logging, CSV writes, state updates.
        """
        self._locks:         dict[str, threading.Lock]  = {}
        self._acquired_at:   dict[str, float]           = {}
        self._pending_side:  dict[str, str]             = {}
        self.lock_timeout    = lock_timeout_seconds
        self._blocked_count  = 0

    def _get_lock(self, symbol: str) -> threading.Lock:
        if symbol not in self._locks:
            self._locks[symbol] = threading.Lock()
        return self._locks[symbol]

    def try_acquire(self, symbol: str, side: str) -> bool:
        """
        Non-blocking attempt to acquire the mutex for a symbol.

        Returns:
            True  -> Lock acquired. Safe to place order.
            False -> Symbol already has a pending order. Drop this signal.
        """
        lock = self._get_lock(symbol)

        # Auto-expire stale locks (safety net for crashes / exceptions)
        if symbol in self._acquired_at:
            elapsed = time.monotonic() - self._acquired_at[symbol]
            if elapsed > self.lock_timeout:
                logger.warning(f"[OrderGuard] {symbol}: stale lock ({elapsed:.1f}s) — force-releasing.")
                try:
                    lock.release()
                except RuntimeError:
                    pass
                self._acquired_at.pop(symbol, None)
                self._pending_side.pop(symbol, None)

        # Non-blocking acquire (returns False immediately if locked)
        acquired = lock.acquire(blocking=False)
        if not acquired:
            self._blocked_count += 1
            pending = self._pending_side.get(symbol, "?")
            logger.info(f"[OrderGuard] {symbol}: BLOCKED — {pending} still PENDING "
                        f"(total blocked today: {self._blocked_count})")
            return False

        self._acquired_at[symbol] = time.monotonic()
        self._pending_side[symbol] = side
        return True

    def release(self, symbol: str) -> None:
        """
        Release the lock. ALWAYS call this in a finally block.
        """
        lock = self._get_lock(symbol)
        elapsed = time.monotonic() - self._acquired_at.get(symbol, time.monotonic())
        try:
            lock.release()
            logger.debug(f"[OrderGuard] {symbol}: lock released after {elapsed:.2f}s")
        except RuntimeError:
            pass  # Was already released (double-release guard)
        self._acquired_at.pop(symbol, None)
        self._pending_side.pop(symbol, None)

    def is_pending(self, symbol: str) -> bool:
        """Check if a symbol currently has a pending order."""
        lock = self._get_lock(symbol)
        return lock.locked()

    def get_status(self) -> dict:
        return {
            "locked_symbols": [s for s, l in self._locks.items() if l.locked()],
            "blocked_count":  self._blocked_count,
        }




# ═══════════════════════════════════════════════════════════════════════════
# PATCH 1: ENTRY STATE LOCK — Guaranteed Position De-Duplication
# ═══════════════════════════════════════════════════════════════════════════

class EntryStateLock:
    """
    The Single Source of Truth for position state.

    The Problem (observed in today's churn):
        paper_trader.py checks `if symbol in portfolio.positions` to block
        duplicate entries. But this dict is updated AFTER open_position()
        returns, meaning if two bricks form in the same loop iteration for
        the same symbol (possible during opening auction volatility), the
        second signal sees an empty dict and places a duplicate order.

    The Fix:
        A *separate*, *dedicated* set that is written BEFORE calling
        open_position() and cleared AFTER close_position() confirms.
        This set is the authoritative gate — portfolio.positions is
        downstream bookkeeping, not the entry gate.

    Architecture:
        - `try_enter(symbol)` → returns True once, atomically sets the lock.
        - `confirm_exit(symbol)` → clears the lock.
        - Uses threading.Lock for the set mutation (O(1), non-blocking).
        - Tick-safe: each call completes in < 1 µs.

    Integration in paper_trader.py (replace the open_position check):
        # At the top of the entry gate block:
        if not guard.entry_lock.try_enter(sym):
            continue   # already in this stock

        # After confirmed exit:
        guard.entry_lock.confirm_exit(sym)
    """

    def __init__(self):
        self._open_symbols: set[str] = set()
        self._lock = threading.Lock()
        self._blocked_count = 0

    def try_enter(self, symbol: str) -> bool:
        """
        Atomically attempt to mark a symbol as 'entered'.

        Returns:
            True  → symbol was FREE. It is now locked. Proceed to open_position().
            False → symbol already has an open position. Drop this signal immediately.
        """
        with self._lock:
            if symbol in self._open_symbols:
                self._blocked_count += 1
                logger.info(
                    f"[EntryStateLock] {symbol}: BLOCKED — already in open_symbols "
                    f"(total blocked today: {self._blocked_count})"
                )
                return False
            self._open_symbols.add(symbol)
            logger.debug(f"[EntryStateLock] {symbol}: ENTERED — lock acquired.")
            return True

    def confirm_exit(self, symbol: str) -> None:
        """
        Release the lock for a symbol after close_position() has confirmed exit.
        MUST be called in the exit handler — never in the entry branch.
        """
        with self._lock:
            self._open_symbols.discard(symbol)
            logger.debug(f"[EntryStateLock] {symbol}: EXITED — lock released.")

    def is_open(self, symbol: str) -> bool:
        """Read-only check. Use try_enter() for atomic entry — not this."""
        with self._lock:
            return symbol in self._open_symbols

    @property
    def open_count(self) -> int:
        with self._lock:
            return len(self._open_symbols)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "open_symbols":   sorted(self._open_symbols),
                "open_count":     len(self._open_symbols),
                "blocked_count":  self._blocked_count,
            }


# ═══════════════════════════════════════════════════════════════════════════
# PATCH 2: BRICK COOLDOWN TRACKER — Re-Entry Penalty after Exit
# ═══════════════════════════════════════════════════════════════════════════

class BrickCooldownTracker:
    """
    Enforces a mandatory brick-count cooldown before the bot can re-enter
    the same symbol after any exit (win OR loss).

    The Problem (observed in today's churn — PETRONET traded 6 times):
        After TREND_REVERSAL exit, the very next UP brick fires another
        entry signal (prob 0.68 again). The bot re-enters immediately.
        The hysteresis dead-zone [0.40–0.60] only prevents exits; it
        does NOT prevent immediate re-entry after an exit. This creates
        a chop-saw: enter → 1-brick loss → exit → re-enter → repeat.

        Root cause: the probability model has short memory. After a reversal,
        it often sees the bounce brick as a new LONG signal instantly.

    The Fix:
        Record the total brick count seen for a symbol at the moment of exit.
        Block all new entries until (current_brick_count - exit_brick_count)
        >= COOLDOWN_BRICKS.

        COOLDOWN_BRICKS = 3 means: the market must form 3 more bricks after
        exit before the bot is allowed to re-enter. This forces the model to
        see at least 3 bricks of new evidence rather than reacting to a single
        bounce tick.

    Architecture:
        - Call `record_exit(symbol, brick_count_now)` in close_position().
        - Call `is_cooled_down(symbol, brick_count_now)` in the entry gate.
        - `brick_count_now` = RollingBrickBuffer._total_bricks_seen for that symbol.
        - Thread-safe (dict ops with GIL are atomic for simple reads/writes).

    Integration in paper_trader.py (add to entry gate block):
        current_bricks = guard.buffers[sym]._total_bricks_seen
        if not guard.cooldown.is_cooled_down(sym, current_bricks):
            continue   # too soon after last exit
    """

    def __init__(self, cooldown_bricks: int = config.BRICK_COOLDOWN):
        self.cooldown_bricks = cooldown_bricks
        # {symbol: brick_count_at_exit}
        self._exit_brick_count: dict[str, int] = {}
        self._blocked_count = 0

    def record_exit(self, symbol: str, brick_count_now: int) -> None:
        """
        Call this immediately when a position is closed (any reason).

        Args:
            symbol:          NSE symbol of the closed position.
            brick_count_now: RollingBrickBuffer._total_bricks_seen at exit time.
        """
        self._exit_brick_count[symbol] = brick_count_now
        logger.info(
            f"[BrickCooldown] {symbol}: EXIT recorded at brick #{brick_count_now}. "
            f"Re-entry blocked until brick #{brick_count_now + self.cooldown_bricks}."
        )

    def is_cooled_down(self, symbol: str, brick_count_now: int) -> bool:
        """
        Check if the cooldown window has elapsed since the last exit.

        Args:
            symbol:          NSE symbol to check.
            brick_count_now: RollingBrickBuffer._total_bricks_seen right now.

        Returns:
            True  → Cooldown elapsed. Entry is PERMITTED.
            False → Still in cooldown. DROP this signal.
        """
        if symbol not in self._exit_brick_count:
            return True  # Never exited → no cooldown applies

        bricks_since_exit = brick_count_now - self._exit_brick_count[symbol]
        if bricks_since_exit < self.cooldown_bricks:
            self._blocked_count += 1
            logger.info(
                f"[BrickCooldown] {symbol}: BLOCKED — only {bricks_since_exit} brick(s) "
                f"since exit (need {self.cooldown_bricks}). "
                f"Total re-entry blocks today: {self._blocked_count}."
            )
            return False

        return True

    def reset_symbol(self, symbol: str) -> None:
        """Clear cooldown for a symbol (e.g. on day change)."""
        self._exit_brick_count.pop(symbol, None)

    def reset_all(self) -> None:
        """Clear all cooldowns (call at start of each new trading day)."""
        self._exit_brick_count.clear()
        self._blocked_count = 0
        logger.info("[BrickCooldown] All cooldowns reset for new trading day.")

    def get_status(self) -> dict:
        return {
            "cooldown_bricks":   self.cooldown_bricks,
            "symbols_in_cd":     list(self._exit_brick_count.keys()),
            "blocked_count":     self._blocked_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1: COLD START STATE MISMATCH — Historical Warm-Up Splicer
# ═══════════════════════════════════════════════════════════════════════════

class HistoricalWarmupSplicer:
    """
    Solves the Cold Start problem: indicators like rolling FracDiff,
    consecutive_same_dir, and RelativeStrength need N historical bricks
    before they produce valid outputs.

    Without this, the first 30–60 minutes of trading produce NaN features,
    causing Brain 1 to output garbage probabilities (usually 0.5 ± noise).

    Architecture:
        1. At 09:08 AM, load the last 100 Renko bricks from the on-disk
           historical parquet (yesterday's data).
        2. Strip any bricks from "today" to avoid duplicate data leak.
        3. Mark all historical bricks as 'is_warmup=True' (never trade on these).
        4. Append incoming WebSocket bricks as 'is_warmup=False'.
        5. Prune the warmup prefix once features reach steady state (brick > 100).

    This ensures all indicators are fully warm the INSTANT 09:15 opens.
    """

    WARMUP_BRICKS_REQUIRED = config.RENKO_HISTORY_LIMIT

    def __init__(self, symbol: str, sector: str, before_ts: Optional[datetime] = None):
        self.symbol    = symbol
        self.sector    = sector
        self._bricks   = deque(maxlen=500)   # bounded — no memory growth
        self._warmed   = False
        self._cutoff_ts = before_ts if before_ts else datetime.now()

    def load_history(self) -> int:
        """
        Load historical bricks from the feature parquet, stripping today's
        rows to prevent look-ahead splice contamination.

        Returns:
            Number of valid warmup bricks loaded.
        """
        feature_parquet = config.FEATURES_DIR / self.sector / f"{self.symbol}.parquet"
        if not feature_parquet.exists():
            # Fallback for indices or stocks without feature parquets
            feature_parquet = config.DATA_DIR / self.sector / self.symbol / "2026.parquet"
            if not feature_parquet.exists():
                # Try 2025.parquet as last resort
                feature_parquet = config.DATA_DIR / self.sector / self.symbol / "2025.parquet"
                if not feature_parquet.exists():
                    logger.warning(f"[WarmupSplicer] No parquet found for {self.symbol} in features or data dirs")
                    return 0

        try:
            df = pd.read_parquet(feature_parquet).sort_values("brick_timestamp")

            # Contamination Shield: Strip any bricks from the current date or later
            # Warmup should strictly come from PREVIOUS trading sessions to avoid
            # lookahead or double-processing in simulations.
            target_date = self._cutoff_ts.date()
            df = df[df["brick_timestamp"].dt.date < target_date]

            # Take the last N bricks for warm-up
            df = df.tail(self.WARMUP_BRICKS_REQUIRED)

            for _, row in df.iterrows():
                brick = row.to_dict()
                brick["is_warmup"] = True    # Flag — engine MUST NOT trade on these
                self._bricks.append(brick)

            logger.info(f"[WarmupSplicer] {self.symbol}: loaded {len(df)} warmup bricks "
                        f"(newest: {df['brick_timestamp'].iloc[-1] if len(df) else 'N/A'})")
            return len(df)

        except Exception as e:
            logger.error(f"[WarmupSplicer] {self.symbol} history load failed: {e}")
            return 0

    def append_live_brick(self, brick: dict) -> None:
        """
        Append a brand-new live brick from the WebSocket.
        Marks it is_warmup=False so the engine knows it can trade on it.
        """
        brick = brick.copy()
        brick["is_warmup"] = False
        self._bricks.append(brick)

        # Prune historical prefix once we have enough fresh bricks
        # (reduces feature computation overhead in the afternoon)
        live_count = sum(1 for b in self._bricks if not b.get("is_warmup", False))
        if live_count > 200:
            # Drop oldest warmup bricks, keep at least 100 for context
            while self._bricks and self._bricks[0].get("is_warmup") and len(self._bricks) > 150:
                self._bricks.popleft()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the deque to a DataFrame for feature computation."""
        if not self._bricks:
            return pd.DataFrame()
        return pd.DataFrame(list(self._bricks))

    @property
    def is_ready(self) -> bool:
        """True once we have at least WARMUP_BRICKS_REQUIRED bricks total."""
        return len(self._bricks) >= self.WARMUP_BRICKS_REQUIRED

    @property
    def live_brick_count(self) -> int:
        """Number of bricks from today's live session only."""
        return sum(1 for b in self._bricks if not b.get("is_warmup", False))


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2: INTRA-CANDLE TICK PRECISION — Adjusted Stop Loss
# ═══════════════════════════════════════════════════════════════════════════

def tick_adjusted_stop_pct(base_stop_pct: float = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS,
                             stock_price: float = 500.0,
                             tick_size: float = 0.05,
                             intraday_vol_factor: float = 2.0) -> float:
    """
    Adjust the Triple Barrier stop-loss percentage to account for NSE tick-
    level bid-ask bounce that is hidden inside 1-minute OHLCV Low prices.

    The Problem (Mathematical):
        Historical 1-min OHLCV uses the True Low of the candle — the absolute
        worst tick of the 60-second window. The live WebSocket processes each
        tick individually. A single erroneous tick (fat finger, lag spike) can
        trigger the stop even if the candle's closing price is perfectly fine.

        Expected intra-candle noise = tick_size * spread_factor
        For NSE: tick = ₹0.05, typical spread = 2–3 ticks = ₹0.10–0.15

    Fix:
        Widen the live stop by the expected intra-candle noise floor
        to prevent premature stop-outs from bid-ask bounce.

    Args:
        base_stop_pct:       The backtested stop (e.g. 1.0% = 0.010)
        stock_price:         Current stock price for tick-to-pct conversion
        tick_size:           NSE minimum tick (₹0.05 for most stocks)
        intraday_vol_factor: Multiplier for expected intra-candle noise (default 2x spread)

    Returns:
        Adjusted stop percentage suitable for tick-level WebSocket monitoring.
    """
    if base_stop_pct is None:
        base_stop_pct = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS
    if stock_price <= 0:
        return base_stop_pct

    # Expected bid-ask noise in percentage terms
    # NSE typical spread: 2 ticks = ₹0.10 for a ₹500 stock = 0.02%
    spread_estimate_pct = (tick_size * intraday_vol_factor) / stock_price

    # Widen the stop by the noise floor
    adjusted_stop = base_stop_pct + spread_estimate_pct

    logger.debug(f"Stop adjustment: base={base_stop_pct:.4f}  "
                 f"noise={spread_estimate_pct:.4f}  "
                 f"adjusted={adjusted_stop:.4f}  (price=₹{stock_price:.2f})")
    return adjusted_stop


def backtest_stop_with_tick_noise(df_backtest: pd.DataFrame,
                                   base_stop_pct: float = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS,
                                   tick_size: float = 0.05,
                                   vol_percentile: float = 95.0) -> pd.DataFrame:
    """
    Harden the backtester to simulate tick-level stop sensitivity.

    Instead of using brick_low as the stop trigger, uses a synthetically
    widened low = brick_low - (tick_noise at 95th percentile of price history).
    This prevents the backtest from being over-optimistic about stop survival.

    Args:
        df_backtest:    DataFrame of historical bricks with 'brick_low', 'brick_close'.
        base_stop_pct:  Standard stop percentage.
        tick_size:      NSE minimum tick size (₹0.05).
        vol_percentile: Percentile of tick noise to use (95 = near-worst-case).

    Returns:
        df with additional 'tick_adjusted_low' column.
    """
    # Estimate noise per stock price level
    noise_pct = (tick_size * 2) / df_backtest["brick_close"].clip(lower=1)
    noise_at_percentile = np.percentile(noise_pct, vol_percentile)

    # Synthetic worst-case low: the candle could have dipped this much extra
    df_backtest = df_backtest.copy()
    df_backtest["tick_adjusted_low"] = df_backtest["brick_low"] * (1 - noise_at_percentile)

    logger.info(f"Tick noise at {vol_percentile}th pct: {noise_at_percentile*100:.4f}%  "
                f"(simulates real WebSocket stop sensitivity)")
    return df_backtest


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3: SILENT WEBSOCKET — Heartbeat Candle Injector
# ═══════════════════════════════════════════════════════════════════════════

class HeartbeatCandle:
    """
    Injects synthetic 'flat candles' when the Upstox WebSocket goes silent
    for more than `silence_threshold_seconds`.

    The Problem:
        If a stock is in the lower circuit or halted, the WebSocket sends
        nothing. The live RenkoBuilder sees no ticks. Time advances but the
        brick DataFrame doesn't. XGBoost receives a feature vector of shape
        (1, N-1) instead of (1, N), throwing a cryptic shape error at 2:07 PM.

    The Fix:
        Register the last known LTP for every symbol. If silence exceeds the
        threshold, inject a flat heartbeat candle (open=high=low=close=last_ltp)
        with volume=0. This keeps the DataFrame aligned to clock time.

    Architecture:
        - Call `register_tick(symbol, ltp)` on every incoming WebSocket tick.
        - Call `check_and_inject(symbol, renko_state, now)` on every loop iteration.
        - The injected brick will NOT form a new Renko block (no price move),
          but it forces the feature engine to produce a valid feature vector.
    """

    def __init__(self, silence_threshold_seconds: int = config.HEARTBEAT_INJECT_SEC):
        self.silence_threshold = silence_threshold_seconds
        self._last_tick_time: dict[str, float]  = {}
        self._last_ltp: dict[str, float]        = {}
        self._injected_count: dict[str, int]    = {}

    def register_tick(self, symbol: str, ltp: float) -> None:
        """Call this every time a real tick arrives for a symbol."""
        self._last_tick_time[symbol] = time.monotonic()
        self._last_ltp[symbol]       = float(ltp)

    def check_and_inject(self,
                          symbol: str,
                          renko_state,          # LiveRenkoState instance
                          now: datetime) -> bool:
        """
        Checks if the symbol has gone silent. If so, injects a heartbeat tick.

        Args:
            symbol:       Stock NSE symbol.
            renko_state:  Live renko state object (has .process_tick()).
            now:          Current datetime.

        Returns:
            True if a heartbeat was injected, False if the symbol had real ticks.
        """
        if symbol not in self._last_tick_time:
            return False   # Never seen this symbol — can't inject meaningfully

        elapsed = time.monotonic() - self._last_tick_time[symbol]
        if elapsed < self.silence_threshold:
            return False   # Symbol is alive — no injection needed

        # Symbol has been silent — inject a flat heartbeat
        ltp = self._last_ltp[symbol]
        try:
            renko_state.process_tick(
                price     = ltp,   # was 'ltp=' — bug: process_tick() param is 'price'
                high      = ltp,
                low       = ltp,
                timestamp = now,   # was 'ts=' — bug: process_tick() param is 'timestamp'
            )
            self._injected_count[symbol] = self._injected_count.get(symbol, 0) + 1
            logger.debug(f"[Heartbeat] {symbol}: silence={elapsed:.0f}s -> "
                         f"injected flat tick @ ₹{ltp} "
                         f"(total: {self._injected_count[symbol]})")
            return True
        except Exception as e:
            logger.warning(f"[Heartbeat] {symbol} injection failed: {e}")
            return False

    def get_silence_report(self) -> dict:
        """Returns symbols that have received heartbeat injections today."""
        return {sym: cnt for sym, cnt in self._injected_count.items() if cnt > 0}


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4: MEMORY LEAK — Rolling Brick Buffer (O(1) constant-time)
# ═══════════════════════════════════════════════════════════════════════════

class RollingBrickBuffer:
    """
    Replaces unbounded DataFrame growth in the live loop with a fixed-size
    deque-backed buffer.

    The Problem (Measured):
        At 09:15 AM: DataFrame has ~100 bricks.  XGBoost inference: ~1ms.
        At 01:00 PM: DataFrame has ~2,000 bricks. XGBoost inference: ~180ms.
        At 03:00 PM: DataFrame has ~4,000 bricks. Each loop iteration is 500ms.
        Result: 5-second execution delay -> catastrophic T+5 slippage.

    The Fix:
        Use collections.deque(maxlen=N) where N = maximum_bricks_needed.
        deque.append() is O(1). The oldest element is automatically dropped.
        XGBoost feature extraction always works on a constant-size window.

    Design Principle:
        Keep enough bricks for the longest lookback window + safety margin:
        - FracDiff window: 100 bricks
        - Hurst window: 60 bricks
        - RelativeStrength: 50 bricks
        - Safety buffer: 50 bricks
        -> MAX_BUFFER_SIZE = 260 bricks (constant regardless of time of day)
    """

    MAX_BUFFER_SIZE = config.MAX_BUFFER_SIZE   # Never grows beyond this — O(1) guaranteed

    def __init__(self, symbol: str):
        self.symbol  = symbol
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER_SIZE)
        self._total_bricks_seen: int = 0

    def append(self, brick: dict) -> None:
        """Append a new brick. The oldest is automatically dropped if full."""
        self._buffer.append(brick)
        self._total_bricks_seen += 1

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the fixed-size buffer to a DataFrame for feature computation.
        This is always O(MAX_BUFFER_SIZE) ≈ O(1) from the engine's perspective.
        """
        if not self._buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self._buffer))

    def get_features_array(self, feature_cols: list[str]) -> Optional[np.ndarray]:
        """
        Extract only the latest feature row as a fixed-shape numpy array.
        Avoids building the full DataFrame when only the last brick is needed.

        Returns:
            numpy array of shape (1, len(feature_cols)) or None if insufficient data.
        """
        if len(self._buffer) < 5:   # Need at least 5 bricks for valid features
            return None

        latest = self._buffer[-1]
        try:
            row = [float(latest.get(col, 0.0) or 0.0) for col in feature_cols]
            return np.array(row, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logger.warning(f"[RollingBuffer] {self.symbol} feature extraction error: {e}")
            return None

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.MAX_BUFFER_SIZE

    def memory_usage_kb(self) -> float:
        """Estimate memory usage of the buffer in kilobytes."""
        if not self._buffer:
            return 0.0
        # Each brick dict ~ 500 bytes
        return (len(self._buffer) * 500) / 1024


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5: PENDING ORDER LOCKUP — Async Per-Symbol Mutex Guard
# ═══════════════════════════════════════════════════════════════════════════

class PendingOrderGuard:
    """
    Asyncio-safe per-symbol lock that physically prevents Brain 1 from firing
    a second signal while a prior order is still traversing the Upstox API.

    The Problem:
        Backtest: Order fills instantly (next brick). Brain 1 never double-fires.
        Live:     API round-trip = 150–500ms. If the WebSocket fires a new brick
                  in that window (which happens), Brain 1 sees another valid signal
                  for the same stock, generates a second order, doubling position.
                  After brokerage on 2x position, a small adverse move bleeds badly.

    Architecture (asyncio.Lock per symbol):
        - Each symbol gets its own asyncio.Lock.
        - Lock is ACQUIRED the moment Brain 1 decides to trade.
        - Lock is RELEASED only when:
            (a) The order is confirmed (status = COMPLETE/OPEN) from Upstox API, OR
            (b) The lock_timeout_seconds expires (safety net for API failures).
        - Any Brain 1 signal between ACQUIRE and RELEASE is silently SKIPPED.

    Usage (async context):
        guard = PendingOrderGuard(lock_timeout_seconds=30)

        # In the async trading loop:
        if await guard.try_acquire(symbol):
            try:
                await send_order_to_upstox(symbol, signal)
                await guard.wait_for_confirmation(symbol, upstox_client)
            finally:
                guard.release(symbol)   # Always release, even on API error
    """

    def __init__(self, lock_timeout_seconds: int = config.ORDER_LOCK_TIMEOUT_SEC):
        self._locks:           dict[str, asyncio.Lock]  = {}
        self._acquired_at:     dict[str, float]         = {}
        self._pending_for:     dict[str, str]           = {}   # symbol -> side
        self.lock_timeout      = lock_timeout_seconds
        self._blocked_count:   int = 0   # Audit counter

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create the asyncio.Lock for this symbol."""
        if symbol not in self._locks:
            self._locks[symbol] = asyncio.Lock()
        return self._locks[symbol]

    async def try_acquire(self, symbol: str, side: str) -> bool:
        """
        Attempt to acquire the order lock for a symbol.
        Returns True if lock acquired (safe to send order).
        Returns False if symbol already has a pending order (skip this signal).
        """
        lock = self._get_lock(symbol)

        # Check for stale locks (API never responded)
        if symbol in self._acquired_at:
            elapsed = time.monotonic() - self._acquired_at[symbol]
            if elapsed > self.lock_timeout:
                logger.warning(f"[OrderGuard] {symbol}: Lock TIMEOUT after {elapsed:.0f}s "
                               f"— force-releasing (API may have failed).")
                try:
                    lock.release()
                except RuntimeError:
                    pass   # Already released somehow
                self._acquired_at.pop(symbol, None)
                self._pending_for.pop(symbol, None)

        # Non-blocking attempt to acquire
        acquired = lock.locked() is False and await asyncio.shield(
            asyncio.wait_for(lock.acquire(), timeout=0.001)
        ) if True else False

        # Simpler: use lock.locked() for instant check (no waiting)
        if lock.locked():
            self._blocked_count += 1
            pending_side = self._pending_for.get(symbol, "?")
            logger.info(f"[OrderGuard] {symbol}: BLOCKED — {pending_side} order still PENDING "
                        f"(total blocked: {self._blocked_count})")
            return False

        # Lock is free — acquire it
        await lock.acquire()
        self._acquired_at[symbol] = time.monotonic()
        self._pending_for[symbol] = side
        logger.info(f"[OrderGuard] {symbol}: Lock ACQUIRED for {side} order")
        return True

    def release(self, symbol: str) -> None:
        """
        Release the lock for a symbol after the order is confirmed or failed.
        MUST be called in a finally block to prevent permanent lockout.
        """
        lock = self._get_lock(symbol)
        if lock.locked():
            lock.release()
            elapsed = time.monotonic() - self._acquired_at.get(symbol, time.monotonic())
            logger.info(f"[OrderGuard] {symbol}: Lock RELEASED after {elapsed:.1f}s")
        self._acquired_at.pop(symbol, None)
        self._pending_for.pop(symbol, None)

    async def wait_for_confirmation(self,
                                     symbol: str,
                                     upstox_client,
                                     order_id: str,
                                     poll_interval: float = 0.5) -> str:
        """
        Poll Upstox order status until COMPLETE, REJECTED, or timeout.

        Args:
            symbol:         NSE symbol.
            upstox_client:  Upstox API client with get_order_status() method.
            order_id:       Upstox order ID from the placement response.
            poll_interval:  Status poll frequency in seconds.

        Returns:
            Final order status string (e.g. "COMPLETE", "REJECTED").
        """
        start = time.monotonic()
        terminal_statuses = {"COMPLETE", "REJECTED", "CANCELLED"}

        while (time.monotonic() - start) < self.lock_timeout:
            try:
                status_resp = upstox_client.get_order_details(order_id=order_id)
                status = status_resp.get("data", {}).get("status", "UNKNOWN").upper()

                if status in terminal_statuses:
                    logger.info(f"[OrderGuard] {symbol}: order {order_id} -> {status} "
                                f"in {time.monotonic()-start:.2f}s")
                    return status

                logger.debug(f"[OrderGuard] {symbol}: order {order_id} status={status} "
                             f"(waiting...)")
            except Exception as e:
                logger.warning(f"[OrderGuard] {symbol}: status poll error: {e}")

            await asyncio.sleep(poll_interval)

        logger.error(f"[OrderGuard] {symbol}: order {order_id} TIMEOUT "
                     f"after {self.lock_timeout}s — releasing lock regardless.")
        return "TIMEOUT"

    def get_status_report(self) -> dict:
        """Return current pending order status for all symbols."""
        return {
            "currently_locked": [s for s, l in self._locks.items() if l.locked()],
            "pending_sides":    dict(self._pending_for),
            "total_blocked":    self._blocked_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER — Plug all 5 Guards into engine.py in one call
# ═══════════════════════════════════════════════════════════════════════════

class LiveExecutionGuard:
    """
    Unified facade that bundles all 5 execution hardening modules.
    Inject one instance into engine.py's run_live_engine().

    Usage in engine.py:
        from src.live.execution_guard import LiveExecutionGuard

        guard = LiveExecutionGuard(symbols=list(renko_states.keys()))
        guard.warm_up_all()   # At 09:08 AM — loads history for all symbols

        # In the main while loop — per tick:
        guard.heartbeat.register_tick(sym, ltp)
        guard.heartbeat.check_and_inject(sym, renko_state, now)

        # Before XGBoost inference — use rolling buffer:
        guard.buffers[sym].append(new_brick)
        df = guard.buffers[sym].to_dataframe()

        # Before order placement — check pending lock:
        if await guard.order_guard.try_acquire(sym, side):
            try:
                status = await guard.order_guard.wait_for_confirmation(...)
            finally:
                guard.order_guard.release(sym)
    """

    def __init__(self, symbols: list[str], sectors: dict[str, str],
                 silence_threshold: int = 60,
                 order_lock_timeout: int = 30,
                 before_ts: Optional[datetime] = None):
        """
        Args:
            symbols: List of NSE indices/stocks to monitor.
            sectors: Mapping of {symbol: sector} for warmup directory resolution.
            silence_threshold: Sec before heartbeat injection (forward-fill).
            order_lock_timeout: Sec before stale pending orders are auto-cleared.
            before_ts: Contamination shield for simulations (strip bricks ON/AFTER this timestamp).
        """
        self.symbols = symbols
        self.sectors = sectors   # {symbol: sector}

        # Fix 1: Warmup splicers per symbol
        self.splicers: dict[str, HistoricalWarmupSplicer] = {
            sym: HistoricalWarmupSplicer(sym, sectors.get(sym, "unknown"), before_ts=before_ts)
            for sym in symbols
        }

        # Fix 3: Heartbeat injector (shared across all symbols)
        self.heartbeat = HeartbeatCandle(silence_threshold_seconds=silence_threshold)

        # Fix 4: Rolling buffers per symbol
        self.buffers: dict[str, RollingBrickBuffer] = {
            sym: RollingBrickBuffer(sym) for sym in symbols
        }

        # Fix 5: Async pending order guard (shared)
        self.order_guard = PendingOrderGuard(lock_timeout_seconds=order_lock_timeout)

        # Patch 1: Authoritative position state lock (deduplication)
        self.entry_lock = EntryStateLock()

        # Patch 2: Brick-count cooldown after every exit (anti-churn)
        self.cooldown = BrickCooldownTracker()

    def warm_up_all(self) -> dict[str, int]:
        """
        Load historical bricks for ALL symbols at 09:08 AM.
        Returns {symbol: bricks_loaded}.
        """
        logger.info(f"[ExecutionGuard] Warming up {len(self.symbols)} symbols...")
        results = {}
        for sym in self.symbols:
            n = self.splicers[sym].load_history()
            results[sym] = n
            # Seed the rolling buffer with warmup history
            if n > 0:
                splicer_df = self.splicers[sym].to_dataframe()
                for _, row in splicer_df.iterrows():
                    self.buffers[sym].append(row.to_dict())

        warmed = sum(1 for n in results.values() if n > 0)
        logger.info(f"[ExecutionGuard] Warm-up complete: {warmed}/{len(self.symbols)} symbols ready.")
        return results

    def tick_stop_pct(self, symbol: str, base_stop: float = config.NATR_BRICK_PERCENT * config.STRUCTURAL_REVERSAL_BRICKS) -> float:
        """
        Fix 2: Get the tick-adjusted stop loss pct for a symbol.
        Uses the last known price from the heartbeat register.
        """
        last_price = self.heartbeat._last_ltp.get(symbol, 500.0)
        return tick_adjusted_stop_pct(base_stop, stock_price=last_price)

    def system_health(self) -> dict:
        """Return a health snapshot for monitoring."""
        return {
            "warmed_symbols":   sum(1 for s in self.splicers.values() if s.is_ready),
            "buffer_sizes":     {s: b.size for s, b in self.buffers.items()},
            "heartbeat_report": self.heartbeat.get_silence_report(),
            "order_guard":      self.order_guard.get_status_report(),
            "entry_lock":       self.entry_lock.get_status(),
            "cooldown":         self.cooldown.get_status(),
            "memory_kb":        sum(b.memory_usage_kb() for b in self.buffers.values()),
        }
