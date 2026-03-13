"""
src/core/features.py — Feature Calculators (v3.0 — Institutional Alpha)
=========================================================================
Features computed on Renko brick DataFrames:

  Core (Phase 1):
  1. Renko Velocity       — log₁₀(avg_dur_last_10 / current_dur)
  2. Wick Pressure        — True wick delta relative to brick body (direction-aware)
  3. Relative Strength    — stock_Z − sector_Z (rolling 50-brick)
  4. FracDiff Price       — Fractionally differentiated price (memory-preserving stationarity)
  5. Hurst Exponent       — Regime filter (H > 0.55 = trending)
  6-14. Long-lookback features: velocity_long, trend_slope, rolling_range_pct, etc.

  Phase 2 (Institutional Alpha — Anti-Peak, Pro-Early-Entry):
  15. VWAP Z-Score        — Distance from 20-brick VWAP. >2.5σ = exhaustion peak (BLOCK LONG).
  16. VPT Acceleration    — 2nd derivative of Volume Price Trend. Spike = Absorption before rally.
  17. Squeeze Z-Score     — Brick density Z-score. Expansion after squeeze = breakout.
  18. Streak Exhaustion   — Sigmoid decay applied after 8+ consecutive same-direction bricks.

Also provides placeholder columns for future extensions:
  • whale_oi_score    (NaN — Option Chain OI)
  • sentiment_score   (NaN — Sentiment Engine)
"""

import numpy as np
import pandas as pd
from typing import Optional

import config


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_velocity(df: pd.DataFrame, lookback: int = config.VELOCITY_LOOKBACK) -> pd.Series:
    """
    Renko Velocity (Momentum)
    ─────────────────────────
    Formula:  log₁₀( avg_dur_last_N / current_dur )
    Positive -> explosive (institutional) · Negative -> grinding (retail)
    
    Fix: Distribute duration for synthetic intra-candle bricks safely so 
    velocity doesn't spike artificially when multiple bricks share a minute.
    """
    durations = df["duration_seconds"].copy()
    
    # Identify identical timestamps (synthetic OHLCV expansion bricks)
    ts = df["brick_timestamp"]
    
    # Groups of identical timestamps
    # Normalise element-wise: the buffer can hold a mix of tz-aware (warmup) and
    # tz-naive (today's ticks) Timestamps, which makes groupby crash.
    ts = ts.apply(lambda t: t.tz_localize(None) if (hasattr(t, "tzinfo") and t.tzinfo is not None) else t)
    identicals = ts.groupby(ts).transform('count')
    
    # For identical timestamps, artificially space their duration.
    # Instead of them all being 1 second, assume they took equal fractions of 60 seconds.
    # We clip to at least 1 second to avoid div by zero.
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.MIN_BRICK_DURATION), durations)
    
    s_durations = pd.Series(durations, index=df.index)

    avg_dur = s_durations.rolling(window=lookback, min_periods=1).mean()
    ratio = avg_dur / s_durations.clip(lower=config.MIN_BRICK_DURATION)
    return np.log10(ratio.clip(lower=1e-9))


def compute_wick_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Wick Pressure (Hidden Flow) — Direction-Aware
    ──────────────────────────────────────────────
    LONG bricks:  upper wick = (brick_high − brick_close) / brick_size
                  High value → price ran up then got rejected above close → absorption trap
    SHORT bricks: lower wick = (brick_close − brick_low)  / brick_size
                  High value → price fell then bounced above close → support trap

    Previously used `abs(brick_high − brick_close)` for ALL bricks, which measured the
    ENTIRE brick body for SHORT bricks (giving wick_pressure=1.0 always) and blocked
    100% of short entries. This fix correctly measures only the shadow/wick beyond close.
    """
    is_long = df["direction"] > 0
    long_wick  = (df["brick_high"] - df["brick_close"]) / df["brick_size"].clip(lower=1e-9)
    short_wick = (df["brick_close"] - df["brick_low"])  / df["brick_size"].clip(lower=1e-9)
    return long_wick.where(is_long, short_wick).clip(lower=0.0)



def compute_consecutive_same_dir(df: pd.DataFrame) -> pd.Series:
    """
    Consecutive Same Direction
    ──────────────────────────
    Count of consecutive bricks in the same direction ending at current brick.
    High -> strong trend (safe to enter) · Low (1-2) -> choppy / whipsaw risk.
    """
    if not getattr(config, "FEATURE_OPTIMIZATION_ENABLED", True):
        from src.core.legacy_logic import compute_consecutive_same_dir_ITERATIVE
        return compute_consecutive_same_dir_ITERATIVE(df)

    dirs = df["direction"].values
    if len(dirs) == 0: return pd.Series([], index=df.index)
    
    # Vectorized consecutive count using group-by logic
    # Find points where direction changes
    changes = np.diff(dirs, prepend=dirs[0] + 1) != 0
    # Create a grouping ID for each streak of same direction
    group_ids = np.cumsum(changes)
    # Calculate the position within each streak
    # (index - first_index_of_group + 1)
    group_starts = np.where(changes)[0]
    counts = np.arange(len(dirs)) - group_starts[group_ids - 1] + 1
    return pd.Series(counts, index=df.index)


def compute_brick_oscillation_rate(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Brick Oscillation Rate
    ──────────────────────
    Fraction of direction changes in the last N bricks.
    High (>0.6) -> whipsaw/choppy regime · Low (<0.3) -> clean trend.
    """
    dirs = df["direction"]
    changes = (dirs != dirs.shift(1)).astype(float)
    return changes.rolling(window=window, min_periods=1).mean()


# ═══════════════════════════════════════════════════════════════════════════
# LONG-LOOKBACK FEATURES (Anti-Myopia Fix)
# ═══════════════════════════════════════════════════════════════════════════

def compute_velocity_long(df: pd.DataFrame, lookback: int = config.VELOCITY_LONG_LOOKBACK) -> pd.Series:
    """
    Long-Period Renko Velocity (20-brick momentum)
    ───────────────────────────────────────────────
    Same formula as compute_velocity but using a 20-brick window.
    Captures sustained institutional momentum vs single-brick noise.
    Formula: log₁₀( avg_dur_last_20 / current_dur )
    """
    durations = df["duration_seconds"].copy()
    ts = df["brick_timestamp"]
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.VELOCITY_LONG_MIN_DURATION), durations)
    s_durations = pd.Series(durations, index=df.index)
    avg_dur = s_durations.rolling(window=lookback, min_periods=max(1, lookback // 4)).mean()
    ratio = avg_dur / s_durations.clip(lower=config.VELOCITY_LONG_MIN_DURATION)
    return np.log10(ratio.clip(lower=1e-9))


def compute_trend_slope(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Linear Regression Slope of Brick Close (14-brick)
    ──────────────────────────────────────────────────
    Computes the OLS slope β₁ of price over the last N bricks.
    Positive → sustained up-move. Negative → down-move. Near-zero → sideways.
    Normalized by average price to make it scale-invariant (% per brick).
    """
    closes = df["brick_close"].values.astype(float)
    n = len(closes)
    slopes = np.empty(n)
    slopes[:] = 0.0
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        beta = cov / (x_var + 1e-9)
        slopes[i] = beta / max(y_mean, 1e-9)   # normalize: slope as % of price
    slopes[:window - 1] = 0.0
    return pd.Series(slopes, index=df.index)


def compute_rolling_range_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Rolling Price Range % (14-brick volatility measure)
    ─────────────────────────────────────────────────────
    (max_close − min_close) / avg_close over last N bricks.
    High → trending / breakout · Low → compressed / coiling before move.
    """
    closes = df["brick_close"]
    high_r = closes.rolling(window=window, min_periods=1).max()
    low_r  = closes.rolling(window=window, min_periods=1).min()
    avg_r  = closes.rolling(window=window, min_periods=1).mean().clip(lower=1e-9)
    return (high_r - low_r) / avg_r


def compute_momentum_acceleration(df: pd.DataFrame,
                                   fast: int = config.VELOCITY_LOOKBACK,
                                   slow: int = config.VELOCITY_LONG_LOOKBACK) -> pd.Series:
    """
    Momentum Acceleration (fast − slow velocity diff)
    ──────────────────────────────────────────────────
    Difference between the 5-brick and 14-brick Renko velocity.
    Positive → momentum accelerating (entry signal strengthening).
    Negative → momentum decelerating (hold or exit warning).
    """
    durations = df["duration_seconds"].copy()
    ts = df["brick_timestamp"]
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.VELOCITY_LONG_MIN_DURATION), durations)
    s_dur = pd.Series(durations, index=df.index).clip(lower=config.VELOCITY_LONG_MIN_DURATION)

    avg_fast = s_dur.rolling(window=fast, min_periods=1).mean()
    avg_slow = s_dur.rolling(window=slow, min_periods=1).mean()

    vel_fast = np.log10((avg_fast / s_dur).clip(lower=1e-9))
    vel_slow = np.log10((avg_slow / s_dur).clip(lower=1e-9))
    return vel_fast - vel_slow




def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-Score: (x − μ) / σ."""
    mu = series.rolling(window=window, min_periods=1).mean()
    sigma = series.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
    return (series - mu) / sigma


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: INSTITUTIONAL ALPHA FACTORS (Anti-Peak, Pro-Early-Entry)
# ═══════════════════════════════════════════════════════════════════════════

def compute_vwap_zscore(
    df: pd.DataFrame,
    window: int = config.VWAP_WINDOW,
) -> pd.Series:
    """
    VWAP Z-Score — The Institutional Anchor
    ────────────────────────────────────────
    Measures how many standard deviations the current brick close is from
    the Volume-Weighted Average Price over the last `window` bricks.

    VWAP_t = Σ(typical_price × volume) / Σ(volume)   [rolling window]
    Z = (close - VWAP) / σ(close, window)

    Anti-Peak Logic:
        Z > +2.5 → price is in the upper exhaustion zone (ABB peak scenario)
                   → BLOCK LONG entries (model will penalize this)
        Z < -2.5 → price is deeply oversold from VWAP
                   → SHORT entries or mean-reversion LONG opportunity
        Z ∈ [-2.5, +1.0] → Normal range — entries are fair-valued

    Requires: df must have 'volume' and 'typical_price' columns
    (populated by RenkoBrickBuilder v3.0 via volume passthrough).
    Falls back to price Z-score if volume is missing/zero.
    """
    has_volume = (
        "volume" in df.columns
        and "typical_price" in df.columns
        and df["volume"].sum() > 0
    )

    if has_volume:
        tp  = df["typical_price"].fillna(df["brick_close"])
        vol = df["volume"].fillna(0.0).clip(lower=0.0)
        tp_vol = (tp * vol).rolling(window=window, min_periods=1).sum()
        vol_sum = vol.rolling(window=window, min_periods=1).sum().clip(lower=1e-9)
        vwap = tp_vol / vol_sum
    else:
        # Graceful fallback: VWAP ≈ rolling mean of close
        vwap = df["brick_close"].rolling(window=window, min_periods=1).mean()

    close  = df["brick_close"]
    sigma  = close.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
    return ((close - vwap) / sigma).clip(lower=-5.0, upper=5.0)


def compute_vpt_acceleration(
    df: pd.DataFrame,
    diff_lag: int = config.VPT_ACCEL_DIFF,
) -> pd.Series:
    """
    VPT Acceleration — Institutional Footprint Detector
    ────────────────────────────────────────────────────
    Volume Price Trend (VPT) captures institutional buying pressure:
        VPT_t = VPT_{t-1} + Volume × (Close_t - Close_{t-1}) / Close_{t-1}

    VPT Acceleration = Δ²(VPT) = VPT_t - 2·VPT_{t-1} + VPT_{t-2}
    (The 2nd derivative of the trend)

    Early-Entry Signal:
        High Volume + Flat Price → VPT changes, but Close barely moves
        → VPT Acceleration spikes POSITIVE → Institutions are absorbing supply
        → This is the "Absorption" signal that PRECEDES a rally by 2-5 bricks
        → The ABB rally would have shown this signature before the big move up

    Exhaustion Signal:
        Negative VPT Acceleration after a long streak → supply is hitting bids
        → Peak is forming, exit LONG / enter SHORT

    Requires: df must have 'volume' column.
    Falls back to zero series if volume is missing.
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(0.0, index=df.index)

    close = df["brick_close"]
    vol   = df["volume"].fillna(0.0)

    # Price return per brick (log return, direction-preserving)
    price_ret = close.pct_change().fillna(0.0)

    # VPT: cumulative sum of (volume × price_return)
    vpt = (vol * price_ret).cumsum()

    # 2nd-order finite difference (acceleration)
    vpt_accel = vpt.diff(diff_lag).diff(diff_lag)

    # Normalize to z-score for model compatibility (prevents scale issues)
    return compute_zscore(vpt_accel.fillna(0.0), window=20)


def compute_squeeze_zscore(
    df: pd.DataFrame,
    window: int = config.SQUEEZE_WINDOW,
) -> pd.Series:
    """
    Volatility Squeeze Z-Score — Coil & Breakout Detector
    ───────────────────────────────────────────────────────
    Measures the TIME DENSITY of brick formation:
        density = bricks_formed / time_elapsed (bricks per minute)

    High density  → market is moving fast, many bricks form quickly
    Low density   → market is compressed/coiling (the "squeeze")

    Z-Score of density:
        Z < -1.0  → Below-average brick rate = SQUEEZE phase (coiling)
        Z > +1.5  → First expansion brick after squeeze = BREAKOUT signal
        Z ∈ [-1, +1] → Normal rate — no specific signal

    Anti-Pattern Prevention:
        Entering during a squeeze (low density) is entering before the
        move. Waiting for Z > 0 after a squeeze gives earlier, cleaner
        entries than waiting for the full trend to establish.
    """
    # Brick density proxy: 1/duration_seconds (bricks per second)
    # Higher brick rate = shorter duration = more dense
    dur = df["duration_seconds"].clip(lower=config.VELOCITY_LONG_MIN_DURATION).fillna(60.0)
    density = 1.0 / dur  # bricks per second (inverse of duration)

    return compute_zscore(density, window=window).clip(lower=-4.0, upper=4.0)


def compute_streak_exhaustion(
    df: pd.DataFrame,
    onset: int = config.STREAK_EXHAUSTION_ONSET,
    scale: float = config.STREAK_EXHAUSTION_SCALE,
) -> pd.Series:
    """
    Streak Exhaustion — Mathematical Momentum Decay Filter
    ───────────────────────────────────────────────────────
    Applies a sigmoid decay function to consecutive same-direction brick
    count to model the exhaustion of momentum after a sustained streak.

    Formula:
        streak = consecutive_same_dir  (from compute_consecutive_same_dir)
        exhaustion = -sigmoid((streak - onset) × scale)

    Behavior:
        streak ≤ 8:   exhaustion ≈ 0.0  (no penalty — momentum is fresh)
        streak = 10:  exhaustion ≈ -0.27 (mild warning)
        streak = 12:  exhaustion ≈ -0.46 (strong warning)
        streak ≥ 15:  exhaustion → -0.50 (maximum penalty, ~late ABB peak)

    XGBoost learns: high negative exhaustion + high probability → FALSE positive.
    This directly teaches the model to reject the ABB peak scenario.

    Range: [-0.5, 0.0]. Zero when streak is fresh, negative when streak is old.
    """
    streak = compute_consecutive_same_dir(df).clip(lower=0)
    # Sigmoid: σ(x) = 1 / (1 + np.exp(-x))
    # Shift so onset → 0, then scale for steepness
    x = (streak - onset) * scale
    sigmoid = 1.0 / (1.0 + np.exp(-x.clip(lower=-50, upper=50)))
    # Negate and halve: range becomes [-0.5, 0]
    return (-sigmoid * 0.5).where(streak >= onset, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: CONTEXTUAL VOLATILITY FEATURES (Mid-Day Trend Sensitivity)
# ═══════════════════════════════════════════════════════════════════════════

def compute_tib_zscore(
    df: pd.DataFrame,
    window: int = 50,
) -> pd.Series:
    """
    Time-in-Brick (TiB) Z-Score — Formation Speed Normalizer
    ──────────────────────────────────────────────────────────
    Measures how fast the current brick formed relative to the
    recent rolling window of brick durations.

    Formula:
        duration_i = timestamp_i - timestamp_{i-1}   (seconds)
        μ = rolling_mean(duration, window).shift(1)   ← NO self-inclusion
        σ = rolling_std(duration, window).shift(1)    ← NO self-inclusion
        TiB_Z = (duration_i - μ) / σ

    Interpretation:
        Z < -1.5  → Current brick formed MUCH faster than recent average
                     → Explosive momentum (institutional flow)
        Z > +1.5  → Current brick formed MUCH slower than recent average
                     → Stalling / liquidity drying up
        Z ∈ [-1, +1] → Normal formation speed

    Anti-Lookahead: shift(1) ensures the rolling window only contains
    durations from PREVIOUS bricks, never the current one.
    Capped to [-5, +5] to prevent gradient explosion in XGBoost.
    """
    # Duration between consecutive brick timestamps (seconds)
    ts = df["brick_timestamp"]
    dur = ts.diff().dt.total_seconds().fillna(60.0).clip(lower=1.0)

    # Rolling stats from PREVIOUS bricks only (shift excludes current)
    mu    = dur.rolling(window=window, min_periods=1).mean().shift(1)
    sigma = dur.rolling(window=window, min_periods=1).std().shift(1).clip(lower=1e-9)

    # Fill the very first row (where shift produces NaN) with 0
    z = ((dur - mu) / sigma).fillna(0.0)
    return z.clip(lower=-5.0, upper=5.0)


def compute_vpb_roc(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Volume-per-Brick (VpB) Rate of Change — Local Volume Spike Detector
    ────────────────────────────────────────────────────────────────────
    Measures the percentage deviation of the current brick's volume from
    the rolling mean of the previous N bricks.

    Formula:
        μ_vol = rolling_mean(volume, window).shift(1)   ← NO self-inclusion
        VpB_RoC = (volume_i - μ_vol) / μ_vol

    Interpretation:
        VpB_RoC > +1.0  → Volume is 2x the recent average → Institutional activity
        VpB_RoC ≈  0.0  → Normal volume
        VpB_RoC < -0.5  → Volume drying up → Caution

    Why this matters for mid-day:
        Absolute volume drops after 10:30 AM, but a LOCAL spike (vs the
        quiet mid-day mean) is a strong signal. This feature captures
        that relative spike without being distorted by morning extremes.

    Falls back to 0.0 if volume column is missing or all zeros.
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(0.0, index=df.index)

    vol = df["volume"].fillna(0.0).clip(lower=0.0)

    # Rolling mean from PREVIOUS bricks only (shift excludes current)
    mu_vol = vol.rolling(window=window, min_periods=1).mean().shift(1).clip(lower=1e-9)

    # Fill the very first row with 0
    roc = ((vol - mu_vol) / mu_vol).fillna(0.0)
    return roc


def compute_market_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical Market Regime (IST) — One-Hot Encoded Time Buckets
    ───────────────────────────────────────────────────────────────
    Splits the trading day into three regimes based on the brick's
    own timestamp. One-hot encoded so XGBoost treats them as logical
    branches, not a continuous scale.

    Regimes:
        0: 09:15 – 10:30  → Morning Momentum (high vol, wide range)
        1: 10:30 – 13:30  → Lunchtime Chop (low vol, mean-reverting)
        2: 13:30 – 15:30  → Afternoon Breakout / Euro Open (trending)

    Output:  3 binary columns:
        regime_morning    (1 during regime 0, else 0)
        regime_midday     (1 during regime 1, else 0)
        regime_afternoon  (1 during regime 2, else 0)

    No lookahead: uses only the current brick's own timestamp.
    """
    ts = df["brick_timestamp"]
    # Decimal hour: 9:30 → 9.5, 13:45 → 13.75
    decimal_hour = ts.dt.hour + ts.dt.minute / 60.0

    regime = pd.Series(1, index=df.index, dtype=int)          # default: midday
    regime = regime.where(~(decimal_hour < 10.5), 0)          # morning: < 10:30
    regime = regime.where(~(decimal_hour >= 13.5), 2)         # afternoon: >= 13:30

    dummies = pd.DataFrame({
        "regime_morning":   (regime == 0).astype(int),
        "regime_midday":    (regime == 1).astype(int),
        "regime_afternoon": (regime == 2).astype(int),
    }, index=df.index)
    return dummies


# ═══════════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class RelativeStrengthCalculator:
    """
    RS = Stock_Z − Sector_Z  (rolling 50-brick, merge_asof aligned).
    Positive -> Leader · Negative -> Laggard
    """

    def __init__(self, window: int = config.RS_ROLLING_WINDOW):
        self.window = window
        self._sector_cache: dict[str, pd.DataFrame] = {}

    def load_sector_index(self, sector: str) -> pd.DataFrame:
        if sector in self._sector_cache:
            return self._sector_cache[sector]

        sector_dir = config.DATA_DIR / sector
        if not sector_dir.exists():
            return pd.DataFrame()

        frames = []
        for subdir in sector_dir.iterdir():
            if subdir.is_dir() and subdir.name.upper().startswith("NIFTY"):
                for pq in sorted(subdir.glob("*.parquet")):
                    frames.append(pd.read_parquet(pq))

        if not frames:
            return pd.DataFrame()

        sdf = pd.concat(frames, ignore_index=True).sort_values("brick_timestamp", kind="mergesort").reset_index(drop=True)
        # Normalize timezone to Asia/Kolkata (Upstox uses pytz.FixedOffset(330))
        if sdf["brick_timestamp"].dt.tz is not None:
            sdf["brick_timestamp"] = sdf["brick_timestamp"].dt.tz_convert("Asia/Kolkata")
        else:
            sdf["brick_timestamp"] = sdf["brick_timestamp"].dt.tz_localize("Asia/Kolkata")
        sdf["sector_zscore"] = compute_zscore(sdf["brick_close"], self.window)
        sdf = sdf[["brick_timestamp", "sector_zscore"]].copy()
        self._sector_cache[sector] = sdf
        return sdf

    def compute_rs(self, stock_df: pd.DataFrame, sector: str) -> pd.Series:
        sector_df = self.load_sector_index(sector)
        if sector_df.empty:
            return pd.Series(0.0, index=stock_df.index)

        stock_z = compute_zscore(stock_df["brick_close"], self.window)
        temp = stock_df[["brick_timestamp"]].copy()
        temp["stock_zscore"] = stock_z.values

        # ── Timezone Safety ─────────────────────────────────────────────────
        # Warmup bricks loaded from Parquet carry tz-aware timestamps
        # (Asia/Kolkata), while live tick-formed bricks from the spoofer/
        # paper_trader are tz-naive. merge_asof raises TypeError if the two
        # sides have mismatched tz awareness. Strip both to naive UTC to be safe.
        def _to_naive(col: pd.Series) -> pd.Series:
            if pd.api.types.is_datetime64_any_dtype(col) and col.dt.tz is not None:
                return col.dt.tz_convert("UTC").dt.tz_localize(None)
            return col

        temp = temp.copy()
        temp["brick_timestamp"] = _to_naive(temp["brick_timestamp"])
        sector_df = sector_df.copy()
        sector_df["brick_timestamp"] = _to_naive(sector_df["brick_timestamp"])
        # ────────────────────────────────────────────────────────────────────

        merged = pd.merge_asof(
            temp.sort_values("brick_timestamp", kind="mergesort"),
            sector_df.sort_values("brick_timestamp", kind="mergesort"),
            on="brick_timestamp",
            direction="backward",
        )
        rs_raw = merged["stock_zscore"] - merged["sector_zscore"].fillna(0)
        
        # Apply smoothing to ignore 1-day noise/shakeouts
        window = getattr(config, "RS_SMOOTHING_WINDOW", 1)
        if window > 1:
            return rs_raw.rolling(window=window, min_periods=1).mean().values
        return rs_raw.values



# ═══════════════════════════════════════════════════════════════════════════
# PLACEHOLDER COLUMNS (Future Extensions)
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: Whale OI Tracker
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Integrate NSE F&O option chain data to detect institutional accumulation.
# HOW TO ACTIVATE:
#   1. Subscribe to an NSE option chain data provider (e.g. Upstox options API).
#   2. Compute net OI change per strike, then score the stock from -1 (bearish) to +1 (bullish).
#   3. Plug the score into the XGBoost feature set and retrain the model.
# ─────────────────────────────────────────────────────────────────────────────
# def add_whale_oi_placeholder(df: pd.DataFrame) -> pd.DataFrame:
#     """[FUTURE] Whale Tracker — Option Chain OI data."""
#     df["whale_oi_score"] = np.nan
#     return df


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE: News Sentiment Score Feature Column
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Add the FinBERT sentiment score as a direct training feature for the model.
# HOW TO ACTIVATE:
#   1. The HybridNewsEngine (src/core/hybrid_news.py) already computes the score live.
#   2. When retraining, join the historical sentiment scores to the brick DataFrame by timestamp.
#   3. Use this function to initialize the column to NaN for tickers with no news on a given day.
#   4. Add 'sentiment_score' to the FEATURE_COLS list in config.py and retrain.
# ─────────────────────────────────────────────────────────────────────────────
# def add_sentiment_placeholder(df: pd.DataFrame) -> pd.DataFrame:
#     """[FUTURE] Sentiment Engine — News/Social sentiment score."""
#     df["sentiment_score"] = np.nan
#     return df


# ═══════════════════════════════════════════════════════════════════════════
# LIVE-MODE HELPER (used by src/live/engine.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_features_live(
    bricks_df: pd.DataFrame,
    sector_bricks_df: pd.DataFrame,
    fracdiff_d: float = config.FRACDIFF_D,
    hurst_window: int = config.HURST_WINDOW,
    hurst_threshold: float = config.HURST_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute the full feature set on a live (incrementally growing) brick DataFrame.
    Produces all 18 features the model is trained on:

    Core (14):
      velocity, wick_pressure, relative_strength, brick_size,
      duration_seconds, consecutive_same_dir, brick_oscillation_rate,
      fracdiff_price, hurst, is_trending_regime, velocity_long,
      trend_slope, rolling_range_pct, momentum_acceleration

    Phase 2 (4 new institutional alpha features):
      vwap_zscore, vpt_acceleration, squeeze_zscore, streak_exhaustion
    """
    from src.core.quant_fixes import (
        FractionalDifferentiator,
        compute_hurst_exponent,
    )

    df = bricks_df.copy()

    # ── Timezone Safety (Global) ─────────────────────────────────────────────
    # Historical warmup bricks loaded from Parquet carry tz-aware (Asia/Kolkata)
    # timestamps.  Live tick-formed bricks from the spoofer / paper_trader are
    # tz-naive.  Mixing them causes TypeError in any timestamp comparison
    # (compute_velocity groupby, merge_asof for RS, etc.).
    # Strip tz from both inputs ONCE here so ALL downstream feature functions
    # receive a consistent tz-naive column.
    def _strip_tz(col: pd.Series) -> pd.Series:
        # Robustly convert to UTC and then remove tz info, handles mixed object types
        return pd.to_datetime(col, utc=True).dt.tz_localize(None)

    if "brick_timestamp" in df.columns:
        df["brick_timestamp"] = _strip_tz(df["brick_timestamp"])
    if not sector_bricks_df.empty and "brick_timestamp" in sector_bricks_df.columns:
        sector_bricks_df = sector_bricks_df.copy()
        sector_bricks_df["brick_timestamp"] = _strip_tz(sector_bricks_df["brick_timestamp"])
    # ────────────────────────────────────────────────────────────────────────

    # ── Core features ────────────────────────────────────────────────────────
    df["velocity"]              = compute_velocity(df)
    df["velocity_long"]         = compute_velocity_long(df)
    df["wick_pressure"]         = compute_wick_pressure(df)
    df["trend_slope"]           = compute_trend_slope(df)
    df["rolling_range_pct"]     = compute_rolling_range_pct(df)
    df["momentum_acceleration"] = compute_momentum_acceleration(df)

    # ── Relative Strength ────────────────────────────────────────────────────
    stock_z = compute_zscore(df["brick_close"], config.RS_ROLLING_WINDOW)
    if not sector_bricks_df.empty:
        sector_z = compute_zscore(sector_bricks_df["brick_close"], config.RS_ROLLING_WINDOW)
        ts = pd.DataFrame({"brick_timestamp": df["brick_timestamp"], "stock_z": stock_z.values})
        ss = pd.DataFrame({"brick_timestamp": sector_bricks_df["brick_timestamp"], "sector_z": sector_z.values})

        # ── Timezone Safety ──────────────────────────────────────────────────
        # Warmup bricks (from Parquet) are tz-aware; live tick-formed bricks
        # are tz-naive.  merge_asof raises TypeError when they differ.
        def _to_naive(col: pd.Series) -> pd.Series:
            if pd.api.types.is_datetime64_any_dtype(col) and col.dt.tz is not None:
                return col.dt.tz_convert("UTC").dt.tz_localize(None)
            return col

        ts["brick_timestamp"] = _to_naive(ts["brick_timestamp"])
        ss["brick_timestamp"] = _to_naive(ss["brick_timestamp"])
        # ────────────────────────────────────────────────────────────────────

        m = pd.merge_asof(
            ts.sort_values("brick_timestamp"),
            ss.sort_values("brick_timestamp"),
            on="brick_timestamp",
            direction="backward",
        )
        rs_raw = m["stock_z"] - m["sector_z"].fillna(0)
        
        # Apply smoothing to ignore 1-day noise/shakeouts
        window = getattr(config, "RS_SMOOTHING_WINDOW", 1)
        if window > 1:
            df["relative_strength"] = rs_raw.rolling(window=window, min_periods=1).mean().values
        else:
            df["relative_strength"] = rs_raw.values
    else:
        df["relative_strength"] = stock_z.values

    df["consecutive_same_dir"]  = compute_consecutive_same_dir(df)
    df["brick_oscillation_rate"] = compute_brick_oscillation_rate(df)

    # ── Fix 1: Fractional Differentiation (with warmup to avoid early NaN) ──
    try:
        fd = FractionalDifferentiator()
        log_prices = np.log(df["brick_close"].clip(lower=1e-9))

        warmup = getattr(config, "FRACDIFF_WARMUP_BRICKS", 60)
        original_len = len(log_prices)

        if original_len < warmup:
            # Not enough bricks yet — pad start by repeating first value
            pad = pd.Series([log_prices.iloc[0]] * (warmup - original_len))
            log_prices_padded = pd.concat([pad, log_prices], ignore_index=True)
            fd_series_full = fd.transform(log_prices_padded, fracdiff_d)
            fd_series = fd_series_full.iloc[warmup - original_len:].reset_index(drop=True)
        else:
            fd_series = fd.transform(log_prices, fracdiff_d)

        # Fill any remaining NaN with forward-fill then 0
        fd_series = fd_series.ffill().fillna(0.0)
        df["fracdiff_price"] = fd_series.values
    except Exception:
        df["fracdiff_price"] = 0.0


    # ── Fix 4: Rolling Hurst Exponent + Regime Gate ──────────────────────────
    prices = df["brick_close"].values
    n = len(prices)
    hurst_vals = np.full(n, 0.5)
    for i in range(hurst_window, n):
        sub = pd.Series(prices[i - hurst_window: i])
        hurst_vals[i] = compute_hurst_exponent(sub, min_lag=2, max_lag=hurst_window // 2)
    df["hurst"] = hurst_vals
    df["is_trending_regime"] = (df["hurst"] > hurst_threshold).astype(int)

    # ── Phase 2: Institutional Alpha Factors ──────────────────────────────────
    df["vwap_zscore"]       = compute_vwap_zscore(
        df, window=getattr(config, "VWAP_WINDOW", 20)
    )
    df["vpt_acceleration"]  = compute_vpt_acceleration(
        df, diff_lag=getattr(config, "VPT_ACCEL_DIFF", 2)
    )
    df["squeeze_zscore"]    = compute_squeeze_zscore(
        df, window=getattr(config, "SQUEEZE_WINDOW", 20)
    )
    df["streak_exhaustion"] = compute_streak_exhaustion(
        df,
        onset=getattr(config, "STREAK_EXHAUSTION_ONSET", 8),
        scale=getattr(config, "STREAK_EXHAUSTION_SCALE", 0.5),
    )

    # ── Phase 3: Contextual Volatility Features ──────────────────────────────
    df["feature_tib_zscore"] = compute_tib_zscore(df, window=50)
    df["feature_vpb_roc"]    = compute_vpb_roc(df, window=20)
    regime_dummies = compute_market_regime_dummies(df)
    for col in regime_dummies.columns:
        df[col] = regime_dummies[col]

    # ── Placeholders ─────────────────────────────────────────────────────────
    df["whale_oi_score"]  = np.nan
    df["sentiment_score"] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PATCH 3: FEATURE SANITY CHECK — Live vs Historical Distribution Audit
# ═══════════════════════════════════════════════════════════════════════════

class FeatureSanityCheck:
    """
    Forensic diagnostic tool to catch the Sim2Real gap.

    The Problem (the invisible killer):
        The XGBoost model was trained on historical Renko bricks where
        features like Velocity, Hurst, and FracDiff had specific statistical
        distributions (e.g. Velocity ∈ [-0.5, 1.2] in backtest).

        In live trading, subtle differences in how bricks are constructed
        (tick resolution, heartbeat injections, first-brick duration=1s)
        can push these features OUT of the training distribution:
            - Velocity can spike to 3+ on the first brick (duration=1s vs avg=60s)
            - Hurst defaults to 0.5 for the first 60 bricks (not enough history)
            - FracDiff can be NaN if log_prices has inf (price=0 edge case)

        When features are out-of-distribution, XGBoost extrapolates wildly
        and Brain1 outputs prob=0.68 for every single brick — causing the
        exact churn pattern observed today (55 trades, 50 losses).

    The Fix:
        This class computes per-feature statistics from a historical parquet,
        then checks each live feature vector against those bounds.
        Any feature outside [mean ± 4σ] is flagged with a WARNING print.

    How to Activate (Forensic Mode — one-day diagnostic):
        1. Create one instance at the top of run_paper_trader().
        2. Call .fit() once using a historical feature parquet.
        3. Call .check(feature_dict, symbol, timestamp) after every inference.
        4. Read the WARNING lines in paper_trader.log to find which feature
           is causing the model to hallucinate.
        5. Disable after diagnosis (set ENABLED = False below).

    Example Integration in paper_trader.py:
        # --- At startup (once) ---
        from src.core.features import FeatureSanityCheck
        sanity = FeatureSanityCheck(enabled=True)
        sanity.fit_from_parquet("Finance", "NIACL")   # use a representative stock

        # --- After every inference (inside the main loop) ---
        feat_dict = latest[FEAT_COLS].fillna(0).to_dict()
        sanity.check(feat_dict, sym, now)
    """

    FEAT_COLS = config.FEATURE_COLS

    # How many σ from mean before we flag as "out of distribution"
    SIGMA_THRESHOLD = config.DRIFT_ACCURACY_THRESHOLD

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._means: dict[str, float] = {}
        self._stds:  dict[str, float] = {}
        self._mins:  dict[str, float] = {}
        self._maxs:  dict[str, float] = {}
        self._fitted = False
        self._check_count = 0
        self._flag_count = 0

    def fit_from_parquet(self, sector: str, symbol: str) -> bool:
        """
        Compute per-feature statistics from the historical feature parquet
        for a given symbol. Call this ONCE at startup.

        Args:
            sector:  e.g. "Finance"
            symbol:  e.g. "NIACL"

        Returns:
            True if successfully fitted, False if parquet missing.
        """
        feature_parquet = config.FEATURES_DIR / sector / f"{symbol}.parquet"
        if not feature_parquet.exists():
            print(f"[FeatureSanityCheck] WARNING: No feature parquet at {feature_parquet}")
            print(f"[FeatureSanityCheck] Run feature_engine.py first to generate training features.")
            return False

        try:
            df = pd.read_parquet(feature_parquet)
            available_cols = [c for c in self.FEAT_COLS if c in df.columns]

            for col in available_cols:
                series = df[col].dropna()
                self._means[col] = float(series.mean())
                self._stds[col]  = float(series.std())
                self._mins[col]  = float(series.min())
                self._maxs[col]  = float(series.max())

            self._fitted = True
            print(f"[FeatureSanityCheck] [OK] Fitted on {len(df)} rows from {symbol} ({sector})")
            print(f"[FeatureSanityCheck] Monitoring {len(available_cols)}/{len(self.FEAT_COLS)} features")
            print(f"[FeatureSanityCheck] Flagging threshold: mean +/- {self.SIGMA_THRESHOLD} stddev")
            return True

        except Exception as e:
            print(f"[FeatureSanityCheck] ERROR: Could not fit from parquet: {e}")
            return False

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        """Alternative: fit directly from a DataFrame (e.g. loaded externally)."""
        for col in self.FEAT_COLS:
            if col in df.columns:
                series = df[col].dropna()
                self._means[col] = float(series.mean())
                self._stds[col]  = float(series.std())
                self._mins[col]  = float(series.min())
                self._maxs[col]  = float(series.max())
        self._fitted = True

    def check(self,
              feat_dict: dict,
              symbol: str,
              timestamp,
              prob: float = -1.0) -> list[str]:
        """
        Compare a live feature vector against the training distribution.
        Prints WARNING lines for any feature that is out-of-distribution.

        Args:
            feat_dict:  Dict of {feature_name: value} from the live inference.
            symbol:     NSE symbol (for log context).
            timestamp:  Current datetime (for log context).
            prob:       Brain1 output probability (for context in the warning).

        Returns:
            List of flagged feature names (empty if all in-distribution).
        """
        if not self.enabled:
            return []

        if not self._fitted:
            print(f"[FeatureSanityCheck] NOT FITTED — call .fit_from_parquet() first.")
            return []

        self._check_count += 1
        flagged = []

        for col in self.FEAT_COLS:
            if col not in feat_dict:
                print(
                    f"[FeatureSanityCheck] [WARNING] MISSING FEATURE: {col} not in live feat_dict "
                    f"for {symbol} @ {timestamp} — model will receive NaN or 0!"
                )
                flagged.append(col)
                continue

            val = feat_dict[col]

            # Flag NaN/Inf
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                print(
                    f"[FeatureSanityCheck] [ERROR] NaN/Inf DETECTED: {col}={val} "
                    f"for {symbol} @ {timestamp} | Brain1_prob={prob:.3f}"
                )
                flagged.append(col)
                continue

            if col not in self._means:
                continue  # Not fitted for this column

            mean = self._means[col]
            std  = self._stds[col]

            if std < 1e-9:
                continue  # Near-constant feature, can't compute sigma

            z_score = abs(val - mean) / std

            if z_score > self.SIGMA_THRESHOLD:
                self._flag_count += 1
                direction = "HIGH" if val > mean else "LOW"
                print(
                    f"[FeatureSanityCheck] [WARNING] OUT-OF-DIST: {col}={val:.4f} is {z_score:.1f} stddev "
                    f"{direction} of training mean ({mean:.4f}±{std:.4f}). "
                    f"Training range=[{self._mins[col]:.4f}, {self._maxs[col]:.4f}]. "
                    f"Symbol={symbol} @ {timestamp} | Brain1_prob={prob:.3f}"
                )
                flagged.append(col)

        return flagged

    def summary(self) -> None:
        """Print a daily diagnostic summary. Call at EOD."""
        print("=" * 70)
        print(f"[FeatureSanityCheck] DAILY SUMMARY")
        print(f"  Total inference checks : {self._check_count}")
        print(f"  Total OOD flags raised : {self._flag_count}")
        if self._check_count > 0:
            rate = self._flag_count / self._check_count * 100
            print(f"  OOD flag rate          : {rate:.1f}%")
        if self._flag_count > 0:
            print(f"  [WARNING] High OOD rate means live features diverge from training.")
            print(f"    Check: brick construct logic, warmup window, or timestamp normalization.")
        print("=" * 70)

