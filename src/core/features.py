"""
src/core/features.py — Feature Calculators
=============================================
Three institutional-grade features computed on Renko brick DataFrames:

  1. Renko Velocity  — log₁₀(avg_dur_last_10 / current_dur)
  2. Wick Pressure   — (brick_high − brick_close) / brick_size
  3. Relative Strength — stock_Z − sector_Z (rolling 50-brick)

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
    identicals = ts.groupby(ts).transform('count')
    
    # For identical timestamps, artificially space their duration.
    # Instead of them all being 1 second, assume they took equal fractions of 60 seconds.
    # We clip to at least 1 second to avoid div by zero.
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=1), durations)
    
    s_durations = pd.Series(durations, index=df.index)

    avg_dur = s_durations.rolling(window=lookback, min_periods=1).mean()
    ratio = avg_dur / s_durations.clip(lower=1)
    return np.log10(ratio.clip(lower=1e-9))


def compute_wick_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Wick Pressure (Hidden Flow)
    ────────────────────────────
    Formula:  (brick_high − brick_close) / brick_size
    >0.6 -> rejection / trap · Low -> clean close
    """
    return (df["brick_high"] - df["brick_close"]).abs() / df["brick_size"].clip(lower=1e-9)


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

def compute_velocity_long(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
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
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=1), durations)
    s_durations = pd.Series(durations, index=df.index)
    avg_dur = s_durations.rolling(window=lookback, min_periods=max(1, lookback // 4)).mean()
    ratio = avg_dur / s_durations.clip(lower=1)
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
                                   fast: int = 5,
                                   slow: int = 14) -> pd.Series:
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
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=1), durations)
    s_dur = pd.Series(durations, index=df.index).clip(lower=1)

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

        merged = pd.merge_asof(
            temp.sort_values("brick_timestamp", kind="mergesort"),
            sector_df.sort_values("brick_timestamp", kind="mergesort"),
            on="brick_timestamp",
            direction="backward",
        )
        return (merged["stock_zscore"] - merged["sector_zscore"].fillna(0)).values


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
    fracdiff_d: float = 0.4,
    hurst_window: int = 60,
    hurst_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Compute features on a live (incrementally growing) brick DataFrame.
    Produces the same 14 features the model was trained on:
      velocity, wick_pressure, relative_strength, brick_size,
      duration_seconds, consecutive_same_dir, brick_oscillation_rate,
      fracdiff_price, hurst, is_trending_regime, velocity_long,
      trend_slope, rolling_range_pct, momentum_acceleration
    """
    # Lazy import to avoid circular dependencies
    from src.core.quant_fixes import (
        FractionalDifferentiator,
        compute_hurst_exponent,
    )

    df = bricks_df.copy()
    df["velocity"] = compute_velocity(df)
    df["velocity_long"] = compute_velocity_long(df)
    df["wick_pressure"] = compute_wick_pressure(df)
    df["trend_slope"] = compute_trend_slope(df)
    df["rolling_range_pct"] = compute_rolling_range_pct(df)
    df["momentum_acceleration"] = compute_momentum_acceleration(df)

    stock_z = compute_zscore(df["brick_close"], config.RS_ROLLING_WINDOW)
    if not sector_bricks_df.empty:
        sector_z = compute_zscore(sector_bricks_df["brick_close"], config.RS_ROLLING_WINDOW)
        ts = pd.DataFrame({"brick_timestamp": df["brick_timestamp"], "stock_z": stock_z.values})
        ss = pd.DataFrame({"brick_timestamp": sector_bricks_df["brick_timestamp"], "sector_z": sector_z.values})
        m = pd.merge_asof(ts.sort_values("brick_timestamp"), ss.sort_values("brick_timestamp"),
                          on="brick_timestamp", direction="backward")
        df["relative_strength"] = (m["stock_z"] - m["sector_z"].fillna(0)).values
    else:
        df["relative_strength"] = stock_z.values

    df["consecutive_same_dir"] = compute_consecutive_same_dir(df)
    df["brick_oscillation_rate"] = compute_brick_oscillation_rate(df)

    # ── Fix 1: Fractional Differentiation (matches feature_engine.py) ────
    try:
        fd = FractionalDifferentiator()
        log_prices = np.log(df["brick_close"].clip(lower=1e-9))
        fd_series = fd.transform(log_prices, fracdiff_d)
        df["fracdiff_price"] = fd_series.values
    except Exception:
        df["fracdiff_price"] = 0.0

    # ── Fix 4: Rolling Hurst Exponent + Regime Gate ───────────────────────
    prices = df["brick_close"].values
    n = len(prices)
    hurst_vals = np.full(n, 0.5)  # default: random walk
    for i in range(hurst_window, n):
        sub = pd.Series(prices[i - hurst_window: i])
        hurst_vals[i] = compute_hurst_exponent(sub, min_lag=2, max_lag=hurst_window // 2)
    df["hurst"] = hurst_vals
    df["is_trending_regime"] = (df["hurst"] > hurst_threshold).astype(int)

    df["whale_oi_score"] = np.nan
    df["sentiment_score"] = np.nan
    return df
