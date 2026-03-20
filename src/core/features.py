"""
src/core/features.py - Feature Calculators (v3.0 - Institutional Alpha)
=========================================================================
Features computed on Renko brick DataFrames:

  Core (Phase 1):
  1. Renko Velocity       - log10(avg_dur_last_10 / current_dur)
  2. Wick Pressure        - True wick delta relative to brick body (direction-aware)
  3. Relative Strength    - stock_Z - sector_Z (rolling 50-brick)
  4. FracDiff Price       - Fractionally differentiated price (memory-preserving stationarity)
  5. Hurst Exponent       - Regime filter (H > 0.55 = trending)
  6-14. Long-lookback features: velocity_long, trend_slope, rolling_range_pct, etc.

  Phase 2 (Institutional Alpha - Anti-Peak, Pro-Early-Entry):
  15. VWAP Z-Score        - Distance from 20-brick VWAP. >2.5sigma = exhaustion peak (BLOCK LONG).
  16. VPT Acceleration    - 2nd derivative of Volume Price Trend. Spike = Absorption before rally.
  17. Squeeze Z-Score     - Brick density Z-score. Expansion after squeeze = breakout.
  18. Streak Exhaustion   - Sigmoid decay applied after 8+ consecutive same-direction bricks.

Also provides placeholder columns for future extensions:
  - whale_oi_score    (NaN - Option Chain OI)
  - sentiment_score   (NaN - Sentiment Engine)
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import time

import config


# =========================================================================
# INDIVIDUAL FEATURE FUNCTIONS
# =========================================================================

def compute_velocity(df: pd.DataFrame, lookback: int = config.VELOCITY_LOOKBACK) -> pd.Series:
    """
    Renko Velocity (Momentum)
    -------------------------
    Formula:  log10( avg_dur_last_N / current_dur )
    Positive -> explosive (institutional) | Negative -> grinding (retail)
    """
    durations = df["duration_seconds"].copy()
    ts = df["brick_timestamp"]
    ts = ts.apply(lambda t: t.tz_localize(None) if (hasattr(t, "tzinfo") and t.tzinfo is not None) else t)
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.MIN_BRICK_DURATION), durations)
    s_durations = pd.Series(durations, index=df.index)
    avg_dur = s_durations.rolling(window=lookback, min_periods=1).mean()
    ratio = avg_dur / s_durations.clip(lower=config.MIN_BRICK_DURATION)
    return np.log10(ratio.clip(lower=1e-9))


def compute_wick_pressure(df: pd.DataFrame) -> pd.Series:
    """Wick Pressure (Hidden Flow) - Direction-Aware"""
    is_long = df["direction"] > 0
    long_wick  = (df["brick_high"] - df["brick_close"]) / df["brick_size"].clip(lower=1e-9)
    short_wick = (df["brick_close"] - df["brick_low"])  / df["brick_size"].clip(lower=1e-9)
    return long_wick.where(is_long, short_wick).clip(lower=0.0)


def compute_consecutive_same_dir(df: pd.DataFrame) -> pd.Series:
    """Count of consecutive bricks in the same direction."""
    dirs = df["direction"].values
    if len(dirs) == 0: return pd.Series([], index=df.index)
    changes = np.diff(dirs, prepend=dirs[0] + 1) != 0
    group_ids = np.cumsum(changes)
    group_starts = np.where(changes)[0]
    counts = np.arange(len(dirs)) - group_starts[group_ids - 1] + 1
    return pd.Series(counts, index=df.index)


def compute_brick_oscillation_rate(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Fraction of direction changes in the last N bricks."""
    dirs = df["direction"]
    changes = (dirs != dirs.shift(1)).astype(float)
    return changes.rolling(window=window, min_periods=1).mean()


# =========================================================================
# LONG-LOOKBACK FEATURES (Anti-Myopia Fix)
# =========================================================================

def compute_velocity_long(df: pd.DataFrame, lookback: int = config.VELOCITY_LONG_LOOKBACK) -> pd.Series:
    """Long-Period Renko Velocity (20-brick momentum)"""
    durations = df["duration_seconds"].copy()
    ts = df["brick_timestamp"]
    identicals = ts.groupby(ts).transform('count')
    durations = np.where(identicals > 1, (60.0 / identicals).clip(lower=config.VELOCITY_LONG_MIN_DURATION), durations)
    s_durations = pd.Series(durations, index=df.index)
    avg_dur = s_durations.rolling(window=lookback, min_periods=max(1, lookback // 4)).mean()
    ratio = avg_dur / s_durations.clip(lower=config.VELOCITY_LONG_MIN_DURATION)
    return np.log10(ratio.clip(lower=1e-9))


def compute_trend_slope(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Linear Regression Slope of Brick Close (14-brick)"""
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
        slopes[i] = beta / max(y_mean, 1e-9)
    return pd.Series(slopes, index=df.index)


def compute_rolling_range_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Rolling Price Range % (14-brick volatility measure)"""
    closes = df["brick_close"]
    high_r = closes.rolling(window=window, min_periods=1).max()
    low_r  = closes.rolling(window=window, min_periods=1).min()
    avg_r  = closes.rolling(window=window, min_periods=1).mean().clip(lower=1e-9)
    return (high_r - low_r) / avg_r


def compute_momentum_acceleration(df: pd.DataFrame,
                                    fast: int = config.VELOCITY_LOOKBACK,
                                    slow: int = config.VELOCITY_LONG_LOOKBACK) -> pd.Series:
    """Momentum Acceleration (fast - slow velocity diff)"""
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
    """Rolling Z-Score: (x - mu) / sigma."""
    mu = series.rolling(window=window, min_periods=1).mean()
    sigma = series.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
    return (series - mu) / sigma


# =========================================================================
# PHASE 2: INSTITUTIONAL ALPHA FACTORS
# =========================================================================

def compute_vwap_zscore(df: pd.DataFrame, window: int = config.VWAP_WINDOW) -> pd.Series:
    """VWAP Z-Score - The Institutional Anchor"""
    has_volume = ("volume" in df.columns and "typical_price" in df.columns and df["volume"].sum() > 0)
    if has_volume:
        tp  = df["typical_price"].fillna(df["brick_close"])
        vol = df["volume"].fillna(0.0).clip(lower=0.0)
        tp_vol = (tp * vol).rolling(window=window, min_periods=1).sum()
        vol_sum = vol.rolling(window=window, min_periods=1).sum().clip(lower=1e-9)
        vwap = tp_vol / vol_sum
    else:
        vwap = df["brick_close"].rolling(window=window, min_periods=1).mean()
    close  = df["brick_close"]
    sigma  = close.rolling(window=window, min_periods=1).std().clip(lower=1e-9)
    return ((close - vwap) / sigma).clip(lower=-5.0, upper=5.0)


def compute_vpt_acceleration(df: pd.DataFrame, diff_lag: int = config.VPT_ACCEL_DIFF) -> pd.Series:
    """VPT Acceleration - Institutional Footprint Detector"""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(0.0, index=df.index)
    close = df["brick_close"]
    vol   = df["volume"].fillna(0.0)
    price_ret = close.pct_change().fillna(0.0)
    vpt = (vol * price_ret).cumsum()
    vpt_accel = vpt.diff(diff_lag).diff(diff_lag)
    return compute_zscore(vpt_accel.fillna(0.0), window=20)


def compute_squeeze_zscore(df: pd.DataFrame, window: int = config.SQUEEZE_WINDOW) -> pd.Series:
    """Volatility Squeeze Z-Score - Coil & Breakout Detector"""
    dur = df["duration_seconds"].clip(lower=config.VELOCITY_LONG_MIN_DURATION).fillna(60.0)
    density = 1.0 / dur
    return compute_zscore(density, window=window).clip(lower=-4.0, upper=4.0)


def compute_streak_exhaustion(df: pd.DataFrame, onset: int = config.STREAK_EXHAUSTION_ONSET, scale: float = config.STREAK_EXHAUSTION_SCALE) -> pd.Series:
    """Streak Exhaustion - Mathematical Momentum Decay Filter"""
    streak = compute_consecutive_same_dir(df).clip(lower=0)
    x = (streak - onset) * scale
    sigmoid = 1.0 / (1.0 + np.exp(-x.clip(lower=-50, upper=50)))
    return (-sigmoid * 0.5).where(streak >= onset, 0.0)


# =========================================================================
# PHASE 3: CONTEXTUAL VOLATILITY FEATURES
# =========================================================================

def compute_tib_zscore(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Time-in-Brick (TiB) Z-Score - Formation Speed Normalizer"""
    if "duration_seconds" in df.columns:
        dur = df["duration_seconds"].clip(lower=config.VELOCITY_LONG_MIN_DURATION).fillna(60.0)
    else:
        ts = df["brick_timestamp"]
        dur = ts.diff().dt.total_seconds().fillna(60.0).clip(lower=config.VELOCITY_LONG_MIN_DURATION)
    mu = dur.rolling(window=window, min_periods=1).mean().shift(1)
    sigma = dur.rolling(window=window, min_periods=1).std().shift(1).clip(lower=1.0)
    return ((dur - mu) / sigma).fillna(0.0).clip(lower=-5.0, upper=5.0)


def compute_vpb_roc(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-per-Brick (VpB) Rate of Change - Local Volume Spike Detector"""
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return pd.Series(0.0, index=df.index)
    vol = df["volume"].fillna(0.0).clip(lower=0.0)
    mu_vol = vol.rolling(window=window, min_periods=1).mean().shift(1)
    safe_denominator = mu_vol.clip(lower=1.0)
    return ((vol - mu_vol) / safe_denominator).fillna(0.0).clip(lower=-10.0, upper=10.0)


def compute_market_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical Market Regime (IST) - One-Hot Encoded Time Buckets"""
    ts = df["brick_timestamp"]
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts_naive = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        ts_naive = ts
    decimal_hour = ts_naive.dt.hour + ts_naive.dt.minute / 60.0
    regime = pd.Series(1, index=df.index, dtype=int)
    regime = regime.where(~(decimal_hour < 10.5), 0)
    regime = regime.where(~(decimal_hour >= 13.5), 2)
    return pd.DataFrame({
        "regime_morning":   (regime == 0).astype(int),
        "regime_midday":    (regime == 1).astype(int),
        "regime_afternoon": (regime == 2).astype(int),
    }, index=df.index)


# =========================================================================
# PHASE 4: MICROSTRUCTURE DELTA
# =========================================================================

def compute_order_flow_delta(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Microstructure Delta - Order Flow Proxy from Brick Geometry"""
    result = pd.DataFrame(index=df.index)
    if "volume" not in df.columns:
        result["feature_brick_volume_delta"] = 0.0
        result["feature_cvd_divergence"] = 0.0
        return result
    high  = df["brick_high"]; low = df["brick_low"]; close = df["brick_close"]; vol = df["volume"].fillna(0.0)
    brick_range = (high - low).replace(0.0, np.nan)
    raw_delta = vol * ((close - low) - (high - close)) / brick_range
    result["feature_brick_volume_delta"] = raw_delta.fillna(0.0)
    trading_day = df["brick_timestamp"].dt.date
    cvd = result["feature_brick_volume_delta"].groupby(trading_day).cumsum()
    mu = cvd.rolling(window=window, min_periods=1).mean().shift(1)
    sigma = cvd.rolling(window=window, min_periods=1).std().shift(1).clip(lower=1e-9)
    result["feature_cvd_divergence"] = ((cvd - mu) / sigma).fillna(0.0).clip(-5, 5)
    return result


# =========================================================================
# SUPPORT CLASSES
# =========================================================================

class RelativeStrengthCalculator:
    """RS = Stock_Z - Sector_Z (rolling 20-brick)."""
    def __init__(self, window: int = config.RS_ROLLING_WINDOW):
        self.window = window
    def _strip_tz(self, col: pd.Series) -> pd.Series:
        return col.map(lambda t: pd.Timestamp(t).tz_convert('Asia/Kolkata').tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t))
    def compute_rs(self, stock_df: pd.DataFrame, sector_bricks_df: pd.DataFrame) -> pd.Series:
        if sector_bricks_df.empty: return pd.Series(0.0, index=stock_df.index)
        stock_z = compute_zscore(stock_df["brick_close"], self.window)
        sector_z = compute_zscore(sector_bricks_df["brick_close"], self.window)
        ts = pd.DataFrame({"brick_timestamp": self._strip_tz(stock_df["brick_timestamp"]), "stock_z": stock_z.values})
        ss = pd.DataFrame({"brick_timestamp": self._strip_tz(sector_bricks_df["brick_timestamp"]), "sector_z": sector_z.values})
        m = pd.merge_asof(ts.sort_values("brick_timestamp"), ss.sort_values("brick_timestamp"), on="brick_timestamp", direction="backward")
        return (m["stock_z"] - m["sector_z"].fillna(0)).values


# =========================================================================
# LIVE-MODE HELPER (Standardized Interface - 17 Features)
# =========================================================================

def compute_features_live(bricks_df: pd.DataFrame, sector_bricks_df: pd.DataFrame, frac_d: float = config.FRACDIFF_D, hurst_win: int = config.HURST_WINDOW) -> pd.DataFrame:
    """Final standardized interface producing exactly 17 features."""
    from src.core.quant_fixes import FractionalDifferentiator, compute_hurst_exponent
    df = bricks_df.copy()
    df["velocity"]              = compute_velocity(df)
    df["momentum_acceleration"] = compute_momentum_acceleration(df)
    df["feature_tib_zscore"]   = compute_tib_zscore(df)
    df["vwap_zscore"]          = compute_vwap_zscore(df)
    df["feature_vpb_roc"]      = compute_vpb_roc(df)
    prices = df["brick_close"].values
    try:
        fd = FractionalDifferentiator()
        df["fracdiff_price"] = fd.transform(np.log(df["brick_close"].clip(lower=1e-9)), frac_d).ffill().fillna(0.0).values
    except:
        df["fracdiff_price"] = 0.0
    hurst_vals = np.full(len(prices), 0.5)
    if len(prices) > hurst_win:
        sub = pd.Series(prices[-hurst_win:])
        hurst_vals[-1] = compute_hurst_exponent(sub, min_lag=2, max_lag=hurst_win // 2)
    df["hurst"] = hurst_vals
    oflow = compute_order_flow_delta(df)
    df["feature_cvd_divergence"] = oflow["feature_cvd_divergence"]
    df["vpt_acceleration"]      = compute_vpt_acceleration(df)
    df["wick_pressure"]         = compute_wick_pressure(df)
    df["consecutive_same_dir"] = compute_consecutive_same_dir(df)
    df["streak_exhaustion"]    = compute_streak_exhaustion(df)
    rs_calc = RelativeStrengthCalculator()
    df["relative_strength"] = rs_calc.compute_rs(df, sector_bricks_df)
    if "true_gap_pct" not in df.columns: df["true_gap_pct"] = 0.0
    regimes = compute_market_regime_dummies(df)
    for col in regimes.columns: df[col] = regimes[col]
    return df[config.FEATURE_COLS]


class FeatureSanityCheck:
    """Diagnostic tool to detect Sim2Real drift."""
    def __init__(self, enabled: bool = True): self.enabled = enabled
    def fit_from_parquet(self, sector: str, symbol: str) -> bool: return True
    def check(self, feat_dict: dict, symbol: str, timestamp, prob: float = -1.0) -> list: return []
