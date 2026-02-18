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

import config


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_velocity(df: pd.DataFrame, lookback: int = config.VELOCITY_LOOKBACK) -> pd.Series:
    """
    Renko Velocity (Momentum)
    ─────────────────────────
    Formula:  log₁₀( avg_dur_last_N / current_dur )
    Positive → explosive (institutional) · Negative → grinding (retail)
    """
    avg_dur = df["duration_seconds"].rolling(window=lookback, min_periods=1).mean()
    ratio = avg_dur / df["duration_seconds"].clip(lower=1)
    return np.log10(ratio.clip(lower=1e-9))


def compute_wick_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Wick Pressure (Hidden Flow)
    ────────────────────────────
    Formula:  (brick_high − brick_close) / brick_size
    >0.6 → rejection / trap · Low → clean close
    """
    return (df["brick_high"] - df["brick_close"]).abs() / df["brick_size"].clip(lower=1e-9)


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
    Positive → Leader · Negative → Laggard
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

        sdf = pd.concat(frames, ignore_index=True).sort_values("brick_timestamp").reset_index(drop=True)
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
            temp.sort_values("brick_timestamp"),
            sector_df.sort_values("brick_timestamp"),
            on="brick_timestamp",
            direction="backward",
        )
        return (merged["stock_zscore"] - merged["sector_zscore"].fillna(0)).values


# ═══════════════════════════════════════════════════════════════════════════
# PLACEHOLDER COLUMNS (Future Extensions)
# ═══════════════════════════════════════════════════════════════════════════

def add_whale_oi_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """[FUTURE] Whale Tracker — Option Chain OI data."""
    df["whale_oi_score"] = np.nan
    return df


def add_sentiment_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    """[FUTURE] Sentiment Engine — News/Social sentiment score."""
    df["sentiment_score"] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════════
# LIVE-MODE HELPER (used by src/live/engine.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_features_live(
    bricks_df: pd.DataFrame,
    sector_bricks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute features on a live (incrementally growing) brick DataFrame.
    """
    df = bricks_df.copy()
    df["velocity"] = compute_velocity(df)
    df["wick_pressure"] = compute_wick_pressure(df)

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

    df["whale_oi_score"] = np.nan
    df["sentiment_score"] = np.nan
    return df
