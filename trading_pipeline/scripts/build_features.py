"""
src/data/feature_engine.py - Phase 2: Batch Feature Computation
================================================================
Loads Renko Parquet files -> computes Velocity, Wick Pressure,
Relative Strength -> saves enriched Parquet to storage/features/.

Run:  python -m src.data.feature_engine
"""

import sys
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import config
from trading_core.core.features import (
    _normalize_ts,
    compute_velocity,
    compute_wick_pressure,
    compute_consecutive_same_dir,
    compute_brick_oscillation_rate,
    RelativeStrengthCalculator,
    # Anti-Myopia: Long-lookback features
    compute_velocity_long,
    compute_trend_slope,
    compute_rolling_range_pct,
    compute_momentum_acceleration,
    # Phase 2: Institutional Alpha Factors
    compute_vwap_zscore,
    compute_vpt_acceleration,
    compute_squeeze_zscore,
    compute_streak_exhaustion,
    # Phase 3: Contextual Volatility Features
    compute_tib_zscore,
    compute_vpb_roc,
    compute_market_regime_dummies,
    # Phase 4: Order Flow Proxy
    compute_order_flow_delta,
)
from trading_core.core.physics.quant_fixes import apply_all_quant_fixes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "feature_engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def enrich_stock(symbol: str, sector: str) -> str:
    """
    Enriches a single stock with features. 
    Supports INCREMENTAL updates if the feature file already exists.
    """
    out_dir = config.FEATURES_DIR / sector
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    
    # --- 1. Load Existing Features (if any) ---
    existing_df = pd.DataFrame()
    last_ts = None
    incremental_enabled = hasattr(config, "FEATURE_INCREMENTAL_ENABLED") and config.FEATURE_INCREMENTAL_ENABLED
    if out_path.exists() and incremental_enabled:
        try:
            existing_df = pd.read_parquet(out_path)
            if not existing_df.empty:
                # The Fix: Force existing_df to Naive IST (Cleanse any legacy aware timestamps)
                for col in existing_df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
                    existing_df[col] = config.to_naive_ist(existing_df[col])
                last_ts = existing_df["brick_timestamp"].max()
        except Exception as e:
            logger.warning(f"Could not read existing features for {symbol}: {e}")

    # --- 2. Load Raw Bricks ---
    stock_dir = config.DATA_DIR / sector / symbol
    if not stock_dir.exists():
        return f"SKIP  {symbol} - raw data dir not found"

    parquets = sorted(stock_dir.glob("*.parquet"))
    if not parquets:
        return f"SKIP  {symbol} - no raw parquet files"

    all_raw_dfs = []
    for f in parquets:
        try:
            chunk = pd.read_parquet(f)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            continue

        # Fix 2: Enforce Naive IST Timestamps (Kolkata Force)
        if "brick_timestamp" in chunk.columns:
            chunk["brick_timestamp"] = _normalize_ts(chunk["brick_timestamp"])
        
        # Incremental filter
        if last_ts is not None:
            # The Fix: Ensure both sides are Naive IST for safe comparison
            chunk["brick_timestamp"] = config.to_naive_ist(chunk["brick_timestamp"])
            chunk = chunk[chunk["brick_timestamp"] > last_ts]
        
        if not chunk.empty:
            all_raw_dfs.append(chunk)

    if not all_raw_dfs:
        return f"SKIP  {symbol} - no NEW bricks since {last_ts}"

    new_raw_df = pd.concat(all_raw_dfs, ignore_index=True).sort_values("brick_timestamp", kind="mergesort")
    
    # FIX 1: Dynamic Lookback Context (Hot-Starting Rolling Windows)
    required_lookback = max(
        getattr(config, "VWAP_WINDOW", 20),
        getattr(config, "VELOCITY_LONG_LOOKBACK", 20),
        getattr(config, "RS_ROLLING_WINDOW", 50),
        getattr(config, "HURST_WINDOW", 100),
        getattr(config, "SQUEEZE_WINDOW", 20)
    )
    lookback_context = required_lookback * 2 # Buffer for stability
    if existing_df.empty:
        # Full re-run
        compute_df = new_raw_df
        context_df = pd.DataFrame()
    else:
        # FIX 10: Incremental VWAP Poisoning (Index-Safe Date Anchor)
        first_new_date = new_raw_df["brick_timestamp"].dt.date.iloc[0]
        day_mask = (existing_df["brick_timestamp"].dt.date == first_new_date).values
        day_start_iloc = int(np.argmax(day_mask)) if day_mask.any() else len(existing_df)
        safe_start_iloc = max(0, day_start_iloc - lookback_context)
        context_df = existing_df.iloc[safe_start_iloc:].copy()
        
        # Drop feature columns from context so they are re-calculated fresh with the new data
        feature_cols = [
            "velocity", "wick_pressure", "relative_strength", "consecutive_same_dir",
            "brick_oscillation_rate", "fracdiff_price", "hurst", "is_trending_regime",
            "whale_oi_score", "sentiment_score",
            # Anti-Myopia: Long-lookback features
            "velocity_long", "trend_slope", "rolling_range_pct", "momentum_acceleration",
            # Phase 2: Institutional Alpha Factors
            "vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion",
            # Phase 3: Contextual Volatility Features
            "feature_tib_zscore", "feature_vpb_roc",
            "regime_morning", "regime_midday", "regime_afternoon",
            # Phase 4: Order Flow Proxy
            "feature_brick_volume_delta", "feature_cvd_divergence",
        ]
        context_df = context_df.drop(columns=[c for c in feature_cols if c in context_df.columns])
        
        compute_df = pd.concat([context_df, new_raw_df], ignore_index=True)

    if len(compute_df) < 2:
        return f"SKIP  {symbol} - too few bricks for math"

    # --- 4. Calculate Features ---
    # Load Sector Data for RS
    sector_dir = config.DATA_DIR / sector / sector
    sector_bricks_df = pd.DataFrame()
    if sector_dir.exists():
        sector_files = sorted(sector_dir.glob("*.parquet"))
        if sector_files:
            try:
                # Robust loading: individual read + normalization to prevent mixed-tz concat
                sector_frames = []
                for f in sector_files:
                    s_chunk = pd.read_parquet(f)
                    for col in s_chunk.select_dtypes(include=["datetime64", "datetimetz"]).columns:
                        s_chunk[col] = config.to_naive_ist(s_chunk[col])
                    sector_frames.append(s_chunk)
                sector_bricks_df = pd.concat(sector_frames, ignore_index=True)
            except Exception as e:
                logger.error(f"Error loading sector data for {sector}: {e}")
    # Fix 4: Parallel Multi-threading Safety (Local Init)
    from trading_core.core.features import RelativeStrengthCalculator
    rs_calc = RelativeStrengthCalculator()

    compute_df["velocity"] = compute_velocity(compute_df)
    compute_df["wick_pressure"] = compute_wick_pressure(compute_df)
    
    # --- Fix: Implement Missing Relative Strength ---
    try:
        compute_df["relative_strength"] = rs_calc.compute_rs(compute_df, sector_bricks_df)
    except Exception as e:
        logger.warning(f"RS calculation failed for {symbol}: {e}")
        compute_df["relative_strength"] = 0.0
    
    compute_df["consecutive_same_dir"] = compute_consecutive_same_dir(compute_df)
    compute_df["brick_oscillation_rate"] = compute_brick_oscillation_rate(compute_df)
    compute_df["velocity_long"] = compute_velocity_long(compute_df)
    compute_df["trend_slope"] = compute_trend_slope(compute_df)
    compute_df["rolling_range_pct"] = compute_rolling_range_pct(compute_df)
    compute_df["momentum_acceleration"] = compute_momentum_acceleration(compute_df)

    compute_df["vwap_zscore"] = compute_vwap_zscore(compute_df, window=config.VWAP_WINDOW)
    compute_df["vpt_acceleration"] = compute_vpt_acceleration(compute_df, diff_lag=config.VPT_ACCEL_LAG)
    compute_df["squeeze_zscore"] = compute_squeeze_zscore(compute_df, window=config.SQUEEZE_WINDOW)
    compute_df["streak_exhaustion"] = compute_streak_exhaustion(
        compute_df, onset=config.STREAK_EXHAUSTION_ONSET, scale=config.STREAK_EXHAUSTION_SCALE
    )

    compute_df["feature_tib_zscore"] = compute_tib_zscore(compute_df, window=50)
    compute_df["feature_vpb_roc"]    = compute_vpb_roc(compute_df, window=20)
    regime_dummies = compute_market_regime_dummies(compute_df)
    for col in regime_dummies.columns:
        compute_df[col] = regime_dummies[col]

    oflow = compute_order_flow_delta(compute_df, window=20)
    compute_df["feature_brick_volume_delta"] = oflow["feature_brick_volume_delta"]
    compute_df["feature_cvd_divergence"]     = oflow["feature_cvd_divergence"]

    compute_df["whale_oi_score"] = 0.0
    compute_df["sentiment_score"] = 0.0
    compute_df["true_gap_pct"] = 0.0 # Placeholder for Renko

    try:
        compute_df = apply_all_quant_fixes(compute_df, fracdiff_d=config.FRACDIFF_D, hurst_window=config.HURST_WINDOW)
    except Exception as e:
        logger.warning(f"quant_fixes skipped for {symbol}: {e}")

    # --- 5. Merge and Save ---
    if existing_df.empty:
        final_df = compute_df
    else:
        new_features_df = compute_df.iloc[len(context_df):].copy()
        final_df = pd.concat([existing_df, new_features_df], ignore_index=True)

    # FIX 3: Symmetry Check (Boundary Validation)
    if not existing_df.empty and "vwap_zscore" in compute_df.columns:
        b_idx = len(context_df)
        v_old = compute_df["vwap_zscore"].iloc[b_idx - 1] if b_idx > 0 else 0
        v_new = compute_df["vwap_zscore"].iloc[b_idx]
        logger.info(f"Boundary ({symbol}): vwap_zscore OLD={v_old:.4f} | NEW={v_new:.4f}")

    final_df.to_parquet(out_path, engine="pyarrow", index=False)
    return f"OK    {symbol} -> +{len(new_raw_df)} new bricks -> {out_path.name}"


def run_feature_engine():
    logger.info("=" * 70)
    logger.info("FEATURE ENGINE - Starting (Parallel + Incremental)")
    logger.info("=" * 70)

    universe_path = config.UNIVERSE_CSV
    if not universe_path.exists():
        logger.error(f"Universe file not found at: {universe_path}")
        return

    universe = pd.read_csv(universe_path)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true", "1", "yes"])
    stocks = universe[~universe["is_index"]]

    ok = skip = fail = 0
    num_workers = max(1, cpu_count() - 1)
    if hasattr(config, "FEATURE_PARALLEL_WORKERS") and config.FEATURE_PARALLEL_WORKERS != -1:
        num_workers = config.FEATURE_PARALLEL_WORKERS
        
    logger.info(f"Using {num_workers} CPU cores for parallel processing...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _, row in stocks.iterrows():
            futures.append(executor.submit(enrich_stock, row["symbol"], row["sector"]))
            
        for future in futures:
            try:
                r = future.result()
                logger.info(r)
                ok += 1 if r.startswith("OK") else 0
                skip += 1 if r.startswith("SKIP") else 0
            except Exception as e:
                logger.error(f"Worker FAIL: {e}")
                fail += 1

    logger.info(f"DONE - OK: {ok}  Skip: {skip}  Fail: {fail}")


if __name__ == "__main__":
    run_feature_engine()

