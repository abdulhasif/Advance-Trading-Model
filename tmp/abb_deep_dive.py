import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

import config
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.core.quant_fixes import IsotonicCalibrationWrapper

def run_deep_dive():
    tick_file = Path("storage/data/raw_ticks/raw_ticks_dump_2026-03-05.csv")
    if not tick_file.exists():
        print(f"Tick file not found: {tick_file}")
        return

    print("Loading ticks...")
    df_all = pd.read_csv(tick_file)
    df_abb = df_all[df_all["symbol"] == "ABB"].copy()
    if df_abb.empty:
        print("No ABB ticks found.")
        return
    
    # Sort by timestamp
    df_abb["timestamp"] = pd.to_datetime(df_abb["timestamp"]).dt.tz_localize(None)
    df_abb = df_abb.sort_values("timestamp")
    
    # Sector ticks for RS (NIFTY INFRA)
    df_infra = df_all[df_all["symbol"] == "NIFTY INFRA"].copy()
    df_infra["timestamp"] = pd.to_datetime(df_infra["timestamp"]).dt.tz_localize(None)
    df_infra = df_infra.sort_values("timestamp")

    print(f"Processing {len(df_abb)} ABB ticks...")

    # Load Models
    print("Loading models...")
    b1_long = IsotonicCalibrationWrapper.load(config.BRAIN1_CALIBRATED_LONG_PATH)
    b1_short = IsotonicCalibrationWrapper.load(config.BRAIN1_CALIBRATED_SHORT_PATH)
    
    # Brick Size (0.15% of first price or fixed)
    # NATR logic: 0.15% of prev close. Let's assume ~8500 price
    first_price = df_abb["ltp"].iloc[0]
    brick_size = first_price * 0.0015 
    print(f"Using brick size: {brick_size:.2f} (0.15% of {first_price:.2f})")

    # Setup Renko States
    st_abb = LiveRenkoState("ABB", "Infrastructure", brick_size)
    st_infra = LiveRenkoState("NIFTY INFRA", "Infrastructure", df_infra["ltp"].iloc[0] * 0.0015 if not df_infra.empty else 1.0)

    bricks = []
    infra_bricks = []
    signals = []
    
    # Process Infra first to have history for RS
    for _, row in df_infra.iterrows():
        new = st_infra.process_tick(row["ltp"], row["ltp"], row["ltp"], row["timestamp"])
        for b in new: infra_bricks.append(b)
        
    infra_df = pd.DataFrame(infra_bricks)

    # Process ABB
    for _, row in df_abb.iterrows():
        new = st_abb.process_tick(row["ltp"], row["ltp"], row["ltp"], row["timestamp"])
        for b in new:
            bricks.append(b)
            # Compute features for inference
            if len(bricks) < 5: continue
            
            abb_df = pd.DataFrame(bricks)
            # Slice infra bricks up to now
            now = b["brick_timestamp"]
            current_infra = infra_df[infra_df["brick_timestamp"] <= now]
            
            feat_df = compute_features_live(abb_df, current_infra)
            latest = feat_df.iloc[-1]
            
            # Predict
            exact_cols = [
                "velocity", "wick_pressure", "relative_strength", "brick_size",
                "duration_seconds", "consecutive_same_dir", "brick_oscillation_rate",
                "fracdiff_price", "hurst", "is_trending_regime", "velocity_long",
                "trend_slope", "rolling_range_pct", "momentum_acceleration",
                "vwap_zscore", "vpt_acceleration", "squeeze_zscore", "streak_exhaustion"
            ]
            X = pd.DataFrame([latest[exact_cols].fillna(0)])
            
            pl = float(b1_long.predict_proba(X)[0][1])
            ps = float(b1_short.predict_proba(X)[0][1])
            
            signals.append({
                "timestamp": now,
                "price": b["brick_close"],
                "p_long": pl,
                "p_short": ps,
                "direction": b["direction"]
            })

    if not signals:
        print("No signals generated (not enough bricks).")
        return

    sig_df = pd.DataFrame(signals)
    
    # Peak stats and feature breakdown
    print(f"\nPeak P(LONG):  {sig_df['p_long'].max():.4f}")
    print(f"Peak P(SHORT): {sig_df['p_short'].max():.4f}")

    # Top 5 peak LONG moments
    print("\n" + "="*50)
    print("TOP 5 PEAK LONG MOMENTS (DIAGNOSTIC)")
    print("="*50)
    top_longs = sig_df.sort_values("p_long", ascending=False).head(5)
    for idx, row in top_longs.iterrows():
        # Find the features for this brick
        now = row["timestamp"]
        # We need to re-run or store features to get more detail
        # For simplicity in this script, let's just find the max one and run a detailed version
        pass
    
    # Let's just find the absolute max and print its features
    max_idx = sig_df["p_long"].idxmax()
    max_row = sig_df.loc[max_idx]
    print(f"Timestamp: {max_row['timestamp']} | P(LONG): {max_row['p_long']:.4f} | Price: {max_row['price']}")
    
    # Re-run for the max moment to get features
    abb_df_full = pd.DataFrame(bricks)
    # The brick index in 'bricks' matches the 'signals' index (if we started adding at the same time)
    # Actually, signals list is slightly shorter due to the len(bricks) < 5 check.
    # index in signals 'i' corresponds to index 'i + 5' in bricks.
    brick_idx = max_idx + 5 
    
    if brick_idx < len(bricks):
        subset = abb_df_full.iloc[:brick_idx+1]
        now = bricks[brick_idx]["brick_timestamp"]
        current_infra = infra_df[infra_df["brick_timestamp"] <= now]
        f_df = compute_features_live(subset, current_infra)
        best_feats = f_df.iloc[-1]
        
        print("\nFeature Values at Peak:")
        for col in exact_cols:
            val = best_feats.get(col, 0)
            print(f"  {col:<22}: {val:.4f}")
            
        # Explainer Logic
        print("\nDIAGNOSTIC EXPLAINER:")
        if best_feats['vwap_zscore'] > 2.0:
            print("  - [BLOCKER] VWAP Z-Score is HIGH (>2.0). Price is in an exhaustion peak.")
        if best_feats['streak_exhaustion'] < -0.2:
            print(f"  - [BLOCKER] Streak Exhaustion is ACTIVE ({best_feats['streak_exhaustion']:.2f}). Too many green bricks.")
        if best_feats['wick_pressure'] > 0.3:
            print(f"  - [BLOCKER] Wick Pressure is HIGH ({best_feats['wick_pressure']:.2f}). Seeing upper rejection.")
        if best_feats['relative_strength'] < 0.5:
            print(f"  - [WEAKNESS] Relative Strength is LOW ({best_feats['relative_strength']:.2f}). ABB is not leading the sector.")
        if best_feats['is_trending_regime'] == 0:
            print(f"  - [WEAKNESS] Hurst Regime is 0 (Random Walk). Market structure is noisy.")
    
    # Re-run for the max SHORT moment to get features
    max_idx_s = sig_df["p_short"].idxmax()
    max_row_s = sig_df.loc[max_idx_s]
    print(f"\nTimestamp (SHORT): {max_row_s['timestamp']} | P(SHORT): {max_row_s['p_short']:.4f} | Price: {max_row_s['price']}")
    
    brick_idx_s = max_idx_s + 5 
    if brick_idx_s < len(bricks):
        subset = abb_df_full.iloc[:brick_idx_s+1]
        now = bricks[brick_idx_s]["brick_timestamp"]
        current_infra = infra_df[infra_df["brick_timestamp"] <= now]
        f_df = compute_features_live(subset, current_infra)
        best_feats_s = f_df.iloc[-1]
        
        print("\nFeature Values at SHORT Peak:")
        for col in exact_cols:
            val = best_feats_s.get(col, 0)
            print(f"  {col:<22}: {val:.4f}")
            
        print("\nDIAGNOSTIC EXPLAINER (SHORT):")
        if best_feats_s['relative_strength'] > -0.5:
            print(f"  - [WEAKNESS] Relative Strength is TOO HIGH ({best_feats_s['relative_strength']:.2f}). Stock is too strong for a short.")
        if best_feats_s['velocity'] < 0.2:
            print(f"  - [WEAKNESS] Velocity is LOW ({best_feats_s['velocity']:.2f}). Move is grinding, not explosive.")
    
    print("="*50)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Price and Bricks
    ax1.plot(df_abb["timestamp"], df_abb["ltp"], color='gray', alpha=0.3, label="Raw Ticks")
    
    # Entry threshold
    thresh_real = 0.55
    thresh_shadow = 0.40
    
    entries_long = sig_df[sig_df["p_long"] >= thresh_shadow]
    entries_short = sig_df[sig_df["p_short"] >= thresh_shadow]
    
    # Plot Shadow Entries
    ax1.scatter(entries_long["timestamp"], entries_long["price"], color='lightgreen', marker='^', s=80, alpha=0.6, label=f"LONG Shadow (p>{thresh_shadow})")
    ax1.scatter(entries_short["timestamp"], entries_short["price"], color='salmon', marker='v', s=80, alpha=0.6, label=f"SHORT Shadow (p>{thresh_shadow})")
    
    # Highlight Real triggers (if any)
    real_long = sig_df[sig_df["p_long"] >= thresh_real]
    real_short = sig_df[sig_df["p_short"] >= thresh_real]
    if not real_long.empty:
        ax1.scatter(real_long["timestamp"], real_long["price"], color='green', marker='^', s=150, edgecolors='black', label=f"LONG ACTUAL (p>{thresh_real})")
    if not real_short.empty:
        ax1.scatter(real_short["timestamp"], real_short["price"], color='red', marker='v', s=150, edgecolors='black', label=f"SHORT ACTUAL (p>{thresh_real})")

    ax1.set_title(f"ABB Today Shadow Analysis (2026-03-05)\nBrick Size: {brick_size:.2f} | Real Thresh: {thresh_real} | Shadow Thresh: {thresh_shadow}")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Probabilities
    ax2.plot(sig_df["timestamp"], sig_df["p_long"], color='green', label="P(LONG)")
    ax2.plot(sig_df["timestamp"], sig_df["p_short"], color='red', label="P(SHORT)")
    ax2.axhline(thresh_real, color='black', linestyle='-', alpha=0.5, label=f"Real ({thresh_real})")
    ax2.axhline(thresh_shadow, color='blue', linestyle='--', alpha=0.5, label=f"Shadow ({thresh_shadow})")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "storage/logs/abb_today_test.png"
    plt.savefig(plot_path)
    print(f"Deep dive plot saved to {plot_path}")
    
    # Summary
    print("\nSummary:")
    print(f"Long Entries: {len(entries_long)}")
    print(f"Short Entries: {len(entries_short)}")
    if len(entries_long) > 0:
        print(f"First Long: {entries_long['timestamp'].iloc[0]} @ {entries_long['price'].iloc[0]}")
    if len(entries_short) > 0:
        print(f"First Short: {entries_short['timestamp'].iloc[0]} @ {entries_short['price'].iloc[0]}")

if __name__ == "__main__":
    run_deep_dive()
