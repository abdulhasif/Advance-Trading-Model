import os
import glob
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(r"c:\Trading Platform\Advance Trading Model")
LOG_DIRS = [ROOT / "logs", ROOT / "spoofer_logs"]
OUTPUT_FILE = ROOT / "storage" / "logs" / "post_mortem.csv"

def aggregate_logs():
    all_trade_files = []
    
    # 1. Find all trade logs
    for d in LOG_DIRS:
        if d.exists():
            all_trade_files.extend(d.glob("*.csv"))
            # If you have specific txt files structured as CSVs, you could add them here
            
    trade_frames = []
    
    for f in all_trade_files:
        try:
            # Attempt to read each CSV
            df = pd.read_csv(f)
            
            # Check if it has trade-like columns by looking for standard names 
            # (adjust these based on your actual log formats)
            cols = [c.lower() for c in df.columns]
            
            if 'symbol' in cols and ('entry_time' in cols or 'entry' in cols):
                # Standardize column names structurally if needed, but let's assume it matches 
                # or is very close to paper_trader / backtester format.
                df["source_file"] = f.name
                trade_frames.append(df)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not trade_frames:
        print("No valid trade logs found.")
        return

    # Combine all
    combined = pd.concat(trade_frames, ignore_index=True)
    
    # Normalize col names
    combined.columns = [c.lower() for c in combined.columns]
    
    # Needs at least entry_time and exit_time to calc duration
    if 'entry_time' not in combined.columns or 'exit_time' not in combined.columns:
        print("Skipping duration calculaton: missing entry_time or exit_time.")
        # Minimal extraction
        extracted = combined
    else:
        # Extract asked columns
        keep_cols = []
        for c in ['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'entry_prob', 'brain1_prob', 'exit_reason', 'net_pnl_pct']:
            if c in combined.columns:
                keep_cols.append(c)

        extracted = combined[keep_cols].copy()
        
        # Calculate Exact Duration_Seconds
        extracted['entry_time'] = pd.to_datetime(extracted['entry_time'], errors='coerce')
        extracted['exit_time'] = pd.to_datetime(extracted['exit_time'], errors='coerce')
        
        extracted['duration_seconds'] = (extracted['exit_time'] - extracted['entry_time']).dt.total_seconds()
        
    # Isolate Losses
    if 'net_pnl_pct' in extracted.columns:
        losing_trades = extracted[extracted['net_pnl_pct'] < 0].copy()
    else:
        losing_trades = extracted.copy()
    
    if losing_trades.empty:
        print("No losing trades found to analyze!")
        # Fallback to outputting all
        losing_trades = extracted.copy()

    # Sort to find the fastest losing trades
    if 'duration_seconds' in losing_trades.columns:
        losing_trades = losing_trades.sort_values(by='duration_seconds', ascending=True)
    
    # Output top 20 fastest losing trades
    print("\n--- TOP 20 FASTEST LOSING TRADES ---")
    print(losing_trades.head(20).to_string())
    
    # Group by Symbol to identify repeat offenders
    print("\n--- REPEAT OFFENDERS (By Number of Losses) ---")
    offenders = losing_trades['symbol'].value_counts().reset_index()
    offenders.columns = ['symbol', 'loss_count']
    print(offenders.head(10).to_string())
    
    # Output to CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    losing_trades.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved post_mortem to: {OUTPUT_FILE}")

if __name__ == "__main__":
    aggregate_logs()
