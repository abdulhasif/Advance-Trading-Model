import os
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import time
import logging

# Configuration
INPUT_DIR = Path("storage/data")
OUTPUT_DIR = Path("storage/data_clean")
LOG_FILE = Path("storage/logs/data_clean_summary.log")
MAX_VARIANCE = 0.15  # 15% jump threshold
STALE_THRESHOLD = 15  # 15 consecutive identical candles
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Setup Logging
os.makedirs(LOG_FILE.parent, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_file(file_path: Path):
    """Processes a single parquet file: cleans anomalies and fills gaps."""
    try:
        # 1. Load Data
        df = pd.read_parquet(file_path)
        if df.empty:
            return None

        symbol = file_path.stem
        sector = file_path.parent.name
        
        # Ensure timestamp is datetime and sorted
        if 'timestamp' not in df.columns:
            # Check if index is datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                logger.warning(f"Skipping {symbol}: No timestamp found.")
                return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # 2. Market Hours Enforcement (NSE: 09:15 - 15:30)
        df = df[df['timestamp'].dt.time >= MARKET_OPEN]
        df = df[df['timestamp'].dt.time <= MARKET_CLOSE]
        
        if df.empty:
            return f"Dropped {symbol}: No data within market hours."

        # 3. Detect and Purge Corporate Actions (>15% intra-day jumps)
        # We group by date to ensure we don't flag overnight gaps
        df['prev_close'] = df.groupby('date')['close'].shift(1)
        df['jump'] = (df['open'] - df['prev_close']).abs() / df['prev_close']
        
        # Identify "dirty" days
        dirty_days = df[df['jump'] > MAX_VARIANCE]['date'].unique()
        
        if len(dirty_days) > 0:
            for d in dirty_days:
                logger.info(f"Purging {symbol} on {d} due to >{MAX_VARIANCE*100}% intra-day price jump.")
            df = df[~df['date'].isin(dirty_days)]
            
        if df.empty:
            return f"Dropped {symbol}: All days purged due to price anomalies."

        # 4. Filter Zero Volume (Active hours)
        df = df[df['volume'] > 0]

        # 5. Stale Data Removal (>15 identical consecutive minutes)
        # Identify where OHLC are all the same
        df['is_stale'] = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
        # Compute consecutive streaks of identical candles
        df['stale_group'] = (df['is_stale'] != df['is_stale'].shift()).cumsum()
        stale_counts = df[df['is_stale']].groupby(['date', 'stale_group']).size()
        
        dirty_stale_days = stale_counts[stale_counts > STALE_THRESHOLD].index.get_level_values(0).unique()
        if len(dirty_stale_days) > 0:
            for d in dirty_stale_days:
                logger.info(f"Purging {symbol} on {d} due to >{STALE_THRESHOLD} mins of stale data.")
            df = df[~df['date'].isin(dirty_stale_days)]

        if df.empty:
            return f"Dropped {symbol}: All days purged due to stale/illiquid data."

        # 6. Gap Forward-Filling (Strict 1-minute grid)
        # To preserve Renko logic, we need every minute represented
        processed_days = []
        for d, day_df in df.groupby('date'):
            # Create the full 1-minute range for this day
            full_range = pd.date_range(
                start=pd.Timestamp.combine(d, MARKET_OPEN),
                end=pd.Timestamp.combine(d, MARKET_CLOSE),
                freq='1min'
            )
            
            # Ensure TZ consistency (remove TZ for reindexing with naive range if needed)
            if day_df['timestamp'].dt.tz is not None:
                day_df['timestamp'] = day_df['timestamp'].dt.tz_localize(None)
                
            day_df = day_df.set_index('timestamp').reindex(full_range)
            
            # Forward-fill OHLC
            day_df['close'] = day_df['close'].ffill()
            day_df['open'] = day_df['open'].fillna(day_df['close'])
            day_df['high'] = day_df['high'].fillna(day_df['close'])
            day_df['low'] = day_df['low'].fillna(day_df['close'])
            # Set volume to 0 for filled gaps
            day_df['volume'] = day_df['volume'].fillna(0)
            
            # Final check: Drop leading NaNs if the day didn't start at 09:15
            day_df = day_df.dropna(subset=['close'])
            
            processed_days.append(day_df.reset_index().rename(columns={'index': 'timestamp'}))

        if not processed_days:
            return f"Dropped {symbol}: No clean days remained after processing."

        final_df = pd.concat(processed_days).drop(columns=['prev_close', 'jump', 'date', 'is_stale', 'stale_group'], errors='ignore')

        # 7. Save Output
        output_path = OUTPUT_DIR / sector / file_path.name
        os.makedirs(output_path.parent, exist_ok=True)
        final_df.to_parquet(output_path)
        
        return f"Sanitized {symbol}: {len(final_df)} rows saved to {sector}/."

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def main():
    logger.info("--- STARTING MARKET DATA SANITIZATION ---")
    
    # Identify all symbol parquets recursively
    # Path: storage/data/[SECTOR]/[SYMBOL]/[YEAR].parquet
    all_files = list(INPUT_DIR.rglob("*.parquet"))
    
    # Filter out raw_ticks or other unrelated directories
    all_files = [f for f in all_files if "raw_ticks" not in str(f)]
            
    if not all_files:
        logger.error(f"No parquet files found in {INPUT_DIR}.")
        return

    logger.info(f"Identified {len(all_files)} files for sanitization.")
    
    # Process in parallel
    num_procs = max(1, cpu_count() - 1)
    logger.info(f"Using {num_procs} CPUs for parallel processing.")
    
    with Pool(num_procs) as pool:
        results = pool.map(process_file, all_files)
        
    # Summary
    success_count = sum(1 for r in results if r and r.startswith("Sanitized"))
    dropped_count = sum(1 for r in results if r and r.startswith("Dropped"))
    
    logger.info("--- CLEANING COMPLETE ---")
    logger.info(f"Files Processed Successfully: {success_count}")
    logger.info(f"Files Dropped Entirely: {dropped_count}")
    
    # Detailed log
    for res in results:
        if res: logger.info(res)

if __name__ == "__main__":
    main()
