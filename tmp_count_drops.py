import pandas as pd
import glob
import numpy as np

features_dir = "storage/features/*/*.parquet"
files = glob.glob(features_dir)

total_signals = 0
drops = {
    'conviction': 0,
    'rs': 0,
    'wick': 0,
    'whipsaw': 0,
    'time': 0,
    'penny': 0
}

# Config matching
LONG_ENTRY_PROB_THRESH = 0.75
SHORT_ENTRY_PROB_THRESH = 0.75
ENTRY_CONV_THRESH = 5
ENTRY_RS_THRESHOLD = 0.5
MAX_ENTRY_WICK = 0.5
MIN_PRICE_FILTER = 100

for pf in files:
    try:
        df = pd.read_parquet(pf)
    except:
        continue
        
    for i, row in df.iterrows():
        pl = row.get("brain1_prob_long", 0.0)
        ps = row.get("brain1_prob_short", 0.0)
        
        sig = "FLAT"
        p = 0.0
        if pl >= LONG_ENTRY_PROB_THRESH and ps >= SHORT_ENTRY_PROB_THRESH:
            if pl >= ps:
                sig, p = "LONG", pl
            else:
                sig, p = "SHORT", ps
        elif pl >= LONG_ENTRY_PROB_THRESH:
            sig, p = "LONG", pl
        elif ps >= SHORT_ENTRY_PROB_THRESH:
            sig, p = "SHORT", ps
            
        if sig == "FLAT":
            continue
            
        total_signals += 1
        
        # Time Lock approximation (just 9:15-3:15)
        # We skip time lock for this exact analysis to focus on structural gates
        
        # 1. Conviction
        conv = row.get("brain2_conviction", 0.0)
        if conv < ENTRY_CONV_THRESH:
            drops['conviction'] += 1
            continue
            
        # 2. Price Filter
        brick_close = row.get("brick_close", 0.0)
        if brick_close < MIN_PRICE_FILTER:
            drops['penny'] += 1
            continue
            
        # 3. RS Anchor
        rel_str = row.get("relative_strength", 0.0)
        if sig == "LONG" and rel_str < ENTRY_RS_THRESHOLD:
            drops['rs'] += 1
            continue
        if sig == "SHORT" and rel_str > -ENTRY_RS_THRESHOLD:
            drops['rs'] += 1
            continue
            
        # 4. Wick Trap
        wick_p = row.get("wick_pressure", 0.0)
        if wick_p > MAX_ENTRY_WICK:
            drops['wick'] += 1
            continue
            
        # 5. Whipsaw Mismatch
        brick_dir = row.get("direction", 1)
        expected_dir = 1 if sig == "LONG" else -1
        if brick_dir != expected_dir or row.get("consecutive_same_dir", 0) < 1:
            drops['whipsaw'] += 1
            continue

print(f"Total Raw Signals > 0.75: {total_signals}")
print(f"Drops: {drops}")
