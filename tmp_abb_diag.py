import pandas as pd
import warnings
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.core.risk import RiskFortress
import config
from src.live.engine import (
    load_models, EXPECTED_FEATURES,
    LONG_ENTRY_PROB_THRESH, SHORT_ENTRY_PROB_THRESH,
    ENTRY_CONV_THRESH, ENTRY_RS_THRESHOLD, MAX_ENTRY_WICK,
    MIN_PRICE_FILTER, MIN_CONSECUTIVE_BRICKS, MIN_BRICKS_TODAY,
    passes_soft_veto
)

def analyze_abb():
    tick_file = Path(r"C:\Trading Platform\Advance Trading Model\storage\data\raw_ticks\raw_ticks_dump_2026-03-06.csv")
    print("Loading tick file...")
    cols = ["timestamp", "symbol", "ltp", "volume", "oi"] 
    # Handle possible header mismatch depending on how live_spoofer dumps it
    df = pd.read_csv(tick_file)
    if "symbol" not in df.columns:
        df = pd.read_csv(tick_file, names=cols, header=None)
        
    df = df[df["symbol"] == "ABB"]
    print(f"Extracted {len(df)} ticks for ABB.")
    
    if df.empty:
        print("No ticks found for ABB today.")
        return

    # Sort and reset
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calculate brick size (using config fallback for offline)
    brick_size = 500 * config.NATR_BRICK_PERCENT
    # Get exact brick size if available
    uni = pd.read_csv(config.UNIVERSE_CSV)
    try:
        sec = uni[uni["symbol"] == "ABB"]["sector"].iloc[0]
    except Exception:
        sec = "Capital Goods"
        
    st = LiveRenkoState("ABB", sec, brick_size)
    print(f"Initialized Renko State for ABB (Sector: {sec}) with Brick Size: {brick_size:.2f}")

    # Load models
    b1_long, b1_short, b2 = load_models()
    
    # Process ticks identically to engine.py
    completed_bricks = []
    
    # Instead of sector processing, we'll mock sector dataframe as empty 
    # (live engine handles empty sector cleanly)
    sec_bdf = pd.DataFrame()
    
    print("\nStarting Tick Replay...")
    blocks = []
    
    for i, row in df.iterrows():
        ts = row["timestamp"]
        price = row["ltp"]
        
        # We don't have high/low in basic tick dump, assume ltp=high=low
        new_b = st.process_tick(price, price, price, ts)
        
        for b in new_b:
            completed_bricks.append(b)
            # ── Model Logic triggering per new brick ──
            
            # Need minimum 2 bricks for feature math
            if len(completed_bricks) < 2:
                continue
                
            bdf_raw = pd.DataFrame(completed_bricks)
            feat_df = compute_features_live(bdf_raw, sec_bdf)
            latest = feat_df.iloc[-1]
            
            raw_high = b["brick_high"]
            raw_close = b["brick_close"]
            print(f"DEBUG: Brick={len(completed_bricks)} RawHigh={raw_high} RawClose={raw_close} IsLastInGap={len(completed_bricks)==len(new_b)}")
            
            # Prediction
            X = pd.DataFrame([latest[EXPECTED_FEATURES].infer_objects(copy=False).fillna(0).to_dict()])
            
            sig_str = "FLAT"
            b1p = 0.0
            p_long = float(b1_long.predict_proba(X)[0, 1])
            p_short = float(b1_short.predict_proba(X)[0, 1])
            
            if p_long >= LONG_ENTRY_PROB_THRESH and p_long > p_short:
                sig_str = "LONG"
                b1p = p_long
            elif p_short >= SHORT_ENTRY_PROB_THRESH and p_short > p_long:
                sig_str = "SHORT"
                b1p = p_short
                
            # Brain2
            X_m = pd.DataFrame([{
                "brain1_prob": b1p,
                "velocity": float(latest.get("velocity", 0)),
                "wick_pressure": float(latest.get("wick_pressure", 0)),
                "relative_strength": float(latest.get("relative_strength", 0)),
            }])
            b2c = float(b2.predict(X_m)[0])
            b2c = max(0.0, min(100.0, b2c))
            
            rs_val = float(latest.get("relative_strength", 0))
            is_veto = not passes_soft_veto(sig_str, rs_val)
            wick_p = float(latest.get("wick_pressure", 0))
            
            # Whipsaw / consecutive check
            recent_bricks = completed_bricks[-MIN_CONSECUTIVE_BRICKS:]
            dirs = [x["direction"] for x in recent_bricks]
            exp_dir = 1 if sig_str == "LONG" else -1
            whipsaw_fail = len(completed_bricks) < MIN_CONSECUTIVE_BRICKS or not all(d == exp_dir for d in dirs)
            
            blocks.append({
                "time": ts.strftime('%H:%M:%S'),
                "price": price,
                "brick_count": len(completed_bricks),
                "sig": sig_str,
                "prob": b1p,
                "conv": b2c,
                "rs": rs_val,
                "wick": wick_p,
                "vetoed": is_veto,
                "whipsaw_fail": whipsaw_fail
            })
            
            if sig_str != "FLAT":
                print(f"[{ts.strftime('%H:%M:%S')}] Brick #{len(completed_bricks)} | {sig_str} | Prob: {b1p:.3f} | Conv: {b2c:.1f}")
                print(f"   -> RS: {rs_val:.2f} (Needs {ENTRY_RS_THRESHOLD}) | Wick: {wick_p:.2f} (Max {MAX_ENTRY_WICK})")
                print(f"   -> Whipsaw Pass: {not whipsaw_fail} | Veto Pass: {not is_veto}")
                
                # Check ALL gates explicitly
                if b1p < (LONG_ENTRY_PROB_THRESH if sig_str=="LONG" else SHORT_ENTRY_PROB_THRESH):
                    print("   => FAILED: Prob too low")
                elif b2c < ENTRY_CONV_THRESH:
                    print(f"   => FAILED: Brain2 Conviction ({b2c:.1f} < {ENTRY_CONV_THRESH})")
                elif is_veto:
                    print("   => FAILED: Soft Veto (Trend/RS discrepancy)")
                elif sig_str == "LONG" and rs_val < ENTRY_RS_THRESHOLD:
                    print(f"   => FAILED: Relative Strength ({rs_val:.2f} < {ENTRY_RS_THRESHOLD} Leader cutoff)")
                elif sig_str == "SHORT" and rs_val > -ENTRY_RS_THRESHOLD:
                    print(f"   => FAILED: Relative Strength ({rs_val:.2f} > -{ENTRY_RS_THRESHOLD} Laggard cutoff)")
                elif wick_p > MAX_ENTRY_WICK:
                    print(f"   => FAILED: Wick Pressure Trap ({wick_p:.2f} > {MAX_ENTRY_WICK})")
                elif whipsaw_fail:
                    print(f"   => FAILED: Whipsaw Protection (Not enough consecutive bricks in dir)")
                else:
                    print("   => PASSED: Signal was technically valid to execute!")

if __name__ == '__main__':
    analyze_abb()
