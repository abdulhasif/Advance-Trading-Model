import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import config
from src.ml.backtester import load_test_data

def run_diagnostic(symbol="ABB"):
    print(f"Loading data for {symbol}...")
    # Load base json models directly
    b1_long = xgb.XGBClassifier()
    b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
    
    b1_short = xgb.XGBClassifier()
    b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))
    
    b2 = xgb.XGBRegressor()
    b2.load_model(str(config.BRAIN2_MODEL_PATH))
    
    # Load test data
    df = load_test_data(start_year=2025, end_year=2026)
    df = df[df["_symbol"] == symbol].copy()
    print(f"Loaded {len(df)} bricks for {symbol}.")
    
    if len(df) == 0:
        return
        
    X = df[config.FEATURE_COLS].fillna(0)
    
    # Run Inference
    p_long = b1_long.predict_proba(X)[:, 1]
    p_short = b1_short.predict_proba(X)[:, 1]
    
    # Brain 2 requires the brain1_prob as a feature
    b1_prob_combined = np.where(p_long > p_short, p_long, p_short)
    X["brain1_prob"] = b1_prob_combined
    
    conv = b2.predict(X)
    
    df["p_long"] = p_long
    df["p_short"] = p_short
    df["conviction"] = conv
    
    # Rejection Counters
    stats = {
        "Signals Generated": 0,
        "Drop: Low Prob (<60%)": 0,
        "Drop: Low Conviction (<1.5)": 0,
        "Drop: Soft Veto": 0,
        "Drop: RS Gate": 0,
        "Drop: Wick Trap": 0,
        "Drop: Morning Lock": 0,
        "Drop: Whipsaw/Freshness": 0,
        "Drop: FOMO Limit (>7)": 0,
        "Accepted Trades": 0
    }
    
    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["brick_timestamp"]
        
        # 1. Base Signal (Max Prob wins)
        signal = "FLAT"
        prob = 0.0
        if row["p_long"] > row["p_short"]:
            signal = "LONG"
            prob = row["p_long"]
        elif row["p_short"] > row["p_long"]:
            signal = "SHORT"
            prob = row["p_short"]
            
        if signal == "FLAT":
            continue
            
        stats["Signals Generated"] += 1
        
        # 2. Timing Gate
        is_too_early = (ts.hour < 9) or (ts.hour == 9 and ts.minute < 20)
        if is_too_early:
            stats["Drop: Morning Lock"] += 1
            continue
            
        # 3. Probability Gate
        if prob < 0.50:  # User changed to 50%
            stats["Drop: Low Prob (<60%)"] += 1
            continue
            
        # 4. Conviction Gate
        if row["conviction"] < 1.5:
            stats["Drop: Low Conviction (<1.5)"] += 1
            continue
            
        # 5. Soft Veto
        rel_str = row["relative_strength"]
        if signal == "LONG" and rel_str < -config.SOFT_VETO_THRESHOLD:
            stats["Drop: Soft Veto"] += 1
            continue
        if signal == "SHORT" and rel_str > config.SOFT_VETO_THRESHOLD:
            stats["Drop: Soft Veto"] += 1
            continue
            
        # 6. RS Gate (Leader/Laggard)
        if signal == "LONG" and rel_str < config.ENTRY_RS_THRESHOLD:
            stats["Drop: RS Gate"] += 1
            continue
        if signal == "SHORT" and rel_str > -config.ENTRY_RS_THRESHOLD:
            stats["Drop: RS Gate"] += 1
            continue
            
        # 7. Wick Trap
        if row["wick_pressure"] > config.MAX_ENTRY_WICK:
            stats["Drop: Wick Trap"] += 1
            continue
            
        # 8. Whipsaw Gate
        brick_dir = 1 if row["trend_slope"] > 0 else -1  # Approx 
        expected_dir = 1 if signal == "LONG" else -1
        # if expected_dir != brick_dir:
        #    stats["Drop: Whipsaw/Freshness"] += 1
        #    continue
            
        # 9. FOMO Rule
        if row["consecutive_same_dir"] >= config.STREAK_LIMIT:
            stats["Drop: FOMO Limit (>7)"] += 1
            continue
            
        stats["Accepted Trades"] += 1

    print("\n========= DIAGNOSTIC REPORT =========")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_diagnostic("ABB")
    print("\n")
    run_diagnostic("INFY")
