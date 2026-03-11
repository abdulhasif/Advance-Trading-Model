"""
debug_spoofer_diagnosis.py
Quick offline diagnostic: loads warm-up buffers, forms a few bricks from today's
ticks for one symbol, then runs exactly the same inference + gate logic the
spoofer uses, printing the result at every step.

Usage:
    python debug_spoofer_diagnosis.py
"""
import sys, traceback
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import config
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.live.execution_guard import LiveExecutionGuard

CSV   = "storage/data/raw_ticks/debug_subset.csv"   # 3-stock subset
SYMS  = ["DIXON"]                                    # Focus on top mover

df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df.sort_values("timestamp")

universe = pd.read_csv(config.UNIVERSE_CSV)
universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])
stocks  = universe[~universe["is_index"]].reset_index(drop=True)
indices = universe[ universe["is_index"]].reset_index(drop=True)
sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}
stock_sectors    = {r["symbol"]: r["sector"]  for _, r in stocks.iterrows()}

print("=== UNIVERSE CHECK ===")
for sym in SYMS:
    sec = stock_sectors.get(sym, "NOT IN UNIVERSE")
    idx = sector_index_map.get(sec, "NO INDEX MAPPED")
    print(f"  {sym} -> sector={sec} -> sector_index={idx}")

# Warm up guard
sim_date = df["timestamp"].iloc[0].date()
all_syms = SYMS + list(sector_index_map.values())
guard = LiveExecutionGuard(symbols=all_syms, sectors=stock_sectors, before_date=sim_date)
guard.warm_up_all()

for sym in SYMS:
    sec    = stock_sectors.get(sym, "")
    bs     = guard.buffers[sym].to_dataframe()
    sec_sym = sector_index_map.get(sec, "")
    bs_sec  = guard.buffers[sec_sym].to_dataframe() if sec_sym in guard.buffers else pd.DataFrame()
    print(f"\n=== BUFFER SIZES ===")
    print(f"  {sym} buffer: {len(bs)} bricks")
    print(f"  {sec_sym} buffer: {len(bs_sec)} bricks")

# Load model
print("\n=== LOADING MODELS ===")
if config.USE_CALIBRATED_MODELS:
    b1_long  = joblib.load(str(config.BRAIN1_CALIBRATED_LONG_PATH))
    b1_short = joblib.load(str(config.BRAIN1_CALIBRATED_SHORT_PATH))
    t_long   = config.LONG_ENTRY_PROB_THRESH
    t_short  = config.SHORT_ENTRY_PROB_THRESH
    print("  Using Calibrated models")
else:
    b1_long  = xgb.XGBClassifier(); b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
    b1_short = xgb.XGBClassifier(); b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))
    t_long   = config.RAW_LONG_ENTRY_PROB_THRESH
    t_short  = config.RAW_SHORT_ENTRY_PROB_THRESH
    print("  Using Raw JSON models")

b2 = xgb.XGBRegressor(); b2.load_model(str(config.BRAIN2_MODEL_PATH))
print(f"  LONG threshold: {t_long}  |  SHORT threshold: {t_short}")
print(f"  CONV threshold: {config.ENTRY_CONV_THRESH}")

# Feed ticks until first brick, then run inference
brick_sizes = {}
for _, row in universe.iterrows():
    sym2, sec2 = row["symbol"], row["sector"]
    if sym2 in SYMS or sym2 in sector_index_map.values():
        sd = config.DATA_DIR / sec2 / sym2
        if sd.exists():
            pqs = sorted(sd.glob("*.parquet"))
            if pqs:
                try:
                    b_df2 = pd.read_parquet(pqs[-1])
                    if not b_df2.empty:
                        brick_sizes[sym2] = float(b_df2["brick_close"].iloc[-1]) * config.NATR_BRICK_PERCENT
                        continue
                except: pass
        brick_sizes[sym2] = 500 * config.NATR_BRICK_PERCENT

renko_states = {}
for sym in SYMS:
    sec = stock_sectors.get(sym, "UNKNOWN")
    renko_states[sym] = LiveRenkoState(sym, sec, brick_sizes.get(sym, 500*config.NATR_BRICK_PERCENT))
    # Seed the renko level from buffer
    bdf = guard.buffers[sym].to_dataframe()
    if not bdf.empty:
        renko_states[sym].renko_level = float(bdf["brick_close"].iloc[-1])

print("\n=== INJECTING TICKS ===")
bricks_seen = 0
for _, row in df[df["symbol"].isin(SYMS)].iterrows():
    sym  = row["symbol"]
    now  = row["timestamp"]
    price= float(row["ltp"])
    st   = renko_states[sym]
    new_bricks = st.process_tick(price, price, price, now, volume=float(row.get("volume",0)))
    for b in new_bricks:
        guard.buffers[sym].append(b)
        bricks_seen += 1

    if new_bricks and guard.buffers[sym].size >= 2:
        sec_sym2 = sector_index_map.get(st.sector, "")
        sec_bdf2 = guard.buffers[sec_sym2].to_dataframe() if sec_sym2 in guard.buffers else pd.DataFrame()
        bdf2     = guard.buffers[sym].to_dataframe()

        try:
            feat_df = compute_features_live(bdf2, sec_bdf2)
            latest  = feat_df.iloc[-1]
            X       = pd.DataFrame([latest.fillna(0).to_dict()])[config.FEATURE_COLS]

            p_long  = float(b1_long.predict_proba(X)[0][1])
            p_short = float(b1_short.predict_proba(X)[0][1])
            b1p     = max(p_long, p_short)

            signal = "FLAT"
            if p_long >= t_long and p_long >= p_short:
                signal = "LONG"
            elif p_short >= t_short:
                signal = "SHORT"

            X_m = pd.DataFrame([{
                "brain1_prob": b1p,
                "velocity": float(latest.get("velocity", 0)),
                "wick_pressure": float(latest.get("wick_pressure", 0)),
                "relative_strength": float(latest.get("relative_strength", 0)),
            }])
            b2c = float(np.clip(b2.predict(X_m)[0], 0, config.TARGET_CLIPPING_BPS))

            streak = int(latest.get("consecutive_same_dir", 0))
            vwap_z = float(latest.get("vwap_zscore", 0))
            wick_p = float(latest.get("wick_pressure", 0))
            rs     = float(latest.get("relative_strength", 0))

            dir_str = "UP" if b["direction"] > 0 else "DN"
            print(f"[{now.time()}] BRICK {sym} {dir_str} | p_long={p_long:.3f} p_short={p_short:.3f} "
                  f"| Signal={signal} | Conv={b2c:.1f} | Streak={streak} | RS={rs:.3f} | VWAP_Z={vwap_z:.2f} | Wick={wick_p:.2f}")

            gates = []
            if signal == "FLAT":
                gates.append(f"FLAT (p_long={p_long:.3f}<{t_long}, p_short={p_short:.3f}<{t_short})")
            if b2c < config.ENTRY_CONV_THRESH:
                gates.append(f"LowConv({b2c:.2f}<{config.ENTRY_CONV_THRESH})")
            if abs(rs) < config.ENTRY_RS_THRESHOLD:
                gates.append(f"LowRS({abs(rs):.3f}<{config.ENTRY_RS_THRESHOLD})")
            if wick_p > config.MAX_ENTRY_WICK:
                gates.append(f"HighWick({wick_p:.2f}>{config.MAX_ENTRY_WICK})")
            if streak >= config.STREAK_LIMIT:
                gates.append(f"Streak({streak}>={config.STREAK_LIMIT})")
            if signal == "LONG" and vwap_z > config.MAX_VWAP_ZSCORE:
                gates.append(f"VWAPEx({vwap_z:.2f}>{config.MAX_VWAP_ZSCORE})")

            if gates:
                print(f"  -> BLOCKED by: {', '.join(gates)}")
            else:
                print(f"  -> ALL GATES PASSED! Would have entered {signal}")

        except Exception as e:
            print(f"  -> INFERENCE FAILED: {e}")
            traceback.print_exc()

print(f"\nTotal bricks formed: {bricks_seen}")
