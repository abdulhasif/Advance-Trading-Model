import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import xgboost as xgb

import config
import joblib
import src.live.paper_trader as pt
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.live.execution_guard import LiveExecutionGuard
from src.live.control_state import CONTROL_STATE

# Redirect logs so we don't pollute the real paper trading results
SPOOFER_DIR = config.PROJECT_ROOT / "spoofer_logs"
SPOOFER_DIR.mkdir(exist_ok=True)

pt.SIGNAL_LOG = SPOOFER_DIR / "spoofer_signals.csv"
pt.TRADE_LOG = SPOOFER_DIR / "spoofer_trades.csv"
pt.DAILY_LOG = SPOOFER_DIR / "spoofer_daily.csv"
pt.LIVE_PNL_FILE = SPOOFER_DIR / "spoofer_pnl.json"

def run_offline_spoofer(csv_file: Path):
    print(f"Loading Spoofer CSV: {csv_file}")
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if "timestamp" in df.columns:
        # Aggressively strip all timezones to naive datetime to prevent subtraction crashes
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp")
    
    # Needs minimum: symbol, timestamp, ltp
    symbols_in_csv = df["symbol"].unique()
    print(f"Symbols found in CSV: {list(symbols_in_csv)}")

    # 1. Load Universe
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])
    stocks  = universe[~universe["is_index"]].reset_index(drop=True)
    indices = universe[ universe["is_index"]].reset_index(drop=True)
    sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}

    # 2. Warmup Brick Sizes
    print("Warming up brick sizes...")
    brick_sizes = {}
    for _, row in universe.iterrows():
        sym, sec = row["symbol"], row["sector"]
        if sym in symbols_in_csv or sym in sector_index_map.values():
            stock_dir = config.DATA_DIR / sec / sym
            if stock_dir.exists():
                pqs = sorted(stock_dir.glob("*.parquet"))
                if pqs:
                    try:
                        b_df = pd.read_parquet(pqs[-1])
                        if not b_df.empty:
                            brick_sizes[sym] = b_df["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
                            continue
                    except: pass
            brick_sizes[sym] = 500 * config.NATR_BRICK_PERCENT

    # 3. Setup States
    renko_states = {}
    for sym in symbols_in_csv:
        sec = stocks[stocks["symbol"] == sym]["sector"].iloc[0] if sym in stocks["symbol"].values else "UNKNOWN"
        # Fallback to a nominal brick size if NATR calculation fails
        fallback_brick = 500 * config.NATR_BRICK_PERCENT
        renko_states[sym] = LiveRenkoState(sym, sec, brick_sizes.get(sym, fallback_brick))
    
    sector_renko = {}
    csv_sectors = set(st.sector for st in renko_states.values())
    for sec in csv_sectors:
        sec_sym = sector_index_map.get(sec)
        if sec_sym:
            fallback_brick = 500 * config.NATR_BRICK_PERCENT
            sector_renko[sec_sym] = LiveRenkoState(sec_sym, sec, brick_sizes.get(sec_sym, fallback_brick))

    stock_sectors = {r["symbol"]: r["sector"] for _, r in stocks.iterrows()}
    all_syms = list(renko_states.keys()) + list(sector_renko.keys())

    # Contamination Shield: Extract the simulation date to prevent warmup data leak
    sim_date = df["timestamp"].iloc[0].date() if not df.empty else None
    guard = LiveExecutionGuard(symbols=all_syms, sectors=stock_sectors, before_date=sim_date)
    
    print("Pre-loading historical buffers...")
    guard.warm_up_all()

    # Strip tzinfo from all loaded historical bricks to match the naive CSV ticks
    for sym, buf in guard.buffers.items():
        for b in buf._buffer:
            if isinstance(b["brick_timestamp"], pd.Timestamp) and b["brick_timestamp"].tzinfo is not None:
                b["brick_timestamp"] = b["brick_timestamp"].tz_localize(None)

    for sym, st in list(renko_states.items()) + list(sector_renko.items()):
        if sym in guard.buffers and not guard.buffers[sym].to_dataframe().empty:
            bdf = guard.buffers[sym].to_dataframe()
            st.renko_level = bdf["brick_close"].iloc[-1]
            st.brick_start_time = bdf["brick_timestamp"].iloc[-1]
        else:
            # If no history, wait for first tick
            pass

    # 4. Load Models & Portfolio
    if config.USE_CALIBRATED_MODELS:
        print("Loading calibrated .pkl models...")
        b1_long = joblib.load(str(config.BRAIN1_CALIBRATED_LONG_PATH))
        b1_short = joblib.load(str(config.BRAIN1_CALIBRATED_SHORT_PATH))
    else:
        print("Loading raw .json models...")
        b1_long = xgb.XGBClassifier(); b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
        b1_short = xgb.XGBClassifier(); b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))

    b2 = xgb.XGBRegressor();  b2.load_model(str(config.BRAIN2_MODEL_PATH))
    portfolio = pt.PaperPortfolio(pt.PAPER_CAPITAL)
    
    last_preds = {}
    last_entry_minutes = {}
    active_positions = {} # FIX: Simulation State Lock (Prevents duplicate entries on same timestamp)
    
    print("=" * 60)
    print("SPOOFER INJECTION STARTED")
    print("=" * 60)
    
    tick_count = 0
    start_time = time.time()
        
    for idx, row in df.iterrows():
        sym = row["symbol"]
        if sym not in renko_states:
            continue
            
        now = row["timestamp"]
        price = float(row["ltp"])
        high = float(row.get("high", price))
        low = float(row.get("low", price))
        
        tick_count += 1
        
        st = renko_states[sym]
        now_minute = f"{now.hour:02d}:{now.minute:02d}"
        
        v = float(row.get("volume", 0))
        
        # 1. Process Tick
        new_bricks = st.process_tick(price, high, low, now, volume=v)
        for b in new_bricks:
            guard.buffers[sym].append(b)
            # Print physical brick formation
            dir_str = "UP" if b["direction"] > 0 else "DN"
            print(f"[{now.time()}] BRICK: {sym} {dir_str} @ {b['brick_close']:.2f} (Dur: {b['duration_seconds']}s)")
            
        # 2. Update existing positions
        if sym in portfolio.positions:
            portfolio.positions[sym]["last_price"] = price
            if sym in last_preds:
                lp = last_preds[sym]
                has_new_bricks = len(new_bricks) > 0
                
                # FIX: Use the actual direction of the truly formed bricks, not the stale inference dictionary
                current_brick_dir = new_bricks[-1]["direction"] if has_new_bricks else lp.get("brick_dir", 0)
                
                portfolio.update_position(sym, price, current_brick_dir, lp["b2c"], lp["signal"], lp["b1p"], new_bricks_formed=has_new_bricks)
                exit_reason = portfolio.check_exit(sym, price, now, lp["b2c"], lp["signal"], lp["b1p"], brick_dir=current_brick_dir)
                if exit_reason:
                    pos = portfolio.close_position(sym, price, now, exit_reason)
                    
                    # Track Daily Loss for Whipsaw Protection (Ported from paper_trader)
                    if hasattr(portfolio.simulator, "trade_history") and len(portfolio.simulator.trade_history) > 0:
                        last_trade = portfolio.simulator.trade_history[-1]
                        if last_trade.symbol == sym and last_trade.net_pnl <= 0:
                            portfolio._daily_stock_losses[sym] = portfolio._daily_stock_losses.get(sym, 0) + 1

                    if sym in active_positions: del active_positions[sym] # Release Lock
                    print(f"[{now.time()}] EXIT: {sym} {lp['signal']} @ {price:.2f} | PnL: Rs {pos.get('unrealized_pnl', 0):.2f} | Reason: {exit_reason}")
                    
        # 3. Model Inference (only if new bricks formed and enough history)
        if not new_bricks or guard.buffers[sym].size < 2:
            continue
            
        sec_sym = sector_index_map.get(st.sector, "")
        sec_bdf = guard.buffers[sec_sym].to_dataframe() if sec_sym in guard.buffers else pd.DataFrame()
        bdf = compute_features_live(guard.buffers[sym].to_dataframe(), sec_bdf)
        latest = bdf.iloc[-1]
        
        feat_dict = latest.infer_objects(copy=False).fillna(0).to_dict()
        
        # Build DataFrame with exact column order
        X = pd.DataFrame([feat_dict])[config.FEATURE_COLS]
        # Inference
        p_long = float(b1_long.predict_proba(X)[0][1])
        p_short = float(b1_short.predict_proba(X)[0][1])
        
        b1p = max(p_long, p_short)
        b1d = 1 if p_long >= p_short else -1
        signal = "FLAT"
        
        # FIX 1: Strict Directional Select with raw-aware thresholds
        t_long  = config.LONG_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
        t_short = config.SHORT_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH

        if p_long >= t_long and p_long >= p_short:
            signal = "LONG"
        elif p_short >= t_short:
            signal = "SHORT"
            
        # Entry Gates
        if signal == "FLAT":
            continue

        entry_prob_ok = True
        
        # FIX: Brain 2 expects exactly its defined subset of features
        X_m = X.copy()
        X_m["brain1_prob"] = b1p
        
        # Use config.BRAIN2_FEATURES which we know implies exactly the 4 trained cols
        if hasattr(config, "BRAIN2_FEATURES"):
            b2_cols = config.BRAIN2_FEATURES
        else:
            b2_cols = ['brain1_prob', 'velocity', 'wick_pressure', 'relative_strength']
            
        X_m = X_m[b2_cols]
        b2c = float(np.clip(b2.predict(X_m)[0], 0, config.TARGET_CLIPPING_BPS))
        sec_dir = guard.buffers[sec_sym]._buffer[-1]["direction"] if sec_sym in guard.buffers and guard.buffers[sec_sym].size > 0 else 0
        score = b2c
        rel_str = float(latest.get("relative_strength", 0))
        brick_dir = int(latest.get("direction", 0))
        
        last_preds[sym] = {
            "b1p": b1p, "b2c": b2c, "signal": signal, 
            "score": score, "rel_str": rel_str, "brick_dir": brick_dir
        }
        
        print(f"[{now.time()}] INFERENCE: {sym} | Brain1 (PROB): {b1p:.4f} | Brain2 (CONV): {b2c:.1f} | Signal: {signal}")
        
        do_log = b1p > 0.5 or (1 - b1p) > 0.5 # More verbose for sniping debugging
        
        # End of Day Block (No entries after cutoff)
        no_entry = (now.hour > config.NO_NEW_ENTRY_HOUR) or (now.hour == config.NO_NEW_ENTRY_HOUR and now.minute >= config.NO_NEW_ENTRY_MIN)
        if no_entry:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: EOD No Entry Block")
            continue
            
        # morning Entry Lock (Respect config.ENTRY_LOCK_MINUTES)
        morning_lock_min = config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES
        morning_lock_hour = config.MARKET_OPEN_HOUR + (morning_lock_min // 60)
        morning_lock_min %= 60
        
        try:
            _start_time = latest.get("brick_start_time")
            if _start_time:
                _st_dt = pd.to_datetime(_start_time)
                if _st_dt.hour < morning_lock_hour or (_st_dt.hour == morning_lock_hour and _st_dt.minute < morning_lock_min):
                    if do_log: print(f"[{now.time()}] [DROP] {sym}: Morning Gate Block")
                    continue
        except Exception as e:
            pass
            
        # Gate: Sector Alignment (Soft Veto) (Bypassed by high conviction)
        if b2c < config.VETO_BYPASS_CONV:
            if signal == "LONG" and rel_str < -config.SOFT_VETO_THRESHOLD:
                if do_log: print(f"[{now.time()}] [DROP] {sym}: Soft Veto (Sector Weak: {rel_str:.2f})")
                continue
            if signal == "SHORT" and rel_str > config.SOFT_VETO_THRESHOLD:
                if do_log: print(f"[{now.time()}] [DROP] {sym}: Soft Veto (Sector Strong: {rel_str:.2f})")
                continue

        # 6. RS/Z-Score/Wick Gates
        # USER REQUEST: Implemented VWAP Z-Score Exhaustion in paper trading and matching logs here.
        z_vwap = float(latest.get("vwap_zscore", 0))
        if signal == "LONG" and z_vwap > config.MAX_VWAP_ZSCORE:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: VWAP_ZSCORE_EXHAUSTION_({round(z_vwap,2)})")
            continue
        if signal == "SHORT" and z_vwap < -config.MAX_VWAP_ZSCORE:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: VWAP_ZSCORE_EXHAUSTION_({round(z_vwap,2)})")
            continue
            
        wick_p = float(latest.get("wick_pressure", 0))
        if wick_p > config.MAX_ENTRY_WICK:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Wick Pressure Cut ({wick_p:.2f})")
            continue
            
        if not entry_prob_ok:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Low Prob ({b1p:.2f})")
            continue
        if b2c < config.ENTRY_CONV_THRESH:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Low Conv ({b2c:.2f} < {config.ENTRY_CONV_THRESH})")
            continue
        # Gate: RS Anchor — only trade leaders/laggards (Bypassed by high conviction)
        if b2c < config.VETO_BYPASS_CONV:
            if signal == "LONG" and rel_str < config.ENTRY_RS_THRESHOLD:
                if do_log: print(f"[{now.time()}] [DROP] {sym}: Low RS ({rel_str:.2f} < {config.ENTRY_RS_THRESHOLD})")
                continue
            if signal == "SHORT" and rel_str > -config.ENTRY_RS_THRESHOLD:
                if do_log: print(f"[{now.time()}] [DROP] {sym}: Low RS ({rel_str:.2f} > {-config.ENTRY_RS_THRESHOLD})")
                continue
        if float(latest.get("wick_pressure", 0)) > config.MAX_ENTRY_WICK: # Wick Gate
            if do_log: print(f"[{now.time()}] [DROP] {sym}: High Wick Pressure ({latest.get('wick_pressure', 0):.2f} > {config.MAX_ENTRY_WICK})")
            continue
        if last_entry_minutes.get(sym) == now_minute:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Same Minute")
            continue
        
        # State Lock Check (Prevents duplicate entries on flicker)
        if sym in active_positions:
            continue
        
        # Whipsaw checks
        if len(st.bricks) < config.MIN_CONSECUTIVE_BRICKS: # MIN_CONSECUTIVE_BRICKS
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Not enough bricks ({len(st.bricks)} < {config.MIN_CONSECUTIVE_BRICKS})")
            continue
        recent_bricks = st.bricks[-config.MIN_CONSECUTIVE_BRICKS:]
        recent_dirs = [b["direction"] for b in recent_bricks]
        expected_dir = 1 if signal == "LONG" else -1
        if not all(d == expected_dir for d in recent_dirs):
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Whipsaw mismatch (Dirs: {recent_dirs} | Expected: {expected_dir})")
            continue
            
        # Gate: Anti-FOMO Streak Limit
        streak_count = int(latest.get("consecutive_same_dir", 0))
        if streak_count >= config.STREAK_LIMIT:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: FOMO Streak Limit ({streak_count} >= {config.STREAK_LIMIT})")
            continue
            
        today_date = now.date()
        today_bricks = sum(1 for b in st.bricks if pd.to_datetime(b["brick_timestamp"]).date() == today_date)
        # USER REQUEST: Commented out MIN_BRICKS_TODAY check to match spoofer behavior.
        # if today_bricks < config.MIN_BRICKS_TODAY: # MIN_BRICKS_TODAY
        #     if do_log: print(f"[{now.time()}] [DROP] {sym}: Not enough bricks today ({today_bricks} < {config.MIN_BRICKS_TODAY})")
        #     continue
            
        if portfolio._daily_stock_losses.get(sym, 0) >= config.MAX_LOSSES_PER_STOCK: # MAX_LOSSES_PER_STOCK
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Max Losses")
            continue
        if len(portfolio.positions) >= config.MAX_OPEN_POSITIONS: # MAX_OPEN_POSITIONS
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Max Open Positions")
            continue

        # Acquisition of State Lock
        active_positions[sym] = True
        
        # Execution!
        if sym in portfolio.positions:
            continue
        opened = portfolio.open_position(sym, st.sector, signal, price, now)
        if opened:
            last_entry_minutes[sym] = now_minute
            print(f"[{now.time()}] EXECUTION: {sym} {signal} @ {price:.2f}")

    # 5. EOD Square Off
    print(f"[{now.time()}] End of Day reached. Squaring off remaining {len(portfolio.positions)} positions.")
    portfolio.close_all_eod(now)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"SPOOFER COMPLETE.")
    print(f"Ticks Injected: {tick_count}")
    print(f"Time Taken: {elapsed:.2f}s ({(tick_count/elapsed):.0f} ticks/sec)")
    print(f"Final Equity: Rs {portfolio.simulator.total_capital:.2f}")
    if len(portfolio.simulator.trade_history) > 0:
        wins = sum(1 for t in portfolio.simulator.trade_history if t.net_pnl > 0)
        print(f"Trades Taken: {len(portfolio.simulator.trade_history)} | Wins: {wins} | Losses: {len(portfolio.simulator.trade_history) - wins}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Spoofer: Hyper-speed tick injector")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file with tick data")
    args = parser.parse_args()
    
    run_offline_spoofer(Path(args.file))
