import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import xgboost as xgb

import config
import src.live.paper_trader as pt
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
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
        renko_states[sym] = LiveRenkoState(sym, sec, brick_sizes.get(sym, 0.75))
    
    sector_renko = {}
    csv_sectors = set(st.sector for st in renko_states.values())
    for sec in csv_sectors:
        sec_sym = sector_index_map.get(sec)
        if sec_sym:
            sector_renko[sec_sym] = LiveRenkoState(sec_sym, sec, brick_sizes.get(sec_sym, 0.75))

    stock_sectors = {r["symbol"]: r["sector"] for _, r in stocks.iterrows()}
    all_syms = list(renko_states.keys()) + list(sector_renko.keys())
    guard = LiveExecutionGuard(symbols=all_syms, sectors=stock_sectors)
    
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
    print("Loading models and initialized dummy portfolio...")
    b1_long = xgb.XGBClassifier(); b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
    b1_short = xgb.XGBClassifier(); b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))
    b2 = xgb.XGBRegressor();  b2.load_model(str(config.BRAIN2_MODEL_PATH))
    portfolio = pt.PaperPortfolio(pt.PAPER_CAPITAL)
    
    last_preds = {}
    last_entry_minutes = {}
    
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
        
        # 1. Process Tick
        new_bricks = st.process_tick(price, high, low, now)
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
                portfolio.update_position(sym, price, lp["brick_dir"], lp["b2c"], lp["signal"], lp["b1p"])
                exit_reason = portfolio.check_exit(sym, price, now, lp["b2c"], lp["signal"], lp["b1p"], brick_dir=lp["brick_dir"])
                if exit_reason:
                    pos = portfolio.close_position(sym, price, now, exit_reason)
                    print(f"[{now.time()}] EXIT: {sym} {lp['signal']} @ {price:.2f} | PnL: Rs {pos.get('unrealized_pnl', 0):.2f} | Reason: {exit_reason}")
                    
        # 3. Model Inference (only if new bricks formed and enough history)
        if not new_bricks or guard.buffers[sym].size < 2:
            continue
            
        sec_sym = sector_index_map.get(st.sector, "")
        sec_bdf = guard.buffers[sec_sym].to_dataframe() if sec_sym in guard.buffers else pd.DataFrame()
        bdf = compute_features_live(guard.buffers[sym].to_dataframe(), sec_bdf)
        latest = bdf.iloc[-1]
        
        # Enforce exact column order for XGBoost Brain1
        exact_brain1_cols = [
            'velocity', 'wick_pressure', 'relative_strength', 'brick_size',
            'duration_seconds', 'consecutive_same_dir', 'brick_oscillation_rate',
            'fracdiff_price', 'hurst', 'is_trending_regime', 'velocity_long',
            'trend_slope', 'rolling_range_pct', 'momentum_acceleration',
            'vwap_zscore', 'vpt_acceleration', 'squeeze_zscore', 'streak_exhaustion',
            # Phase 3: Temporal Alpha Features
            'true_gap_pct', 'time_to_form_seconds', 'volume_intensity_per_sec', 'is_opening_drive'
        ]
        feat_dict = latest.infer_objects(copy=False).fillna(0).to_dict()
        
        # Build DataFrame with exact column order
        X = pd.DataFrame([feat_dict])[exact_brain1_cols]
        
        p_long = float(b1_long.predict_proba(X)[0, 1])
        p_short = float(b1_short.predict_proba(X)[0, 1])
        
        if p_long > p_short:
            b1p = p_long
            b1d = 1
            signal = "LONG"
        else:
            b1p = p_short
            b1d = -1
            signal = "SHORT"
        
        X_m = pd.DataFrame([{
            "brain1_prob": b1p,
            "velocity": float(latest.get("velocity", 0)),
            "wick_pressure": float(latest.get("wick_pressure", 0)),
            "relative_strength": float(latest.get("relative_strength", 0)),
        }])
        b2c = float(np.clip(b2.predict(X_m)[0], 0, 100))
        sec_dir = guard.buffers[sec_sym]._buffer[-1]["direction"] if sec_sym in guard.buffers and guard.buffers[sec_sym].size > 0 else 0
        score = b2c
        rel_str = float(latest.get("relative_strength", 0))
        brick_dir = int(latest.get("direction", 0))
        
        last_preds[sym] = {
            "b1p": b1p, "b2c": b2c, "signal": signal, 
            "score": score, "rel_str": rel_str, "brick_dir": brick_dir
        }
        
        print(f"[{now.time()}] INFERENCE: {sym} | Brain1 (PROB): {b1p:.4f} | Brain2 (CONV): {b2c:.1f} | Signal: {signal}")
        
        # 4. Entry Gates (Mirroring paper_trader exactly — per-direction threshold)
        if signal == "LONG":
            entry_prob_ok = b1p >= pt.LONG_ENTRY_PROB_THRESH
        else:
            entry_prob_ok = (1 - b1p) >= pt.SHORT_ENTRY_PROB_THRESH
            
        do_log = b1p > 0.7 or (1 - b1p) > 0.7
        
        # End of Day Block (No entries after 15:00)
        no_entry = (now.hour > pt.NO_ENTRY_HOUR) or (now.hour == pt.NO_ENTRY_HOUR and now.minute >= pt.NO_ENTRY_MINUTE)
        if no_entry:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: EOD No Entry Block")
            continue
            
        # Option 2: Require Fresh Evidence (No Morning Gate Rush)
        try:
            _start_time = latest.get("brick_start_time")
            if _start_time:
                _st_dt = pd.to_datetime(_start_time)
                if _st_dt.hour < 9 or (_st_dt.hour == 9 and _st_dt.minute < 30):
                    if do_log: print(f"[{now.time()}] [DROP] {sym}: PRE_930_MOMENTUM")
                    continue
        except Exception as e:
            pass
            
        if not entry_prob_ok:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Low Prob ({b1p:.2f})")
            continue
        if b2c < pt.ENTRY_CONV_THRESH:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Low Conv ({b2c:.2f} < {pt.ENTRY_CONV_THRESH})")
            continue
        if abs(rel_str) < pt.ENTRY_RS_THRESHOLD:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Low RS ({abs(rel_str):.2f} < {pt.ENTRY_RS_THRESHOLD})")
            continue
        if float(latest.get("wick_pressure", 0)) > pt.MAX_ENTRY_WICK:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: High Wick Pressure ({latest.get('wick_pressure', 0):.2f})")
            continue
        if last_entry_minutes.get(sym) == now_minute:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Same Minute")
            continue
        
        # Whipsaw checks
        if len(st.bricks) < pt.MIN_CONSECUTIVE_BRICKS:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Not enough bricks ({len(st.bricks)} < {pt.MIN_CONSECUTIVE_BRICKS})")
            continue
        recent_bricks = st.bricks[-pt.MIN_CONSECUTIVE_BRICKS:]
        recent_dirs = [b["direction"] for b in recent_bricks]
        expected_dir = 1 if signal == "LONG" else -1
        if not all(d == expected_dir for d in recent_dirs):
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Whipsaw mismatch (Dirs: {recent_dirs} | Expected: {expected_dir})")
            continue
            
        # Gate: Anti-FOMO Streak Limit
        streak_count = int(latest.get("consecutive_same_dir", 0))
        if streak_count >= 7:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: FOMO Streak Limit ({streak_count} >= 7)")
            continue
            
        today_date = now.date()
        today_bricks = sum(1 for b in st.bricks if pd.to_datetime(b["brick_timestamp"]).date() == today_date)
        if today_bricks < pt.MIN_BRICKS_TODAY:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Not enough bricks today ({today_bricks} < {pt.MIN_BRICKS_TODAY})")
            continue
            
        if portfolio._daily_stock_losses.get(sym, 0) >= pt.MAX_LOSSES_PER_STOCK:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Max Losses")
            continue
        if len(portfolio.positions) >= pt.MAX_OPEN_POSITIONS:
            if do_log: print(f"[{now.time()}] [DROP] {sym}: Max Open Positions")
            continue
        
        # Execution!
        opened = portfolio.open_position(sym, st.sector, signal, price, now)
        if opened:
            last_entry_minutes[sym] = now_minute
            print(f"[{now.time()}] EXECUTION: {sym} {signal} @ {price:.2f}")

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
