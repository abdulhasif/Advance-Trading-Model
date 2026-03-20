import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import xgboost as xgb
import logging

import config
import keras
from src.live.upstox_simulator import UpstoxSimulator
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.live.execution_guard import LiveExecutionGuard

# Redirection setup
SPOOFER_DIR = config.PROJECT_ROOT / "spoofer_logs"
SPOOFER_DIR.mkdir(exist_ok=True)

def run_offline_spoofer(csv_file: Path):
    # Redirect logs so we don't pollute the real paper trading results
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(SPOOFER_DIR / "spoofer_trader.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("spoofer")

    # Redirect logs
    logger = logging.getLogger("spoofer")

    print(f"Loading Spoofer CSV: {csv_file}")
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if "timestamp" in df.columns:
        # Aggressively strip all timezones to naive datetime to prevent subtraction crashes
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp")
    
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

    # Contamination Shield
    sim_ts = df["timestamp"].iloc[0] if not df.empty else None
    sim_date = sim_ts.date() if sim_ts else "UNKNOWN"
    
    # FIX: Include indices in sectors mapping so they can be warmed up correctly
    stock_sectors = {r["symbol"]: r["sector"] for _, r in stocks.iterrows()}
    index_sectors = {r["symbol"]: r["sector"] for _, r in indices.iterrows()}
    all_sectors = {**stock_sectors, **index_sectors}
    
    guard = LiveExecutionGuard(symbols=all_syms, sectors=all_sectors, before_ts=sim_ts)
    
    print("Pre-loading historical buffers...")
    guard.warm_up_all()

    for sym, buf in guard.buffers.items():
        for b in buf._buffer:
            if isinstance(b["brick_timestamp"], pd.Timestamp) and b["brick_timestamp"].tzinfo is not None:
                b["brick_timestamp"] = b["brick_timestamp"].tz_localize(None)

    for sym, st in list(renko_states.items()) + list(sector_renko.items()):
        if sym in guard.buffers and not guard.buffers[sym].to_dataframe().empty:
            bdf = guard.buffers[sym].to_dataframe()
            st.renko_level = bdf["brick_close"].iloc[-1]
            st.brick_start_time = bdf["brick_timestamp"].iloc[-1]

    # 4. Load Models (CNN)
    os.environ["KERAS_BACKEND"] = "torch"
    try:
        brain1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
        brain1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
        scaler   = joblib.load(str(config.BRAIN1_SCALER_PATH))
        brain2 = xgb.Booster(); brain2.load_model(str(config.BRAIN2_MODEL_PATH))
        print("Models loaded (CNN Baseline + Scaler)")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    # Init Simulator
    simulator = UpstoxSimulator(starting_capital=config.STARTING_CAPITAL)
    
    last_preds = {}
    last_entry_minutes = {}
    active_positions = {}
    
    last_preds = {}
    last_entry_minutes = {}
    active_positions = {}
    
    print("=" * 60)
    print("SPOOFER INJECTION STARTED")
    print("=" * 60)
    
    tick_count = 0
    start_time = time.time()
    
    pending_signals = []
    last_minute = None
    _already_squared_off = False

    for idx, row in df.iterrows():
        sym = row["symbol"]
        if sym not in renko_states:
            continue
            
        now = row["timestamp"]
        price = float(row["ltp"])
        high = float(row.get("high", price))
        low = float(row.get("low", price))
        
        # Priority Queue Logic
        current_minute = now.replace(second=0, microsecond=0)
        if last_minute is not None and current_minute > last_minute:
            if pending_signals:
                pending_signals.sort(key=lambda x: x["score"], reverse=True)
                for sig_data in pending_signals:
                    s_sym = sig_data["symbol"]
                    s_signal = sig_data["signal"]
                    s_price = sig_data["price"]
                    s_now = sig_data["now"]
                    s_st = renko_states[s_sym]
                    
                    if s_sym not in simulator.active_trades:
                        sl_price = 0.0
                        if s_signal == "LONG":
                            sl_price = s_price - (config.STRUCTURAL_REVERSAL_BRICKS * s_st.brick_size)
                        elif s_signal == "SHORT":
                            sl_price = s_price + (config.STRUCTURAL_REVERSAL_BRICKS * s_st.brick_size)
                        
                        req = simulator.place_order(s_sym, "BUY" if s_signal=="LONG" else "SELL", 1, s_price, sl_price, s_now)
                        if req and req.state != "REJECTED":
                            simulator.fill_pending_order(s_sym, s_now)
                            print(f"[{s_now.time()}] EXECUTION: {s_sym} {s_signal} @ {s_price:.2f} (Score: {sig_data['score']:.2f})")
            pending_signals = []
        last_minute = current_minute

        tick_count += 1
        st = renko_states[sym]
        now_minute = f"{now.hour:02d}:{now.minute:02d}"
        v = float(row.get("volume", 0))

        # ── Intraday Auto-Square-Off (synced with engine.py) ──────────
        if not _already_squared_off and (now.hour > config.EOD_SQUARE_OFF_HOUR or \
           (now.hour == config.EOD_SQUARE_OFF_HOUR and now.minute >= config.EOD_SQUARE_OFF_MIN)):
            if len(portfolio.positions) > 0:
                logger.warning(f"{config.EOD_SQUARE_OFF_HOUR}:{config.EOD_SQUARE_OFF_MIN:02d} - Auto-Square-Off for all open positions.")
                portfolio.close_all_eod(now)
            _already_squared_off = True

        # ── Process sector index ticks from CSV (if present) ──────────
        if sym in sector_renko:
            new_bricks = sector_renko[sym].process_tick(price, high, low, now, volume=v)
            for b in new_bricks:
                guard.buffers[sym].append(b)
            continue  # Indices are not traded, skip entry logic

        # 1. Process Tick
        new_bricks = st.process_tick(price, high, low, now, volume=v)
        for b in new_bricks:
            guard.buffers[sym].append(b)
            
        # 2. Update existing positions
        if sym in portfolio.positions:
            portfolio.positions[sym]["last_price"] = price
            if sym in last_preds:
                lp = last_preds[sym]
                has_new_bricks = len(new_bricks) > 0
                current_brick_dir = new_bricks[-1]["direction"] if has_new_bricks else lp.get("brick_dir", 0)
                
                portfolio.update_position(sym, price, current_brick_dir, lp["b2c"], lp["signal"], lp["b1p"], new_bricks_formed=has_new_bricks)
                exit_reason = portfolio.check_exit(sym, price, now, lp["b2c"], lp["signal"], lp["b1p"], brick_dir=current_brick_dir)
                if exit_reason:
                    pos = portfolio.close_position(sym, price, now, exit_reason)
                    portfolio.log_signal(now, sym, st.sector, lp["signal"], lp["b1p"], lp["b2c"], lp["rel_str"], lp["score"], price, "EXIT", exit_reason)
                    
                    if hasattr(portfolio.simulator, "trade_history") and len(portfolio.simulator.trade_history) > 0:
                        last_trade = portfolio.simulator.trade_history[-1]
                        if last_trade.symbol == sym and last_trade.net_pnl <= 0:
                            portfolio._daily_stock_losses[sym] = portfolio._daily_stock_losses.get(sym, 0) + 1

                    if sym in active_positions: del active_positions[sym]
                    # print(f"[{now.time()}] EXIT: {sym} {lp['signal']} @ {price:.2f} | Reason: {exit_reason}")
                    
        # 3. Model Inference (ONLY if new brick formed)
        if new_bricks:
            # Use efficiently sized rolling buffer instead of growing list
            st_bdf = guard.buffers[sym].to_dataframe()
            if st_bdf.empty:
                continue
                
            # Get sector state for Relative Strength from its buffer
            sec_sym = sector_index_map.get(st.sector, "")
            sec_bricks = guard.buffers[sec_sym].to_dataframe() if sec_sym in guard.buffers else pd.DataFrame()
            
            # Compute features using constant-sized dataframes
            latest_full = compute_features_live(st_bdf, sec_bricks)
            latest = latest_full.tail(1)
            
            feat_dict = latest.infer_objects(copy=False).fillna(0).to_dict('records')[0]
            
            # Use config.FEATURE_COLS to ensure identical order as training
            try:
                X = pd.DataFrame([feat_dict])[config.FEATURE_COLS]
            except KeyError as e:
                # Handle missing features by adding them as 0
                for col in config.FEATURE_COLS:
                    if col not in feat_dict:
                        feat_dict[col] = 0.0
                X = pd.DataFrame([feat_dict])[config.FEATURE_COLS]

            # Brain 1: Directional Probability (3D CNN + Scaling)
            win_df = guard.buffers[sym].to_dataframe().tail(config.CNN_WINDOW_SIZE)
            if len(win_df) < config.CNN_WINDOW_SIZE:
                 continue
            
            # Apply scaling
            feats_2d = win_df[config.FEATURE_COLS].fillna(0)
            scaled_2d = scaler.transform(feats_2d)
            feat_3d   = np.array([scaled_2d], dtype=np.float32)
            
            p_long  = float(brain1_long.predict(feat_3d, verbose=0)[0])
            p_short = float(brain1_short.predict(feat_3d, verbose=0)[0])
            
            b1p = 0.0
            signal = "FLAT"
            b1d = 0
            # Static Threshold Selection
            thresh_long  = config.LONG_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
            thresh_short = config.SHORT_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH

            long_ok  = p_long  >= thresh_long
            short_ok = p_short >= thresh_short

            if long_ok and short_ok:
                if p_long >= p_short:
                    signal, b1p, b1d = "LONG", p_long, 1
                else:
                    signal, b1p, b1d = "SHORT", p_short, -1
            elif long_ok:
                signal, b1p, b1d = "LONG", p_long, 1
            elif short_ok:
                signal, b1p, b1d = "SHORT", p_short, -1

            if signal == "FLAT":
                rel_str_val = float(feat_dict.get("relative_strength", 0))
                portfolio.log_signal(now, sym, st.sector, signal, max(p_long, p_short), 0.0, rel_str_val, 0.0, price, "SKIP", "LOW_PROB")
                continue

            # Build the feature matrix for Brain 2 dynamically from config.BRAIN2_FEATURES
            meta_feat_dict = {"brain1_prob": b1p}
            for f_name in config.BRAIN2_FEATURES:
                if f_name == "brain1_prob": continue
                meta_feat_dict[f_name] = float(feat_dict.get(f_name, 0))
            
            X_m = pd.DataFrame([meta_feat_dict])
            b2c = float(np.clip(b2.predict(X_m)[0], 0, config.TARGET_CLIPPING_BPS))
            
            sec_dir = 0
            if sec_sym in guard.buffers and guard.buffers[sec_sym].size > 0:
                sec_dir = int(guard.buffers[sec_sym]._buffer[-1]["direction"])
            
            score = risk.score_signal(b1p, b2c, b1d, sec_dir)
            rel_str = float(feat_dict.get("relative_strength", 0))
            brick_dir = int(feat_dict.get("direction", 0))
            
            last_preds[sym] = {
                "b1p": b1p, "b2c": b2c, "signal": signal, 
                "score": score, "rel_str": rel_str, "brick_dir": brick_dir
            }
            
            # ══════════════════════════════════════════════════════════
            #  ENTRY GATES — Exact mirror of engine.py gate ordering
            # ══════════════════════════════════════════════════════════

            # Gate 0: Penny Stock Filter (synced with engine.py)
            if price < config.MIN_PRICE_FILTER:
                continue

            # Gate: Time Gate (morning lock + no new entry cutoff)
            morning_lock_min = config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES
            morning_lock_hour = config.MARKET_OPEN_HOUR + (morning_lock_min // 60)
            morning_lock_min %= 60
            
            is_too_early = (now.hour < morning_lock_hour) or (now.hour == morning_lock_hour and now.minute < morning_lock_min)
            is_too_late = (now.hour > config.NO_NEW_ENTRY_HOUR) or (now.hour == config.NO_NEW_ENTRY_HOUR and now.minute >= config.NO_NEW_ENTRY_MIN)
            
            if is_too_early or is_too_late:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "TIME_GATE")
                continue

            # Gate 1: Entry Probability (Static Threshold)
            thresh = config.LONG_ENTRY_PROB_THRESH if signal == "LONG" else config.SHORT_ENTRY_PROB_THRESH
            if not config.USE_CALIBRATED_MODELS:
                thresh = config.RAW_LONG_ENTRY_PROB_THRESH if signal == "LONG" else config.RAW_SHORT_ENTRY_PROB_THRESH

            entry_prob_ok = b1p >= thresh
            if not entry_prob_ok:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "LOW_PROB_ENTRY")
                continue

            # Gate 2: Conviction
            if b2c < config.ENTRY_CONV_THRESH:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "LOW_CONVICTION")
                continue

            # Gate 3: Soft Veto (sector alignment) — with conviction bypass
            is_vetoed = False
            if b2c < config.VETO_BYPASS_CONV:
                if signal == "LONG" and rel_str < -config.SOFT_VETO_THRESHOLD:
                    is_vetoed = True
                if signal == "SHORT" and rel_str > config.SOFT_VETO_THRESHOLD:
                    is_vetoed = True
            if is_vetoed:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "SECTOR_VETO")
                continue

            # Gate 4: RS Anchor — only trade sector leaders/laggards (bypass if high conviction)
            if b2c < config.VETO_BYPASS_CONV:
                if signal == "LONG" and rel_str < config.ENTRY_RS_THRESHOLD:
                    portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "LOW_RS")
                    continue
                if signal == "SHORT" and rel_str > -config.ENTRY_RS_THRESHOLD:
                    portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "LOW_RS")
                    continue

            # Gate 5: Wick Trap
            wick_p = float(feat_dict.get("wick_pressure", 0))
            if wick_p > config.MAX_ENTRY_WICK:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "WICK_PRESSURE")
                continue

            # Gate 6: VWAP Exhaustion
            z_vwap = float(feat_dict.get("vwap_zscore", 0))
            if signal == "LONG" and z_vwap > config.MAX_VWAP_ZSCORE:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "VWAP_EXHAUSTION")
                continue
            if signal == "SHORT" and z_vwap < -config.MAX_VWAP_ZSCORE:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "VWAP_EXHAUSTION")
                continue

            # Gate 7: Whipsaw Guard — consecutive bricks
            if len(st.bricks) < config.MIN_CONSECUTIVE_BRICKS:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "WHIPSAW_BRICKS")
                continue
            recent_bricks = st.bricks[-config.MIN_CONSECUTIVE_BRICKS:]
            recent_dirs = [b["direction"] for b in recent_bricks]
            expected_dir = 1 if signal == "LONG" else -1
            if not all(d == expected_dir for d in recent_dirs):
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "WHIPSAW_MIXED_DIR")
                continue

            # Gate 8: Streak / FOMO limit
            streak_count = int(feat_dict.get("consecutive_same_dir", 0))
            if streak_count >= config.STREAK_LIMIT:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "STREAK_LIMIT")
                continue

            # Gate 9: Daily stock loss limit
            if portfolio._daily_stock_losses.get(sym, 0) >= config.MAX_LOSSES_PER_STOCK:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "DAILY_LOSS_LIMIT")
                continue

            # Gate 10: Same-minute duplicate check
            if last_entry_minutes.get(sym) == now_minute:
                continue

            # Gate 11: Already in position
            if sym in active_positions or sym in portfolio.positions:
                continue

            # Gate 12: Position cap
            if len(portfolio.positions) >= config.MAX_OPEN_POSITIONS:
                portfolio.log_signal(now, sym, st.sector, signal, b1p, b2c, rel_str, score, price, "SKIP", "MAX_POSITIONS")
                continue

            # Avoid adding multiple signals for same symbol in same minute to the queue
            if not any(s["symbol"] == sym for s in pending_signals):
                pending_signals.append({
                    "symbol": sym, "signal": signal, "price": price, "now": now,
                    "score": score, "b1p": b1p, "b2c": b2c, "rel_str": rel_str
                })

    print(f"[{now.time()}] End of Day reached. Squaring off remaining {len(portfolio.positions)} positions.")
    portfolio.close_all_eod(now)
    
    # Generate and print summary
    summary = portfolio.record_daily_summary(str(sim_date))

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"SPOOFER SUMMARY | Date: {sim_date}")
    print(f"Total Trades: {summary['trades']}")
    print(f"Wins: {summary['wins']} | Losses: {summary['losses']} | Win Rate: {summary['win_rate']}%")
    print(f"Realized PnL: Rs {summary['realized_pnl']:+.2f}")
    print(f"Final Equity: Rs {portfolio.simulator.total_capital:.2f}")
    print(f"Time Taken: {elapsed:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Spoofer")
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    run_offline_spoofer(Path(args.file))
