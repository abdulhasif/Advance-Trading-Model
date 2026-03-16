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
import joblib
import src.live.paper_trader as pt
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

    # Override PaperPortfolio logging paths
    pt.SIGNAL_LOG = SPOOFER_DIR / "spoofer_signals.csv"
    pt.TRADE_LOG = SPOOFER_DIR / "spoofer_trades.csv"
    pt.DAILY_LOG = SPOOFER_DIR / "spoofer_daily.csv"
    pt.LIVE_PNL_FILE = SPOOFER_DIR / "spoofer_pnl.json"

    print(f"Loading Spoofer CSV: {csv_file}")
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if "timestamp" in df.columns:
        ts_col = pd.to_datetime(df["timestamp"])
        if ts_col.dt.tz is not None:
            df["timestamp"] = ts_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        else:
            df["timestamp"] = ts_col
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
    sim_date = df["timestamp"].iloc[0].date() if not df.empty else None
    
    # FIX: Include indices in sectors mapping so they can be warmed up correctly
    stock_sectors = {r["symbol"]: r["sector"] for _, r in stocks.iterrows()}
    index_sectors = {r["symbol"]: r["sector"] for _, r in indices.iterrows()}
    all_sectors = {**stock_sectors, **index_sectors}
    
    guard = LiveExecutionGuard(symbols=all_syms, sectors=all_sectors, before_date=sim_date)
    
    print("Pre-loading historical buffers...")
    guard.warm_up_all()

    for sym, buf in guard.buffers.items():
        for b in buf._buffer:
            if hasattr(b["brick_timestamp"], "tzinfo") and b["brick_timestamp"].tzinfo:
                if b["brick_timestamp"].tzinfo is not None:
                    b["brick_timestamp"] = b["brick_timestamp"].tz_convert("Asia/Kolkata").tz_localize(None)
                else:
                    b["brick_timestamp"] = b["brick_timestamp"].replace(tzinfo=None)

    for sym, st in list(renko_states.items()) + list(sector_renko.items()):
        if sym in guard.buffers and not guard.buffers[sym].to_dataframe().empty:
            bdf = guard.buffers[sym].to_dataframe()
            st.renko_level = bdf["brick_close"].iloc[-1]
            st.brick_start_time = bdf["brick_timestamp"].iloc[-1]

    # 2. Load Models
    print("Loading ML models...")
    try:
        if config.USE_CALIBRATED_MODELS:
            b1_long  = joblib.load(config.BRAIN1_CALIBRATED_LONG_PATH)
            b1_short = joblib.load(config.BRAIN1_CALIBRATED_SHORT_PATH)
            print("Using CALIBRATED Brain1 models.")
        else:
            b1_long  = xgb.XGBClassifier()
            b1_long.load_model(str(config.BRAIN1_MODEL_LONG_PATH))
            b1_short = xgb.XGBClassifier()
            b1_short.load_model(str(config.BRAIN1_MODEL_SHORT_PATH))
            print("Using RAW Brain1 (.json) models.")
            
        b2 = xgb.XGBRegressor()
        b2.load_model(str(config.BRAIN2_MODEL_PATH))
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    from src.core.risk import RiskFortress
    risk = RiskFortress()
    portfolio = pt.PaperPortfolio(pt.PAPER_CAPITAL)
    
    last_preds = {}
    last_entry_minutes = {}
    active_positions = {}
    exit_brick_indices = {}  # Track exit brick count for cooldown
    
    # ── Calculate Entry Lock Boundary ──────────────────────────
    morning_lock_min = config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES
    morning_lock_hour = config.MARKET_OPEN_HOUR + (morning_lock_min // 60)
    morning_lock_min %= 60
    
    print("=" * 60)
    print("SPOOFER INJECTION STARTED")
    print("=" * 60)
    
    tick_count = 0
    start_time = time.time()
    
    _already_squared_off = False

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
                    exit_brick_indices[sym] = len(st.bricks)
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

            # Brain 1: Directional Probability
            if config.USE_CALIBRATED_MODELS:
                p_long = float(b1_long.predict_proba(X)[0][1])
                p_short = float(b1_short.predict_proba(X)[0][1])
            else:
                dmat = xgb.DMatrix(X)
                p_long = float(b1_long.predict(dmat)[0])
                p_short = float(b1_short.predict(dmat)[0])
            
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
            
            # ── Entry Gates (Sync with engine.py) ───────────────────
            # Gate 1: Probability
            thresh = config.LONG_ENTRY_PROB_THRESH if signal == "LONG" else config.SHORT_ENTRY_PROB_THRESH
            if not config.USE_CALIBRATED_MODELS:
                thresh = config.RAW_LONG_ENTRY_PROB_THRESH if signal == "LONG" else config.RAW_SHORT_ENTRY_PROB_THRESH
            
            if b1p < thresh:
                continue

            # Gate 2: Brain 2 Conviction
            if b2c < config.ENTRY_CONV_THRESH:
                continue

            # Gate 3: Soft Veto (Sector Alignment)
            is_vetoed = (b1d != sec_dir)
            if is_vetoed and b2c < config.VETO_BYPASS_CONV:
                continue

            # Gate 4: RS Anchor
            if b2c < config.VETO_BYPASS_CONV:
                if signal == "LONG" and rel_str < config.ENTRY_RS_THRESHOLD:
                    continue
                if signal == "SHORT" and rel_str > -config.ENTRY_RS_THRESHOLD:
                    continue

            # Gate 5: Wick Trap
            wick_p = float(feat_dict.get("wick_pressure", 0))
            if b2c < config.VETO_BYPASS_CONV and wick_p > config.MAX_ENTRY_WICK:
                continue

            # Gate 6: VWAP Exhaustion
            z_vwap = float(feat_dict.get("vwap_zscore", 0))
            if b2c < config.VETO_BYPASS_CONV:
                if (signal == "LONG" and z_vwap > config.MAX_VWAP_ZSCORE) or \
                   (signal == "SHORT" and z_vwap < -config.MAX_VWAP_ZSCORE):
                    continue

            # Gate 7: Anti-FOMO Streak Limit
            if b2c < config.VETO_BYPASS_CONV:
                streak_count = int(feat_dict.get("consecutive_same_dir", 0))
                if streak_count >= config.STREAK_LIMIT:
                    continue

            # Gate 8: Brick Cooldown
            last_exit_idx = exit_brick_indices.get(sym, -1000)
            if (len(st.bricks) - last_exit_idx) < config.BRICK_COOLDOWN:
                continue

            # Gate 9: Daily Stock Loss Protection
            if portfolio._daily_stock_losses.get(sym, 0) >= config.MAX_LOSSES_PER_STOCK:
                continue

            # ── Gate 1.5: Morning Entry Lock ────────────────────────
            if now.hour < morning_lock_hour or (now.hour == morning_lock_hour and now.minute < morning_lock_min):
                continue

            # Skip if already used this minute
            if last_entry_minutes.get(sym) == now_minute:
                continue

            # Gate 11: Already in position
            if sym in portfolio.positions:
                continue

            # Gate 12: Position cap
            if len(portfolio.positions) >= config.MAX_OPEN_POSITIONS:
                continue

            # Passed all gates! EXECUTE IMMEDIATELY (Sync with engine.py)
            execution_price = price
            execution_time = now
            s_st = renko_states[sym]
            
            sl_price = 0.0
            if signal == "LONG":
                sl_price = execution_price - (config.STRUCTURAL_REVERSAL_BRICKS * s_st.brick_size)
            elif signal == "SHORT":
                sl_price = execution_price + (config.STRUCTURAL_REVERSAL_BRICKS * s_st.brick_size)
            
            opened = portfolio.open_position(sym, s_st.sector, signal, execution_price, sl_price, execution_time)
            if opened:
                last_entry_minutes[sym] = now_minute
                print(f"[{execution_time.time()}] EXECUTION: {sym} {signal} @ {execution_price:.2f} (Score: {score:.2f})")
                portfolio.log_signal(execution_time, sym, s_st.sector, signal, b1p, b2c, rel_str, score, execution_price, "ENTRY")

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
