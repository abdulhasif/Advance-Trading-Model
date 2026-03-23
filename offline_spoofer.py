"""
offline_spoofer.py - Event-Driven High-Fidelity Simulation Engine
================================================================
Mirrors src/live/engine.py exactly, using SimClock and T+1 fill logic.
"""

import sys
import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import logging
import pandas as pd
import numpy as np
import joblib
import keras
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import gc

import config
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.live.upstox_simulator import UpstoxSimulator
from src.live.execution_guard import LiveExecutionGuard
from src.core.strategy import check_entry_gates, check_exit_conditions

# Redirection setup
SPOOFER_DIR = config.PROJECT_ROOT / "spoofer_logs"
SPOOFER_DIR.mkdir(exist_ok=True)

# -- Sim Clock Abstraction ---------------------------------------------------
class SimClock:
    """Eliminates system-time calls for deterministic simulation."""
    _current_time = None

    @classmethod
    def set(cls, ts: datetime):
        cls._current_time = ts

    @classmethod
    def now(cls) -> datetime:
        return cls._current_time

# -- Metrics Tracker ---------------------------------------------------------
class SummaryStats:
    def __init__(self):
        self.trades = []
        self.equity_curve = [config.STARTING_CAPITAL]
        self.max_equity = config.STARTING_CAPITAL
        self.max_drawdown = 0.0

    def add_trade(self, trade):
        self.trades.append(trade)
        new_equity = self.equity_curve[-1] + trade.net_pnl
        self.equity_curve.append(new_equity)
        
        if new_equity > self.max_equity:
            self.max_equity = new_equity
        
        dd = (self.max_equity - new_equity) / self.max_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def report(self):
        if not self.trades:
            return "No trades executed."
        
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        total_pnl = sum(t.net_pnl for t in self.trades)
        win_rate = (len(wins) / len(self.trades)) * 100
        
        profit_factor = sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses)) if losses else float('inf')
        
        return (
            f"\n--- SIMULATION SUMMARY ---\n"
            f"Total Trades:    {len(self.trades)}\n"
            f"Win Rate:        {win_rate:.2f}%\n"
            f"Total Net PnL:   Rs {total_pnl:.2f}\n"
            f"Max Drawdown:    {self.max_drawdown*100:.2f}%\n"
            f"Profit Factor:   {profit_factor:.2f}\n"
            f"Final Capital:   Rs {config.STARTING_CAPITAL + total_pnl:.2f}\n"
        )

# -- Data Feeder -------------------------------------------------------------
class CsvTickFeeder:
    """Mimics Upstox WebSocket by grouping ticks by timestamp."""
    def __init__(self, csv_path: Path):
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        self.df = df.sort_values("timestamp")
        self.groups = self.df.groupby("timestamp")
        self.timestamps = sorted(self.df["timestamp"].unique())
        self.idx = 0

    def next_event(self):
        if self.idx >= len(self.timestamps):
            return None, None
        ts = self.timestamps[self.idx]
        group = self.groups.get_group(ts)
        self.idx += 1
        return ts, group.to_dict('records')

# -- T+1 Execution Pipeline --------------------------------------------------
class T1ExecutionPipeline:
    def __init__(self, simulator: UpstoxSimulator):
        self.simulator = simulator
        self.pending_buffer = {} # symbol -> order_request

    def place_sim_order(self, symbol, side, qty, price, sl_price, ts):
        """Buffer order for T+1 fill logic."""
        # Simulating live acquire_lock behavior implicitly by dropping if already pending
        if symbol in self.pending_buffer:
            return None
        
        # Mirror slippage from engine.py
        slippage = config.T1_SLIPPAGE_PCT
        fill_price = price * (1.0 + slippage) if side == "BUY" else price * (1.0 - slippage)
        
        self.pending_buffer[symbol] = {
            "side": side, "qty": qty, "price": fill_price, "sl_price": sl_price, "ts": ts
        }
        return True

    def process_fills(self, symbol, current_price, ts):
        """Fill at the NEXT available tick for that symbol."""
        if symbol in self.pending_buffer:
            order = self.pending_buffer.pop(symbol)
            # Fill with the same price as generated (slippage already added)
            # This is because we assume the 'next tick' is the one we just received
            req = self.simulator.place_order(
                symbol, order["side"], order["qty"], order["price"], order["sl_price"], ts
            )
            if req and req.state != "REJECTED":
                self.simulator.fill_pending_order(symbol, ts)
                return order
        return None

# -- Main Simulator ----------------------------------------------------------
class EventDrivenSim:
    def __init__(self, csv_file: Path):
        self.feeder = CsvTickFeeder(csv_file)
        self.simulator = UpstoxSimulator(starting_capital=config.STARTING_CAPITAL)
        self.t1_pipeline = T1ExecutionPipeline(self.simulator)
        self.stats = SummaryStats()
        
        self.logger = logging.getLogger("Sim")
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(sh)

        # Universe & State
        universe = pd.read_csv(config.UNIVERSE_CSV)
        stocks = universe[~universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])]
        self.sector_map = {r["symbol"]: r["sector"] for _, r in universe.iterrows()}
        indices = universe[universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])]
        self.index_symbols = indices["symbol"].tolist()
        self.sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}

        # Models
        self.b1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
        self.b1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
        self.scaler = joblib.load(str(config.BRAIN1_SCALER_PATH))
        self.b2 = xgb.Booster(); self.b2.load_model(str(config.BRAIN2_MODEL_PATH))

        # Renko & Buffers
        start_ts = self.feeder.timestamps[0]
        self.exec_guard = LiveExecutionGuard(symbols=universe["symbol"].tolist(), sectors=self.sector_map, before_ts=start_ts)
        self.exec_guard.warm_up_all()

        self.renko_states = {}
        for sym in universe["symbol"].tolist():
            sec = self.sector_map.get(sym, "UNKNOWN")
            # For simplicity, fallback to NATR brick
            brick = 500 * config.NATR_BRICK_PERCENT
            if sym in self.exec_guard.buffers and not self.exec_guard.buffers[sym].to_dataframe().empty:
                brick = self.exec_guard.buffers[sym].to_dataframe()["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
            
            st = LiveRenkoState(sym, sec, brick)
            if sym in self.exec_guard.buffers:
                st.bricks = list(self.exec_guard.buffers[sym]._buffer)
            self.renko_states[sym] = st

        self.last_entry_minutes = {}
        self.inference_cache = {}
        self.all_signals = []

    def run(self):
        print("Starting deterministic simulation...")
        _already_squared_off = False

        while True:
            ts, ticks = self.feeder.next_event()
            if ts is None: break
            
            SimClock.set(ts)
            now = SimClock.now()
            now_minute = now.replace(second=0, microsecond=0)

            # EOD Check
            if not _already_squared_off and (now.hour > config.EOD_SQUARE_OFF_HOUR or \
               (now.hour == config.EOD_SQUARE_OFF_HOUR and now.minute >= config.EOD_SQUARE_OFF_MIN)):
                active_orders = list(self.simulator.active_trades.values())
                self.simulator.square_off_all(now)
                for order in active_orders:
                    self.stats.add_trade(order)
                _already_squared_off = True

            # Process each tick in the timestamp group
            executable_signals = []

            for t in ticks:
                sym = t["symbol"]
                price = float(t["ltp"])
                vol = float(t.get("volume", 0))
                high = float(t.get("high", price))
                low = float(t.get("low", price))

                st = self.renko_states.get(sym)
                if not st: continue

                # 1. Update active price & process fills
                self.simulator.update_active_price(sym, price)
                filled = self.t1_pipeline.process_fills(sym, price, now)
                if filled:
                    self.logger.info(f"[{now.time()}] FILL: {sym} {filled['side']} @ {filled['price']:.2f}")

                # 2. Process Renko
                prev_cnt = len(st.bricks)
                st.process_tick(price, high, low, now, volume=vol)
                new_brick_formed = len(st.bricks) > prev_cnt
                
                if new_brick_formed:
                    new_brick = st.bricks[-1]
                    self.exec_guard.buffers[sym].append(new_brick)
                
                # 3. Skip indicators for indices except for providing data
                if sym in self.index_symbols: continue

                # Fast fail if insufficient bricks to avoid pandas overhead on every tick
                if len(st.bricks) < config.CNN_WINDOW_SIZE: continue

                # 4. Strategy Inference & Gates
                # We need features and props for both entry and exit checks
                if new_brick_formed or sym not in self.inference_cache:
                    bdf = self.exec_guard.buffers[sym].to_dataframe()
    
                    sec_sym = self.sector_index_map.get(st.sector)
                    sec_bdf = self.exec_guard.buffers[sec_sym].to_dataframe() if sec_sym in self.exec_guard.buffers else pd.DataFrame()
                    
                    feat_df = compute_features_live(bdf, sec_bdf)
                    latest = feat_df.iloc[-1]
                    
                    # CNN Props
                    win_df = bdf.tail(config.CNN_WINDOW_SIZE)
                    feats_2d = feat_df[config.FEATURE_COLS].tail(config.CNN_WINDOW_SIZE).fillna(0)
                    scaled_2d = self.scaler.transform(feats_2d)
                    feat_3d = np.array([scaled_2d], dtype=np.float32)
                    
                    p_long = float(self.b1_long.predict(feat_3d, verbose=0).item())
                    p_short = float(self.b1_short.predict(feat_3d, verbose=0).item())
    
                    # Brain 2
                    signal_str, b1p, b1d = "FLAT", 0.0, 0
                    if p_long >= p_short: signal_str, b1p, b1d = "LONG", p_long, 1
                    else: signal_str, b1p, b1d = "SHORT", p_short, -1
    
                    b2_vals = []
                    for f_name in config.BRAIN2_FEATURES:
                        if f_name == "brain1_prob_long": b2_vals.append(p_long)
                        elif f_name == "brain1_prob_short": b2_vals.append(p_short)
                        elif f_name == "trade_direction": b2_vals.append(float(b1d))
                        else: b2_vals.append(float(latest.get(f_name, 0)))
                    
                    dm_meta = xgb.DMatrix([b2_vals], feature_names=config.BRAIN2_FEATURES)
                    b2c = float(np.clip(self.b2.predict(dm_meta)[0], 0, config.TARGET_CLIPPING_BPS))
                    
                    self.inference_cache[sym] = (latest, p_long, p_short, signal_str, b1p, b2c)
                else:
                    latest, p_long, p_short, signal_str, b1p, b2c = self.inference_cache[sym]

                # EXITS
                if sym in self.simulator.active_trades:
                    order = self.simulator.active_trades[sym]
                    exit_reason = check_exit_conditions(
                        order.side, order.entry_price, price, st.brick_size, b2c, p_long, p_short
                    )
                    if exit_reason:
                        self.simulator.close_position(sym, price, now, exit_reason)
                        self.stats.add_trade(order)
                        self.logger.info(f"[{now.time()}] EXIT: {sym} @ {price:.2f} | PnL: {order.net_pnl:.2f} | Reason: {exit_reason}")
                
                # ENTRIES (Only on new bricks)
                if new_brick_formed:
                    sec_dir = 0
                    if sec_sym in self.renko_states:
                        sec_dir = self.renko_states[sec_sym].bricks[-1]["direction"] if self.renko_states[sec_sym].bricks else 0
                    
                    score = (b1p * 10) + (b2c / 10) # Dummy score for ranking
                    
                    is_in = (sym in self.simulator.active_trades) or (sym in self.t1_pipeline.pending_buffer)
                    stock_losses = sum(1 for t in self.simulator.trade_history if t.symbol == sym and t.net_pnl < 0)
                    
                    recent_dirs = [rb["direction"] for rb in st.bricks[-config.MIN_CONSECUTIVE_BRICKS:]]
                    
                    gate_pass, reason = check_entry_gates(
                        sym, now, price, b1p, b2c, signal_str, float(latest.get("relative_strength", 0)),
                        float(latest.get("wick_pressure", 0)), float(latest.get("vwap_zscore", 0)),
                        int(latest.get("consecutive_same_dir", 0)), int(latest.get("direction", 0)),
                        recent_dirs, stock_losses, len(self.simulator.active_trades), is_in
                    )
                    
                    if gate_pass:
                        if self.last_entry_minutes.get(sym) != now_minute:
                            executable_signals.append({
                                "symbol": sym, "side": "BUY" if signal_str == "LONG" else "SELL",
                                "qty": 1, "price": price, "score": score, "st": st
                            })
                            self.last_entry_minutes[sym] = now_minute
            
            # Execute ranked signals
            if executable_signals:
                executable_signals.sort(key=lambda x: x["score"], reverse=True)
                for sig in executable_signals:
                    s_sym = sig["symbol"]
                    s_st = sig["st"]
                    sl_dist = config.STRUCTURAL_REVERSAL_BRICKS * s_st.brick_size
                    sl_price = sig["price"] - sl_dist if sig["side"] == "BUY" else sig["price"] + sl_dist
                    
                    if self.t1_pipeline.place_sim_order(s_sym, sig["side"], sig["qty"], sig["price"], sl_price, now):
                        self.logger.info(f"[{now.time()}] SIGNAL: {s_sym} {sig['side']} @ {sig['price']:.2f}")
                        self.all_signals.append({
                            "timestamp": now,
                            "symbol": s_sym,
                            "side": sig["side"],
                            "price": round(sig["price"], 2),
                            "score": round(sig["score"], 4)
                        })

        # Final Report
        print(self.stats.report())
        
        # Save Trade Log
        if self.simulator.trade_history:
            trades_data = []
            for t in self.simulator.trade_history:
                d = {
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": t.filled_at,
                    "exit_time": t.closed_at,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "exit_reason": t.exit_reason
                }
                trades_data.append(d)
                
            log_df = pd.DataFrame(trades_data)
            log_path = SPOOFER_DIR / "trade_log.csv"
            log_df.to_csv(log_path, index=False)
            print(f"Trade log saved to {log_path}")
        else:
            print("No trades to save.")

        # Save Signals Log
        if getattr(self, "all_signals", None):
            sig_df = pd.DataFrame(self.all_signals)
            sim_date = self.feeder.timestamps[0].strftime("%Y-%m-%d") if self.feeder.timestamps else "unknown"
            sig_path = SPOOFER_DIR / f"signals_{sim_date}.csv"
            sig_df.to_csv(sig_path, index=False)
            print(f"Signals log saved to {sig_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    
    sim = EventDrivenSim(Path(args.file))
    sim.run()
