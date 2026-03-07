import json
import pandas as pd
from datetime import datetime
from src.live.upstox_simulator import UpstoxSimulator
from pathlib import Path
import yfinance as yf

# Load the current live state
state_file = Path('live_state.json')
if not state_file.exists():
    print("No live_state.json found. Exiting.")
    exit(1)

with open(state_file, 'r') as f:
    state = json.load(f)

active = state.get('active_trades', [])
if not active:
    print("No active trades found in live_state.json.")
    exit(0)

# Fetch closing prices
symbols = [t['symbol'] for t in active]
yf_symbols = [s + '.NS' for s in symbols]
data = yf.download(yf_symbols, period='1d', progress=False)['Close']

sim = UpstoxSimulator()
trade_records = []

for t in active:
    sym = t['symbol']
    # Place and fill in simulator
    o = sim.place_order(sym, t['side'], t['qty'], t['entry_price'], datetime.fromisoformat(t['entry_time']))
    sim.fill_pending_order(sym, datetime.fromisoformat(t['entry_time']))
    
    # Get closing price from Yahoo Finance
    try:
        if len(symbols) == 1:
            close_px = float(data.iloc[-1])
        else:
            close_px = float(data[sym + '.NS'].iloc[-1])
    except:
        close_px = t['entry_price']

    # Close in simulator
    sim.update_active_price(sym, close_px)
    exit_ts = datetime.now().replace(hour=15, minute=30, second=0)
    sim.close_position(sym, close_px, exit_ts, "EOD_SQUARE_OFF")
    
    # Matching CSV header: trade_id,symbol,sector,side,entry_time,entry_price,exit_time,exit_price,qty,bricks_held,favorable,adverse,gross_pnl,cost,net_pnl,exit_reason
    trade_records.append({
        "trade_id":      o.trade_id,
        "symbol":        sym,
        "sector":        "Equity", # Placeholder
        "side":          o.side,
        "entry_time":    o.filled_at.isoformat(),
        "entry_price":   round(o.entry_price, 2),
        "exit_time":     o.closed_at.isoformat(),
        "exit_price":    round(o.exit_price, 2),
        "qty":           o.qty,
        "bricks_held":   0, # Estimating
        "favorable":     0,
        "adverse":       0,
        "gross_pnl":     round(o.gross_pnl, 2),
        "cost":          round(o.total_friction, 2),
        "net_pnl":       round(o.net_pnl, 2),
        "exit_reason":   o.exit_reason
    })

# Write to CSV
trades_log = Path('storage/logs/paper_trades.csv')
df_new = pd.DataFrame(trade_records)

if trades_log.exists():
    df_old = pd.read_csv(trades_log)
    df_final = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(trades_log, index=False)
print(f"Successfully backfilled {len(trade_records)} trades to {trades_log}")

# Also backfill paper_daily.csv
summary = sim.generate_daily_summary(datetime.now().date())
daily_log = Path('storage/logs/paper_daily.csv')
daily_record = {
    "date":           summary["Date"],
    "trades":         summary["Total_Trades"],
    "wins":           int(summary["Win_Ratio_%"] * summary["Total_Trades"] / 100),
    "losses":         int(summary["Total_Trades"] - (summary["Win_Ratio_%"] * summary["Total_Trades"] / 100)),
    "realized_pnl":   round(summary["Net_PnL_Rs "], 2),
    "unrealized_pnl": 0.0,
    "total_equity":   round(summary["Ending_Capital_Rs "], 2),
    "win_rate":       summary["Win_Ratio_%"],
    "open_positions": 0,
    "signals_seen":   0,
    "signals_vetoed": 0
}

df_daily_new = pd.DataFrame([daily_record])
if daily_log.exists():
    df_daily_old = pd.read_csv(daily_log)
    df_daily_final = pd.concat([df_daily_old, df_daily_new], ignore_index=True)
else:
    df_daily_final = df_daily_new

df_daily_final.to_csv(daily_log, index=False)
print(f"Successfully updated daily summary in {daily_log}")
