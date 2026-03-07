import yfinance as yf
import json
from src.live.upstox_simulator import UpstoxSimulator
from datetime import datetime

with open('live_state.json', 'r') as f:
    state = json.load(f)

active = state.get('active_trades', [])

sim = UpstoxSimulator()
symbols = [t['symbol'] for t in active]
yf_symbols = [s + '.NS' for s in symbols]
data = yf.download(yf_symbols, period='1d', progress=False)['Close']

print(f"{'SYMBOL':<15} {'ENTRY':<10} {'CLOSE':<10} {'NET PNL':<10}")
for t in active:
    sym = t['symbol']
    entry = t['entry_price']
    
    # Place order
    o = sim.place_order(sym, t['side'], t['qty'], entry, datetime.fromisoformat(t['entry_time']))
    sim.fill_pending_order(sym, datetime.fromisoformat(t['entry_time']))
    
    try:
        if len(symbols) == 1:
            close_px = float(data.iloc[-1])
        else:
            close_px = float(data[sym + '.NS'].iloc[-1])
    except Exception as e:
        close_px = entry

    sim.update_active_price(sym, close_px)
    sim.close_position(sym, close_px, datetime.now(), "EOD_SQUARE_OFF")
    
    print(f"{sym:<15} {entry:<10.2f} {close_px:<10.2f} {o.net_pnl:<10.2f}")

print("\n--- Daily Summary ---")
print(sim.generate_daily_summary())
