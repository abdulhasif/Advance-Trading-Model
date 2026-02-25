"""Quick diagnostic: are we receiving ticks? Are Renko bricks forming?"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas as pd
from src.live.tick_provider import TickProvider
from src.core.renko import LiveRenkoState

# Test with a few liquid stocks
TEST_SYMBOLS = ["SBIN", "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
                "NIFTY 50", "NIFTY BANK"]

print(f"Access token set: {bool(config.UPSTOX_ACCESS_TOKEN)}")
print(f"Token prefix: {config.UPSTOX_ACCESS_TOKEN[:20]}..." if config.UPSTOX_ACCESS_TOKEN else "NO TOKEN")

tp = TickProvider(TEST_SYMBOLS)
tp.connect()
print(f"Connected: {tp._connected}, Live mode: {tp._use_live}")
print(f"Symbols mapped: {list(tp._sym_to_ikey.keys())}")

# Check brick sizes
for sym in TEST_SYMBOLS:
    universe = pd.read_csv(config.UNIVERSE_CSV)
    row = universe[universe["symbol"] == sym]
    if not row.empty:
        sec = row.iloc[0]["sector"]
        stock_dir = config.DATA_DIR / sec / sym
        if stock_dir.exists():
            pqs = sorted(stock_dir.glob("*.parquet"))
            if pqs:
                df = pd.read_parquet(pqs[-1])
                bs = df["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
                print(f"  {sym}: brick_size={bs:.4f} (from {pqs[-1].name})")

# Read ticks for 30 seconds
renko_states = {}
for sym in TEST_SYMBOLS:
    renko_states[sym] = LiveRenkoState(sym, "Test", 5.0)  # using 5.0 as placeholder

print(f"\nListening for ticks for 30 seconds...")
for i in range(30):
    ticks = tp.get_latest_ticks()
    if ticks:
        print(f"\n[{i+1}s] Got {len(ticks)} ticks:")
        for sym, t in sorted(ticks.items()):
            if sym in TEST_SYMBOLS:
                print(f"  {sym}: LTP={t['ltp']:.2f}  High={t['high']:.2f}  "
                      f"Low={t['low']:.2f}  ts={t['timestamp']}")
                
                # Try feeding to renko
                st = renko_states[sym]
                prev = len(st.bricks)
                st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"])
                if len(st.bricks) > prev:
                    print(f"    >> NEW BRICK #{len(st.bricks)} dir={st.bricks[-1]['direction']}")
    else:
        print(f"[{i+1}s] No ticks received")
    time.sleep(1)

print(f"\nFinal brick counts:")
for sym, st in renko_states.items():
    print(f"  {sym}: {len(st.bricks)} bricks")

tp.disconnect()
print("Done.")
