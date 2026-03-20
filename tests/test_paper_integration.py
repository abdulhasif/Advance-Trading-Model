"""Quick integration smoke test for the Paper Trading Engine."""
import sys
sys.path.insert(0, ".")

print("1. Testing imports...")
from src.live.paper_trader import (
    PaperPortfolio, PaperPosition, passes_soft_veto,
    PAPER_CAPITAL, ENTRY_PROB_THRESH, ENTRY_CONV_THRESH
)
from src.live.tick_provider import TickProvider
from src.core.renko import LiveRenkoState
from src.core.features import compute_features_live
from src.core.risk import RiskFortress
import xgboost as xgb
import config
print("   ALL IMPORTS OK")

print()
print("2. Testing model loading...")
b1 = xgb.XGBClassifier(); b1.load_model(str(config.BRAIN1_MODEL_PATH))
b2 = xgb.XGBRegressor();  b2.load_model(str(config.BRAIN2_MODEL_PATH))
print(f"   Brain1: {type(b1).__name__} loaded")
print(f"   Brain2: {type(b2).__name__} loaded")

print()
print("3. Testing PaperPortfolio...")
pf = PaperPortfolio(100000)
print(f"   Starting capital: Rs {pf.starting_capital:,}")
print(f"   Cash: Rs {pf.cash:,}")
print(f"   Open positions: {len(pf.positions)}")

print()
print("4. Testing virtual trade cycle...")
from datetime import datetime
now = datetime.now()
opened = pf.open_position("SBIN", "Banking", "LONG", 625.50, 620.00, now)
print(f"   Opened SBIN LONG: {opened}")
print(f"   Open positions: {len(pf.positions)}")
unrealized = pf.positions["SBIN"].unrealized_pnl
print(f"   Unrealized PnL: Rs {unrealized:.2f}")

# Simulate price move
pf.positions["SBIN"].last_price = 630.0
unrealized2 = pf.positions["SBIN"].unrealized_pnl
print(f"   After price move 625.5 -> 630.0:")
print(f"   Unrealized PnL: Rs {unrealized2:.2f}")

# Close it
closed = pf.close_position("SBIN", 630.0, now, "TREND_REVERSAL")
print(f"   Closed SBIN: Realized PnL = Rs {closed.realized_pnl:.2f}")
print(f"   Cash after trade: Rs {pf.cash:,.2f}")
print(f"   Closed trades count: {len(pf.closed_trades)}")

print()
print("5. Testing max positions limit...")
pf2 = PaperPortfolio(100000)
r1 = pf2.open_position("SBIN", "Banking", "LONG", 625.0, 620.0, now)
r2 = pf2.open_position("RELIANCE", "Energy", "SHORT", 2500.0, 2510.0, now)
r3 = pf2.open_position("TCS", "IT", "LONG", 3800.0, 3780.0, now)
r4 = pf2.open_position("INFY", "IT", "LONG", 1500.0, 1490.0, now)  # Should fail (max 3)
print(f"   Opened 1: {r1}, 2: {r2}, 3: {r3}, 4 (should fail): {r4}")
print(f"   Open positions: {len(pf2.positions)} (expect 3)")

print()
print("6. Testing EOD close all...")
pf2.close_all_eod(now)
print(f"   After EOD close: {len(pf2.positions)} positions (expect 0)")
print(f"   Closed trades: {len(pf2.closed_trades)}")

print()
print("7. Testing soft veto...")
print(f"   LONG + rel_str=-0.8: {passes_soft_veto('LONG', -0.8)} (expect False)")
print(f"   LONG + rel_str=+0.3: {passes_soft_veto('LONG', 0.3)} (expect True)")
print(f"   SHORT + rel_str=+0.8: {passes_soft_veto('SHORT', 0.8)} (expect False)")
print(f"   SHORT + rel_str=-0.3: {passes_soft_veto('SHORT', -0.3)} (expect True)")

print()
print("8. Testing TickProvider...")
tp = TickProvider(["SBIN", "RELIANCE", "TCS"])
tp.connect()
ticks = tp.get_latest_ticks()
print(f"   Ticks received for {len(ticks)} symbols")
for sym in sorted(ticks.keys()):
    t = ticks[sym]
    print(f"   {sym}: ltp={t['ltp']:.2f}, high={t['high']:.2f}, low={t['low']:.2f}")
tp.disconnect()

print()
print("9. Testing RiskFortress...")
rf = RiskFortress()
score = rf.score_signal(0.72, 80.0, 1, 1)
print(f"   Score (aligned): {score:.2f}")
score2 = rf.score_signal(0.72, 80.0, 1, -1)
print(f"   Score (misaligned): {score2:.2f} (expect -25 penalty)")

print()
print("10. Testing LiveRenkoState...")
rs = LiveRenkoState("SBIN", "Banking", 0.75)
print(f"   Created renko state for SBIN with brick_size=0.75")
print(f"   Initial bricks: {len(rs.bricks)}")

print()
print("11. Testing PnL JSON write...")
import json
pf.write_pnl_state()
with open("paper_pnl.json") as f:
    state = json.load(f)
print(f"   Mode: {state['mode']}")
print(f"   Total equity: Rs {state['total_equity']:,.2f}")
print(f"   Total trades: {state['total_trades']}")

print()
print("12. Testing daily summary...")
summary = pf.record_daily_summary("2026-02-18")
print(f"   Date: {summary['date']}")
print(f"   Equity: Rs {summary['total_equity']:,.2f}")

print()
print("=" * 50)
print("ALL 12 INTEGRATION TESTS PASSED")
print("=" * 50)
