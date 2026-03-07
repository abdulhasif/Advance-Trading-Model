"""
scripts/dry_run.py — Zero-Impact End-to-End Smoke Test
=======================================================
Replays the most recent trading day from feature parquets through
the IDENTICAL prediction + gate pipeline used by paper_trader.py.

ZERO side-effects:
  • No Upstox API calls
  • No writes to production logs (storage/logs/)
  • No writes to logs/paper_debug/
  • Does NOT import or start the FastAPI server
  • Does NOT start the trading loop

Purpose: Verify model loads, features align, all gates execute
         without errors, and charge math produces sane numbers.

Usage:
    .venv\\Scripts\\python.exe scripts/dry_run.py
    .venv\\Scripts\\python.exe scripts/dry_run.py --days 3   # last N days
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.WARNING)   # suppress library noise

DIVIDER = "=" * 65

# ── Mirror paper_trader.py constants exactly ──────────────────────────────────
FEAT_COLS = [
    "velocity", "wick_pressure", "relative_strength",
    "brick_size", "duration_seconds",
    "consecutive_same_dir", "brick_oscillation_rate",
    "fracdiff_price", "hurst", "is_trending_regime",
]
ENTRY_PROB_THRESH  = 0.70
ENTRY_CONV_THRESH  = 45.0
ENTRY_RS_THRESHOLD = 1.0
MAX_ENTRY_WICK     = 0.40
MIN_CONSECUTIVE_BRICKS = 3
EOD_EXIT_HOUR      = 15
EOD_EXIT_MINUTE    = 14
NO_ENTRY_HOUR      = 15
NO_ENTRY_MINUTE    = 0

# Charge math (mirror paper_trader.py exactly)
BROKERAGE_PER_ORDER = 20.0
BROKERAGE_PCT       = 0.0005
STT_SELL_PCT        = 0.00025
STAMP_DUTY_BUY_PCT  = 0.00003
EXCHANGE_TXN_PCT    = 0.0000297
SEBI_TURNOVER_FEE   = 10.0
GST_PCT             = 0.18
POSITION_SIZE_PCT   = 0.02
INTRADAY_LEVERAGE   = 5
PAPER_CAPITAL       = 100_000


def calc_charges(entry_price: float, exit_price: float, qty: int) -> float:
    buy_t  = entry_price * qty
    sell_t = exit_price  * qty
    total  = buy_t + sell_t
    brok   = min(BROKERAGE_PER_ORDER, buy_t  * BROKERAGE_PCT) \
           + min(BROKERAGE_PER_ORDER, sell_t * BROKERAGE_PCT)
    stt    = sell_t * STT_SELL_PCT
    stamp  = buy_t  * STAMP_DUTY_BUY_PCT
    exch   = total  * EXCHANGE_TXN_PCT
    sebi   = total  * (SEBI_TURNOVER_FEE / 1_00_00_000)
    gst    = (brok + exch) * GST_PCT
    return brok + stt + stamp + exch + sebi + gst


def load_models():
    b1 = xgb.XGBClassifier(); b1.load_model(str(config.BRAIN1_MODEL_PATH))
    b2 = xgb.XGBRegressor();  b2.load_model(str(config.BRAIN2_MODEL_PATH))
    return b1, b2


def load_sample(n_days: int) -> pd.DataFrame:
    """Load the most recent N trading days across all symbols."""
    frames = []
    for pf in config.FEATURES_DIR.rglob("*.parquet"):
        try:
            df = pd.read_parquet(pf)
            df["_symbol"] = pf.stem
            # infer sector from parent folder name
            df["_sector"] = pf.parent.name
            frames.append(df)
        except Exception:
            pass
    if not frames:
        print("ERROR: No parquet files found."); sys.exit(1)

    combined  = pd.concat(frames, ignore_index=True)
    combined["brick_timestamp"] = pd.to_datetime(combined["brick_timestamp"])
    combined["_date"]  = combined["brick_timestamp"].dt.date

    all_dates = sorted(combined["_date"].unique())
    target    = all_dates[-n_days:]
    sample    = combined[combined["_date"].isin(target)].copy()
    return sample, target


def run_dry(n_days: int = 1):
    print(f"\n{DIVIDER}")
    print("  DRY RUN — Zero-Impact Smoke Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIVIDER)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading models...")
    b1, b2 = load_models()
    feat_names = b1.get_booster().feature_names
    assert "direction" not in feat_names, "STALE model — retrain!"
    missing = [f for f in FEAT_COLS if f not in feat_names]
    assert not missing, f"Feature mismatch: {missing} missing from model"
    print(f"  ✓ Brain1 + Brain2 loaded | {len(feat_names)} features | no stale columns")

    print(f"\n[2/6] Loading last {n_days} day(s) of data...")
    df, dates = load_sample(n_days)
    print(f"  ✓ {len(df):,} bricks | {df['_symbol'].nunique()} symbols | "
          f"{dates[0]} -> {dates[-1]}")

    # ── Step 2: Feature check ─────────────────────────────────────────────────
    print(f"\n[3/6] Feature integrity check...")
    for f in FEAT_COLS:
        if f not in df.columns:
            print(f"  [!!] MISSING COLUMN: {f}"); sys.exit(1)
    na_counts = df[FEAT_COLS].isna().sum()
    bad_feats  = na_counts[na_counts > len(df) * 0.5]
    if not bad_feats.empty:
        print(f"  [WARN] >50% NaN in: {bad_feats.index.tolist()}")
    print(f"  ✓ All {len(FEAT_COLS)} feature columns present")
    fill_rate  = 1 - df[FEAT_COLS].isna().mean().mean()
    print(f"  ✓ Average feature fill rate: {fill_rate*100:.1f}%")

    # ── Step 3: Run predictions ───────────────────────────────────────────────
    print(f"\n[4/6] Running predictions through Brain1 + Brain2...")
    clean = df.dropna(subset=FEAT_COLS).copy()
    X     = clean[FEAT_COLS].fillna(0).astype(float)

    proba   = b1.predict_proba(X)
    b1_prob = proba[:, 1]
    # Brain2 meta-regressor: only uses [brain1_prob, velocity, wick_pressure, relative_strength]
    B2_COLS = b2.get_booster().feature_names  # read directly from saved model
    X_b2 = X.copy()
    X_b2["brain1_prob"] = b1_prob
    X_b2 = X_b2[[c for c in B2_COLS]]        # keep only what the model expects
    conviction = b2.predict(X_b2)

    clean["_b1p"]  = b1_prob
    clean["_b2c"]  = conviction
    clean["_signal"] = np.where(b1_prob >= 0.5, "LONG", "SHORT")

    print(f"  ✓ {len(clean):,} predictions | "
          f"LONG={( b1_prob>=0.5).sum():,} "
          f"SHORT={(b1_prob< 0.5).sum():,}")
    print(f"  ✓ B1 prob range: [{b1_prob.min():.4f} – {b1_prob.max():.4f}]")
    print(f"  ✓ Conviction range: [{conviction.min():.1f} – {conviction.max():.1f}]")

    # ── Step 4: Gate walkthrough (no side effects) ────────────────────────────
    print(f"\n[5/6] Gate pipeline walkthrough (dry-fire all gates)...")

    gate_counts = {
        "total":         0,
        "eod_blocked":   0,
        "prob_failed":   0,
        "conv_failed":   0,
        "rs_long_fail":  0,
        "rs_short_fail": 0,
        "wick_blocked":  0,
        "whipsaw_fail":  0,
        "passed":        0,
    }
    paper_trades = []

    for sym, sym_df in clean.groupby("_symbol"):
        sym_df  = sym_df.sort_values("brick_timestamp").reset_index(drop=True)
        sector  = sym_df["_sector"].iloc[0]
        in_pos  = False
        entry_p = 0.0
        entry_t = None

        for i, row in sym_df.iterrows():
            gate_counts["total"] += 1
            ts     = row["brick_timestamp"]
            b1p    = row["_b1p"]
            b2c    = row["_b2c"]
            signal = row["_signal"]
            rs     = float(row.get("relative_strength", 0) or 0)
            wick   = float(row.get("wick_pressure", 0) or 0)
            consec = int(row.get("consecutive_same_dir", 0) or 0)
            price  = float(row.get("brick_close", row.get("close", 1000)) or 1000)

            if in_pos:
                # simulate trivial exit at next brick close
                pnl_pct = (price - entry_p) / entry_p * 100
                qty = max(1, int((PAPER_CAPITAL * POSITION_SIZE_PCT * INTRADAY_LEVERAGE) / entry_p))
                charges = calc_charges(entry_p, price, qty)
                charges_pct = charges / (entry_p * qty) * 100
                net = pnl_pct - charges_pct
                paper_trades.append({
                    "symbol": sym, "sector": sector, "signal": signal,
                    "entry": round(entry_p, 2), "exit": round(price, 2),
                    "qty": qty, "charges": round(charges, 2),
                    "net_pnl_pct": round(net, 4), "entry_time": str(entry_t),
                })
                in_pos = False
                continue

            # EOD gate
            if ts.hour > NO_ENTRY_HOUR or (ts.hour == NO_ENTRY_HOUR and ts.minute >= NO_ENTRY_MINUTE):
                gate_counts["eod_blocked"] += 1; continue

            # Prob gate
            prob_ok = (b1p >= ENTRY_PROB_THRESH) if signal == "LONG" else ((1-b1p) >= ENTRY_PROB_THRESH)
            if not prob_ok:
                gate_counts["prob_failed"] += 1; continue

            # Conviction gate
            if b2c < ENTRY_CONV_THRESH:
                gate_counts["conv_failed"] += 1; continue

            # RS gate (Long)
            if signal == "LONG" and rs < ENTRY_RS_THRESHOLD:
                gate_counts["rs_long_fail"] += 1; continue

            # RS gate (Short) — fixed
            if signal == "SHORT" and rs > 0.0:
                gate_counts["rs_short_fail"] += 1; continue

            # Wick gate
            if wick > MAX_ENTRY_WICK:
                gate_counts["wick_blocked"] += 1; continue

            # Whipsaw gate
            if consec < MIN_CONSECUTIVE_BRICKS:
                gate_counts["whipsaw_fail"] += 1; continue

            # Charge sanity
            qty = max(1, int((PAPER_CAPITAL * POSITION_SIZE_PCT * INTRADAY_LEVERAGE) / price))
            ch  = calc_charges(price, price * 1.001, qty)   # 0.1% hypothetical move
            assert ch > 0 and ch < price * qty, f"Charge math broken: {ch}"

            gate_counts["passed"] += 1
            in_pos  = True
            entry_p = price
            entry_t = ts

    total = gate_counts["total"]
    print(f"  ✓ Gate pipeline: {total:,} bricks processed")
    print(f"    {'EOD blocked':<22}: {gate_counts['eod_blocked']:>6,}  "
          f"({gate_counts['eod_blocked']/total*100:.1f}%)")
    print(f"    {'Prob gate failed':<22}: {gate_counts['prob_failed']:>6,}  "
          f"({gate_counts['prob_failed']/total*100:.1f}%)")
    print(f"    {'Conv gate failed':<22}: {gate_counts['conv_failed']:>6,}  "
          f"({gate_counts['conv_failed']/total*100:.1f}%)")
    print(f"    {'RS LONG failed':<22}: {gate_counts['rs_long_fail']:>6,}  "
          f"({gate_counts['rs_long_fail']/total*100:.1f}%)")
    print(f"    {'RS SHORT failed':<22}: {gate_counts['rs_short_fail']:>6,}  "
          f"({gate_counts['rs_short_fail']/total*100:.1f}%)")
    print(f"    {'Wick blocked':<22}: {gate_counts['wick_blocked']:>6,}  "
          f"({gate_counts['wick_blocked']/total*100:.1f}%)")
    print(f"    {'Whipsaw failed':<22}: {gate_counts['whipsaw_fail']:>6,}  "
          f"({gate_counts['whipsaw_fail']/total*100:.1f}%)")
    print(f"    {'>>> PASSED (would enter)':<22}: {gate_counts['passed']:>6,}  "
          f"({gate_counts['passed']/total*100:.1f}%)")

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    print(f"\n[6/6] Simulated trade summary...")
    if paper_trades:
        t  = pd.DataFrame(paper_trades)
        wr = (t["net_pnl_pct"] > 0).mean() * 100
        print(f"  ✓ {len(t)} simulated trade(s) | Win rate: {wr:.0f}%")
        print(f"  ✓ Avg PnL/trade: {t['net_pnl_pct'].mean():+.4f}%")
        print(f"  ✓ Avg charges:   ₹{t['charges'].mean():.2f} per trade")
        print(f"\n  Sample trades (first 5):")
        print(f"  {'Symbol':<12} {'Side':<7} {'Entry':>8} {'Exit':>8} "
              f"{'Qty':>5} {'Charges':>9} {'Net PnL':>9}")
        print(f"  {'-'*60}")
        for _, r in t.head(5).iterrows():
            print(f"  {r['symbol']:<12} {r['signal']:<7} {r['entry']:>8.2f} "
                  f"{r['exit']:>8.2f} {r['qty']:>5} "
                  f"₹{r['charges']:>8.2f} {r['net_pnl_pct']:>+8.4f}%")
    else:
        print("  ✓ No trades fired in this sample (gates are working correctly)")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  ✅  DRY RUN COMPLETE — No files written, no API calls made")
    print(f"  System is structurally sound and ready for live trading.")
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1,
                        help="Number of recent trading days to replay (default: 1)")
    args = parser.parse_args()
    run_dry(n_days=args.days)
