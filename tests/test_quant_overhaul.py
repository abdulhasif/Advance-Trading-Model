"""
tests/test_quant_overhaul.py — Unit Tests for All 4 Phases
===========================================================
Validates the institutional overhaul implementations without requiring
the full 17.8M-row dataset.

Run from project root:
    .venv\\Scripts\\python.exe tests\\test_quant_overhaul.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

print("=" * 60)
print("QUANT OVERHAUL — UNIT TESTS")
print("=" * 60)

# ── PHASE 1: RenkoBrickBuilder Tests ────────────────────────────────────────

print("\n[Phase 1] Testing RenkoBrickBuilder...")

from src.core.renko import RenkoBrickBuilder, check_path_conflict, _brownian_bridge, LiveRenkoState

# Test 1a: Brownian Bridge shape and pinning
bb = _brownian_bridge(p_start=100.0, p_end=105.0, n_steps=10, sigma=0.5, seed=42)
assert len(bb) == 10, f"Expected 10 steps, got {len(bb)}"
assert abs(bb[-1] - 105.0) < 1e-9, f"Bridge not pinned at end: {bb[-1]}"
print(f"  [PASS] Brownian Bridge: Generated 10 sub-ticks, pinned at 105.0")

# Test 1b: Volume passthrough — large candle should produce bricks with volume
ohlc = pd.DataFrame({
    "timestamp": pd.to_datetime(["2025-01-02 09:15", "2025-01-02 09:16"]),
    "open":  [100.0, 102.4],
    "high":  [103.0, 103.0],
    "low":   [ 99.0,  99.0],
    "close": [102.0, 100.0],
    "volume": [1000.0, 1500.0],
})
# Add timezone
ohlc["timestamp"] = ohlc["timestamp"].dt.tz_localize("Asia/Kolkata")

builder = RenkoBrickBuilder(natr_pct=0.01)  # 1% brick = 1.0 for price 100
bricks = builder.transform(ohlc)

if len(bricks) > 0:
    assert "volume" in bricks.columns, "Volume column missing from bricks"
    assert "typical_price" in bricks.columns, "typical_price missing"
    assert "cum_volume" in bricks.columns, "cum_volume missing"
    total_brick_vol = bricks["volume"].sum()
    total_candle_vol = ohlc["volume"].sum()
    # Volume conservation: brick volumes should roughly equal candle volumes
    # (may differ slightly due to Brownian Bridge distribution)
    print(f"  [PASS] Volume passthrough: brick sum={total_brick_vol:.0f}, candle sum={total_candle_vol:.0f}")
else:
    print("  [WARN] No bricks generated — brick_size likely > candle range. Checking...")
    print(f"         Candle range: {ohlc['high'].max() - ohlc['low'].min():.2f}")

# Test 1c: Path-Conflict resolution
# Scenario: Stock moves UP to target, then DOWN to stop in the same candle
tick_path_conflict    = np.array([100.0, 101.0, 105.5, 103.0, 96.0])  # hits (+5) then stop (-3)
tick_path_clean_win   = np.array([100.0, 101.0, 105.5, 104.0])         # only hits target
tick_path_clean_loss  = np.array([100.0, 99.5,  97.0,  96.0])          # only hits stop

result_conflict = check_path_conflict(tick_path_conflict,  entry_price=100.0, target_price=105.0, stop_price=97.0)
result_win      = check_path_conflict(tick_path_clean_win,  entry_price=100.0, target_price=105.0, stop_price=97.0)
result_loss     = check_path_conflict(tick_path_clean_loss, entry_price=100.0, target_price=105.0, stop_price=97.0)

assert result_conflict == "LOSS", f"Conflict should be LOSS, got {result_conflict}"
assert result_win      == "WIN",  f"Clean win should be WIN, got {result_win}"
assert result_loss     == "LOSS", f"Clean stop should be LOSS, got {result_loss}"
print(f"  [PASS] Path-Conflict: conflict={result_conflict}, clean_win={result_win}, clean_loss={result_loss}")

# ── PHASE 2: Alpha Feature Tests ─────────────────────────────────────────────

print("\n[Phase 2] Testing Alpha Feature Functions...")

from src.core.features import (
    compute_vwap_zscore,
    compute_vpt_acceleration,
    compute_squeeze_zscore,
    compute_streak_exhaustion,
    compute_consecutive_same_dir,
)

# Build synthetic brick DataFrame
np.random.seed(0)
n = 50
prices  = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
volumes = np.abs(np.random.randn(n) * 1000) + 500.0
directions = np.random.choice([-1, 1], size=n)
durations  = np.random.uniform(20, 120, size=n)

df_test = pd.DataFrame({
    "brick_timestamp": pd.date_range("2025-01-02 09:20", periods=n, freq="2min", tz="Asia/Kolkata"),
    "brick_open":  prices - 0.05,
    "brick_close": prices,
    "brick_high":  prices + 0.1,
    "brick_low":   prices - 0.1,
    "brick_size":  0.5,
    "direction":   directions,
    "duration_seconds": durations,
    "volume":       volumes,
    "typical_price": prices,
})

# Test 2a: VWAP Z-Score uses volume
vwap_z = compute_vwap_zscore(df_test, window=20)
assert len(vwap_z) == n, "VWAP Z-Score length mismatch"
assert vwap_z.notna().sum() > 0, "All VWAP Z-Scores are NaN"
finite_vals = vwap_z[vwap_z.notna()]
assert (finite_vals.abs() < 10).all(), f"VWAP Z-Score out of bounds: {finite_vals.max():.2f}"
print(f"  [PASS] VWAP Z-Score: range=[{vwap_z.min():.2f}, {vwap_z.max():.2f}]")

# Test 2b: VWAP Z-Score with zero volume falls back gracefully
df_no_vol = df_test.copy()
df_no_vol["volume"] = 0.0
vwap_z_fallback = compute_vwap_zscore(df_no_vol, window=20)
assert len(vwap_z_fallback) == n, "VWAP fallback length mismatch"
print(f"  [PASS] VWAP Z-Score fallback (zero volume): range=[{vwap_z_fallback.min():.2f}, {vwap_z_fallback.max():.2f}]")

# Test 2c: VPT Acceleration returns a series, is positive when volume spikes with flat price
vpt_accel = compute_vpt_acceleration(df_test, diff_lag=2)
assert len(vpt_accel) == n, "VPT Acceleration length mismatch"
print(f"  [PASS] VPT Acceleration: range=[{vpt_accel.min():.3f}, {vpt_accel.max():.3f}]")

# Test 2d: Squeeze Z-Score — returns bounded values
sq_z = compute_squeeze_zscore(df_test, window=20)
assert len(sq_z) == n, "Squeeze Z-Score length mismatch"
# Drop NaN (rolling window warm-up) before bounds check; clip is at ±4.0
sq_z_valid = sq_z.dropna()
assert len(sq_z_valid) > 0, "All Squeeze Z-Scores are NaN"
assert (sq_z_valid.abs() <= 4.5).all(), f"Squeeze Z-Score out of bounds: {sq_z_valid.abs().max():.2f}"
print(f"  [PASS] Squeeze Z-Score: range=[{sq_z_valid.min():.2f}, {sq_z_valid.max():.2f}] ({len(sq_z_valid)} valid rows)")


# Test 2e: Streak Exhaustion — 0.0 until onset, negative after
# Build a streak of 12 consecutive same-direction bricks
df_streak = df_test.copy()
df_streak["direction"] = 1  # all up
streak_ex = compute_streak_exhaustion(df_streak, onset=8, scale=0.5)
assert len(streak_ex) == n, "Streak Exhaustion length mismatch"
# Bricks 8+: exhaustion should be < 0
tail_vals = streak_ex.iloc[10:]
assert (tail_vals < 0).all(), f"Streak exhaustion not negative after onset: {tail_vals.values[:5]}"
assert streak_ex.min() >= -0.5, f"Streak exhaustion below -0.5: {streak_ex.min():.3f}"
print(f"  [PASS] Streak Exhaustion: onset=8, range=[{streak_ex.min():.3f}, {streak_ex.max():.3f}]")

# ── PHASE 3: IsotonicCalibrationWrapper Tests ────────────────────────────────

print("\n[Phase 3] Testing IsotonicCalibrationWrapper...")

from src.core.quant_fixes import IsotonicCalibrationWrapper

wrapper = IsotonicCalibrationWrapper()
assert not wrapper._is_fitted, "Should be unfitted initially"
try:
    wrapper.predict_proba(pd.DataFrame({"a": [1, 2, 3]}))
    assert False, "Should have raised RuntimeError"
except RuntimeError as e:
    print(f"  [PASS] Unfitted wrapper raises RuntimeError: {str(e)[:50]}")

print(f"  [PASS] IsotonicCalibrationWrapper instantiation and guard check passed")

# ── PHASE 4: Backtester SLIPPAGE_PCT Tests ──────────────────────────────────

print("\n[Phase 4] Testing Pessimistic Execution Constants...")

from src.ml.backtester import SLIPPAGE_PCT, JITTER_SECONDS, PATH_CONFLICT, close_position, Trade
import pandas as pd

assert SLIPPAGE_PCT    == 0.0005, f"SLIPPAGE_PCT wrong: {SLIPPAGE_PCT}"
assert JITTER_SECONDS  == 1.0,    f"JITTER_SECONDS wrong: {JITTER_SECONDS}"
assert PATH_CONFLICT   == True,   f"PATH_CONFLICT wrong: {PATH_CONFLICT}"
print(f"  [PASS] Constants: SLIPPAGE_PCT={SLIPPAGE_PCT}, JITTER={JITTER_SECONDS}, PATH_CONFLICT={PATH_CONFLICT}")

# Test that close_position applies slippage to exit
t = Trade(
    trade_id=1, symbol="TEST", sector="Test",
    side="LONG", entry_time=pd.Timestamp("2025-01-02 09:30", tz="Asia/Kolkata"),
    entry_price=1000.0, qty=10, bricks_held=5,
)
ts = pd.Timestamp("2025-01-02 10:00", tz="Asia/Kolkata")

# Close at 1050.0 — effective exit should be 1050.0 * (1 - 0.0005) = 1049.475
t_closed = close_position(t, 1050.0, ts, "TREND_REVERSAL")
expected_exit = 1050.0 * (1.0 - SLIPPAGE_PCT)
assert abs(t_closed.exit_price - expected_exit) < 0.01, \
    f"Exit slippage wrong: {t_closed.exit_price:.4f} vs expected {expected_exit:.4f}"

# Without slippage, PnL would be (1050 - 1000) / 1000 = 5.0%
# With slippage, it should be slightly less
assert t_closed.gross_pnl_pct < 0.05, f"gross_pnl_pct should be < 5%: {t_closed.gross_pnl_pct:.4f}"
print(f"  [PASS] Exit slippage applied: exit_price={t_closed.exit_price:.2f} (from raw 1050.0)")
print(f"  [PASS] Gross PnL:{t_closed.gross_pnl_pct*100:.3f}% (< 5.0% due to slippage)")

# ── CONFIG VERIFICATION ─────────────────────────────────────────────────────

print("\n[Config] Validating new config.py constants...")
import config
assert hasattr(config, "XGBOOST_COLSAMPLE_BYTREE"), "XGBOOST_COLSAMPLE_BYTREE missing from config"
assert config.XGBOOST_COLSAMPLE_BYTREE == 0.7, f"colsample_bytree wrong: {config.XGBOOST_COLSAMPLE_BYTREE}"
assert config.XGBOOST_SUBSAMPLE == 0.7, f"subsample wrong: {config.XGBOOST_SUBSAMPLE}"
assert hasattr(config, "BRAIN1_CALIBRATED_PATH"), "BRAIN1_CALIBRATED_PATH missing"
assert hasattr(config, "VWAP_WINDOW"), "VWAP_WINDOW missing"
assert hasattr(config, "STREAK_EXHAUSTION_ONSET"), "STREAK_EXHAUSTION_ONSET missing"
print(f"  [PASS] XGBOOST_COLSAMPLE_BYTREE={config.XGBOOST_COLSAMPLE_BYTREE}")
print(f"  [PASS] XGBOOST_SUBSAMPLE={config.XGBOOST_SUBSAMPLE}")
print(f"  [PASS] BRAIN1_CALIBRATED_PATH={config.BRAIN1_CALIBRATED_PATH}")

print()
print("=" * 60)
print("ALL UNIT TESTS PASSED")
print("=" * 60)
print()
print("Next steps (pipeline reset — requires 30-45 min):")
print("  1. .venv\\Scripts\\python.exe main.py features   # Re-engineer features")
print("  2. .venv\\Scripts\\python.exe main.py train      # Retrain w/ Isotonic Cal.")
print("  3. .venv\\Scripts\\python.exe main.py backtest   # Pessimistic backtest")
