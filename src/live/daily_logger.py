"""
src/live/daily_logger.py — Daily Rotating Paper Trade Audit Log
================================================================
Creates one CSV per trading day at:
    logs/paper_debug/YYYY-MM-DD.csv

Every row = one new-brick event.  Captures the raw features, model
predictions, every gate (pass/fail), bias state, and final action.
This is the ground truth you need to detect data leaks.

DATA LEAK DETECTION CHECKLIST
──────────────────────────────
1. LOOK-AHEAD BIAS (most common):
   - Sort by timestamp and check if features at time T ever contain
     information from T+1 or later (e.g. next brick's price).
   - Script: python scripts/check_data_leak.py

2. FILTER BIAS:
   - Compare avg(brain1_prob) for ENTRY rows vs SKIP rows.
   - If gate_rs also fires frequently, check relative_strength
     calculation uses only same-tick market data.

3. THRESHOLD DRIFT:
   - Monitor eff_prob_thresh column. Should be 0.75 (no bias) or 0.65
     (bias active). Any other value = code bug.

4. CONSECUTIVE BRICK WINDOW:
   - Check new_bricks column. All ENTRY rows should have new_bricks >= 3.
   - Any ENTRY with new_bricks < 3 = gate bug.

5. SURVIVORSHIP BIAS:
   - Cross-reference symbols that appear in the log vs symbols that
     dropped out (no ticks). If tickers with no ticks are missing, fine.
     If they appear with suspiciously clean data = data source issue.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA (one column per gate for easy filtering in Excel/Pandas)
# ─────────────────────────────────────────────────────────────────────────────
_HEADERS = [
    # ── When & Where ──────────────────────────────────────────────────────
    "timestamp",          # ISO-8601 brick fire time
    "symbol",             # NSE ticker
    "sector",             # Sector string

    # ── Raw Price & Brick ──────────────────────────────────────────────────
    "price",              # LTP at brick fire
    "brick_dir",          # +1 (up) or -1 (down)
    "sec_dir",            # Sector brick direction (+1/-1/0)
    "new_bricks",         # Total bricks fired so far in current run (rolling)

    # ── Raw Feature Values ─────────────────────────────────────────────────
    "velocity",           # Price velocity (brick size / duration_seconds)
    "wick_pressure",      # Wick absorption metric
    "relative_strength",  # RS vs sector index
    "brick_size",         # Absolute brick size in Rs
    "duration_seconds",   # Time to form this brick
    "consecutive_same",   # Consecutive bricks in same direction
    "oscillation_rate",   # Brick flip rate in recent window

    # ── Model Predictions ──────────────────────────────────────────────────
    "brain1_prob",        # P(LONG) from Brain1 XGBoost [0,1]
    "brain2_conv",        # Conviction score from Brain2 [0,100]
    "signal",             # "LONG" or "SHORT" (derived from brain1_prob > 0.5)
    "score",              # RiskFortress composite signal score

    # ── Control State at Decision Time ────────────────────────────────────
    "global_kill",        # CONTROL_STATE["GLOBAL_KILL"]
    "global_pause",       # CONTROL_STATE["GLOBAL_PAUSE"]
    "ticker_paused",      # sym in CONTROL_STATE["PAUSED_TICKERS"]
    "bias",               # CONTROL_STATE["BIAS"].get(sym) or ""

    # ── Effective Threshold ────────────────────────────────────────────────
    "eff_prob_thresh",    # 0.75 (no bias) or 0.65 (bias engaged)

    # ── Gate Verdicts: "PASS" / "FAIL" / "SKIP" (not evaluated) ──────────
    "gate_prob",          # Gate 1: brain1_prob vs eff_prob_thresh
    "gate_conv",          # Gate 1b: brain2_conv >= ENTRY_CONV_THRESH
    "gate_rs",            # Gate 2: |relative_strength| >= ENTRY_RS_THRESHOLD
    "gate_wick",          # Gate 3: wick_pressure < MAX_ENTRY_WICK
    "gate_whipsaw",       # Whipsaw: consecutive bricks >= MIN_CONSECUTIVE_BRICKS
    "gate_losses",        # Daily loss guard: losses < MAX_LOSSES_PER_STOCK
    "gate_positions",     # Position cap: open < MAX_OPEN_POSITIONS

    # ── Final Decision ─────────────────────────────────────────────────────
    "action",             # ENTRY / EXIT / SKIP
    "reason",             # Human-readable gate name or exit reason
    "open_positions",     # Snapshot of open position count
    "live_pnl",           # Aggregate unrealized PnL at this moment
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
_DEBUG_DIR: Path = config.LOGS_DIR / "paper_debug"
_current_date: str = ""
_current_file: Optional[Path] = None


def _get_log_file(date_str: str) -> Path:
    """Return (and create if needed) today's debug CSV path."""
    global _current_date, _current_file
    if date_str != _current_date:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        _current_file = _DEBUG_DIR / f"{date_str}.csv"
        _current_date = date_str
        # Write header only if file is new
        if not _current_file.exists():
            with open(_current_file, "w", newline="") as f:
                csv.writer(f).writerow(_HEADERS)
            logger.info(f"DailyLogger: new file → {_current_file}")
    return _current_file


def log_brick_event(
    *,
    # Required context
    ts: datetime,
    symbol: str,
    sector: str,
    price: float,
    brick_dir: int,
    sec_dir: int,
    new_bricks: int,

    # Raw features (pass the pandas Series or individual values)
    velocity: float = 0.0,
    wick_pressure: float = 0.0,
    relative_strength: float = 0.0,
    brick_size: float = 0.0,
    duration_seconds: float = 0.0,
    consecutive_same: int = 0,
    oscillation_rate: float = 0.0,

    # Model outputs
    brain1_prob: float = 0.0,
    brain2_conv: float = 0.0,
    signal: str = "",
    score: float = 0.0,

    # Control state snapshot (read outside lock before calling)
    global_kill: bool = False,
    global_pause: bool = False,
    ticker_paused: bool = False,
    bias: str = "",

    # Threshold used
    eff_prob_thresh: float = 0.75,

    # Gate results ("PASS" / "FAIL" / "SKIP")
    gate_prob: str = "SKIP",
    gate_conv: str = "SKIP",
    gate_rs: str = "SKIP",
    gate_wick: str = "SKIP",
    gate_whipsaw: str = "SKIP",
    gate_losses: str = "SKIP",
    gate_positions: str = "SKIP",

    # Final action
    action: str = "",
    reason: str = "",
    open_positions: int = 0,
    live_pnl: float = 0.0,
) -> None:
    """
    Append one row to today's debug CSV.

    Call this once per new-brick event, AFTER all gate logic has resolved.
    Pass gate results as "PASS", "FAIL", or "SKIP" (not evaluated because
    an earlier gate already blocked the signal).

    Example:
        from src.live.daily_logger import log_brick_event
        log_brick_event(
            ts=now, symbol=sym, sector=st.sector,
            price=price, brick_dir=brick_dir, sec_dir=sec_dir,
            new_bricks=len(st.bricks),
            velocity=float(latest.get("velocity", 0)),
            ...
            action="ENTRY", reason="ALL_GATES_PASS",
        )
    """
    date_str = ts.strftime("%Y-%m-%d")
    path = _get_log_file(date_str)

    try:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                symbol, sector,
                round(price, 2),
                brick_dir, sec_dir, new_bricks,
                round(velocity, 6),
                round(wick_pressure, 4),
                round(relative_strength, 4),
                round(brick_size, 2),
                round(duration_seconds, 1),
                consecutive_same,
                round(oscillation_rate, 4),
                round(brain1_prob, 6),
                round(brain2_conv, 2),
                signal, round(score, 2),
                int(global_kill), int(global_pause), int(ticker_paused),
                bias,
                round(eff_prob_thresh, 4),
                gate_prob, gate_conv, gate_rs, gate_wick,
                gate_whipsaw, gate_losses, gate_positions,
                action, reason,
                open_positions, round(live_pnl, 2),
            ])
    except Exception as e:
        logger.error(f"DailyLogger write error for {symbol}: {e}")
