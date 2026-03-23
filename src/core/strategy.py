"""
src/core/strategy.py - Centralized Strategy Gates & Logic
==========================================================
Ensures perfect parity between Live Engine and Offline Spoofer.
"""

import logging
import numpy as np
import config

logger = logging.getLogger(__name__)

def check_entry_gates(
    symbol: str, 
    now, 
    price: float, 
    b1p: float, 
    b2c: float, 
    signal_str: str, 
    rel_str: float, 
    wick_p: float, 
    z_vwap: float, 
    streak_count: int, 
    brick_dir: int,
    recent_dirs: list[int],
    stock_losses: int,
    portfolio_size: int,
    is_already_in_position: bool
) -> tuple[bool, str, dict]:
    """
    Evaluates all 12 trading gates. Returns (True, "", audit_dict) if all pass, 
    else (False, "Reason", audit_dict).
    """
    audit = {
        "gate_prob": "SKIP", "gate_conv": "SKIP", "gate_rs": "SKIP",
        "gate_wick": "SKIP", "gate_whipsaw": "SKIP", "gate_losses": "SKIP",
        "gate_positions": "SKIP", "gate_time": "SKIP", "gate_vwap": "SKIP"
    }
    # 0. Penny Stock Filter
    if price < config.MIN_PRICE_FILTER:
        return False, "PENNY_STOCK", audit

    # 1. Time Gate (Morning lock + No new entry cutoff)
    morning_cutoff_min = config.MARKET_OPEN_MINUTE + config.ENTRY_LOCK_MINUTES
    morning_cutoff_hour = config.MARKET_OPEN_HOUR + (morning_cutoff_min // 60)
    morning_cutoff_min %= 60
    
    is_too_early = (now.hour < morning_cutoff_hour) or \
                   (now.hour == morning_cutoff_hour and now.minute < morning_cutoff_min)
    
    is_too_late = (now.hour > config.NO_NEW_ENTRY_HOUR) or \
                   (now.hour == config.NO_NEW_ENTRY_HOUR and now.minute >= config.NO_NEW_ENTRY_MIN)
    
    if is_too_early or is_too_late:
        audit["gate_time"] = "FAIL"
        return False, "TIME_GATE", audit
    audit["gate_time"] = "PASS"

    # 2. Entry Probability (Brain 1)
    thresh = config.LONG_ENTRY_PROB_THRESH if signal_str == "LONG" else config.SHORT_ENTRY_PROB_THRESH
    if not config.USE_CALIBRATED_MODELS:
        thresh = config.RAW_LONG_ENTRY_PROB_THRESH if signal_str == "LONG" else config.RAW_SHORT_ENTRY_PROB_THRESH
    
    if b1p < thresh:
        audit["gate_prob"] = "FAIL"
        return False, "LOW_PROB", audit
    audit["gate_prob"] = "PASS"

    # 3. Conviction (Brain 2)
    if b2c < config.ENTRY_CONV_THRESH:
        audit["gate_conv"] = "FAIL"
        return False, "LOW_CONVICTION", audit
    audit["gate_conv"] = "PASS"

    # 4. Soft Veto (Sector Alignment)
    if b2c < config.VETO_BYPASS_CONV:
        if signal_str == "LONG" and rel_str < -config.SOFT_VETO_THRESHOLD:
            audit["gate_rs"] = "FAIL_VETO"
            return False, "SECTOR_VETO", audit
        if signal_str == "SHORT" and rel_str > config.SOFT_VETO_THRESHOLD:
            audit["gate_rs"] = "FAIL_VETO"
            return False, "SECTOR_VETO", audit
    audit["gate_rs"] = "PASS"

    # 5. RS Anchor (Leader/Laggard)
    if b2c < config.VETO_BYPASS_CONV:
        if signal_str == "LONG" and rel_str < config.ENTRY_RS_THRESHOLD:
            audit["gate_rs"] = "FAIL_RS"
            return False, "LOW_RS", audit
        if signal_str == "SHORT" and rel_str > -config.ENTRY_RS_THRESHOLD:
            audit["gate_rs"] = "FAIL_RS"
            return False, "LOW_RS", audit
    audit["gate_rs"] = "PASS"

    # 6. Wick Trap
    if wick_p > config.MAX_ENTRY_WICK:
        audit["gate_wick"] = "FAIL"
        return False, "WICK_PRESSURE", audit
    audit["gate_wick"] = "PASS"

    # 7. VWAP Exhaustion
    if signal_str == "LONG" and z_vwap > config.MAX_VWAP_ZSCORE:
        audit["gate_vwap"] = "FAIL"
        return False, "VWAP_EXHAUSTION", audit
    if signal_str == "SHORT" and z_vwap < -config.MAX_VWAP_ZSCORE:
        audit["gate_vwap"] = "FAIL"
        return False, "VWAP_EXHAUSTION", audit
    audit["gate_vwap"] = "PASS"

    # 8. Whipsaw Guard (Consecutive Bricks)
    if len(recent_dirs) < config.MIN_CONSECUTIVE_BRICKS:
        audit["gate_whipsaw"] = "FAIL_MIN"
        return False, "WHIPSAW_MIN_BRICKS", audit
    
    expected_dir = 1 if signal_str == "LONG" else -1
    if not all(d == expected_dir for d in recent_dirs):
        audit["gate_whipsaw"] = "FAIL_DIR"
        return False, "WHIPSAW_MIXED_DIR", audit
    audit["gate_whipsaw"] = "PASS"

    # 9. Streak / FOMO Limit
    if streak_count >= config.STREAK_LIMIT:
        return False, "STREAK_EXHAUSTION", audit

    # 10. Daily Stock Loss Limit
    if stock_losses >= config.MAX_LOSSES_PER_STOCK:
        audit["gate_losses"] = "FAIL"
        return False, "DAILY_LOSS_LIMIT", audit
    audit["gate_losses"] = "PASS"

    # 11. Already in Position
    if is_already_in_position:
        return False, "ALREADY_IN_POSITION", audit

    # 12. Portfolio Cap
    if portfolio_size >= config.MAX_OPEN_POSITIONS:
        audit["gate_positions"] = "FAIL"
        return False, "MAX_POSITIONS", audit
    audit["gate_positions"] = "PASS"

    return True, "", audit

def check_exit_conditions(
    order_side: str,
    entry_price: float,
    current_price: float,
    brick_size: float,
    b2c: float,
    p_long: float,
    p_short: float
) -> str | None:
    """
    Evaluates exit conditions. Returns reason string if exit triggered, 
    else None.
    """
    # 1. Conviction Drop
    if b2c < config.EXIT_CONV_THRESH:
        return "LOW_CONVICTION"
    
    # 2. Hysteresis Reversal
    if order_side == "BUY" and p_short > (1.0 - config.HYST_LONG_SELL_FLOOR):
        return "TREND_REVERSAL"
    if order_side == "SELL" and p_long > config.HYST_SHORT_SELL_CEIL:
        return "TREND_REVERSAL"
        
    # 3. Structural SL (Adverse Bricks)
    adverse_dist = (entry_price - current_price) if order_side == "BUY" else (current_price - entry_price)
    adv_bricks = adverse_dist / brick_size
    if adv_bricks >= config.STRUCTURAL_REVERSAL_BRICKS:
        return "STOP_LOSS_STRUCTURAL"

    return None
