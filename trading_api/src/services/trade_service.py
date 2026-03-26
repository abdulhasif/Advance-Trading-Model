import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from trading_api import config

logger = logging.getLogger(__name__)

def read_live_state() -> Dict:
    """Read the live_state.json written by the live engine."""
    try:
        p = Path(config.LIVE_STATE_FILE)
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def get_active_trades(simulator_ref: Optional[object] = None) -> List[Dict]:
    """Return active trades - from in-process simulator OR live_state.json fallback."""
    if simulator_ref is not None:
        trades = []
        # simulator_ref is expected to have active_trades attribute
        for sym, order in getattr(simulator_ref, 'active_trades', {}).items():
            trades.append({
                "symbol":         sym,
                "side":           order.side,
                "qty":            order.qty,
                "entry_price":    round(order.entry_price, 2),
                "sl_price":       round(getattr(order, 'sl_price', 0.0), 2),
                "last_price":     round(order.last_price, 2),
                "unrealized_pnl": round(order.unrealized_pnl, 2),
                "locked_margin":  round(order.locked_margin, 2),
                "entry_time":     order.filled_at.isoformat() if order.filled_at else None,
            })
        return trades

    return read_live_state().get("active_trades", [])

def get_historical_trades(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
    """Returns historical trades from paper_trades.csv with optional date filters."""
    trades_log = config.LOGS_DIR / "paper_trades.csv"
    
    if not trades_log.exists():
        return {"status": "error", "detail": "No trade history found"}

    try:
        df = pd.read_csv(trades_log)
        if df.empty:
            return []

        df['dt'] = pd.to_datetime(df['entry_time'], format='mixed', errors='coerce')
        
        if start_date:
            sd = pd.to_datetime(start_date)
            df = df[df['dt'].dt.date >= sd.date()]

        if end_date:
            ed = pd.to_datetime(end_date)
            df = df[df['dt'].dt.date <= ed.date()]

        df = df.drop(columns=['dt']).fillna("")
        return df.to_dict(orient="records")

    except Exception as e:
        logger.error(f"Failed to read trade history: {e}")
        return {"status": "error", "detail": str(e)}

def generate_daily_report(date_str: str) -> Dict:
    """Returns a daily performance report based on paper_trades.csv."""
    trades_log = config.LOGS_DIR / "paper_trades.csv"
    
    if not trades_log.exists():
        return {"status": "error", "detail": "No trade history found"}

    try:
        df = pd.read_csv(trades_log)
        empty_report = {
            "status": "success", "date": date_str, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "sector_pnl": {}, "symbol_pnl": {}
        }

        if df.empty:
            return empty_report

        df['dt'] = pd.to_datetime(df['entry_time'], format='mixed', errors='coerce')
        target_date = pd.to_datetime(date_str).date()
        day_df = df[df['dt'].dt.date == target_date]

        if day_df.empty:
            return empty_report

        total_trades = len(day_df)
        wins = len(day_df[day_df['net_pnl'] > 0])
        win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0.0
        total_pnl = round(day_df['net_pnl'].sum(), 2)

        sector_pnl = day_df.groupby('sector')['net_pnl'].sum().round(2).to_dict()
        symbol_pnl = day_df.groupby('symbol')['net_pnl'].sum().round(2).to_dict()

        return {
            "status": "success",
            "date": date_str,
            "total_trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "sector_pnl": sector_pnl,
            "symbol_pnl": symbol_pnl
        }

    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        return {"status": "error", "detail": str(e)}

