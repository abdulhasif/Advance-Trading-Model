from pydantic import BaseModel
from typing import Optional

class CommandPayload(BaseModel):
    """
    JSON body for the Android command endpoint.

    3-Tier Control Hierarchy:
      Tier 1 - Engine-level:  {"command": "KILL"}
                                {"command": "GLOBAL_PAUSE"}
                                {"command": "GLOBAL_RESUME"}
      Tier 2 - Ticker-level:  {"command": "PAUSE_TICKER",  "ticker": "RELIANCE"}
                                {"command": "RESUME_TICKER", "ticker": "RELIANCE"}
      Tier 3 - Hunter Mode:   {"command": "BIAS",       "ticker": "LT", "direction": "LONG"}
                                {"command": "CLEAR_BIAS", "ticker": "LT"}
      Info:                    {"command": "STATUS"}
    """
    command:   str
    ticker:    Optional[str] = None
    direction: Optional[str] = None   # "LONG" or "SHORT"

