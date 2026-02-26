import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Mock config and logger for the test
class MockConfig:
    LOGS_DIR = Path("storage/logs")
config = MockConfig()

import logging
logger = logging.getLogger("test")

# Import the function to test
# We need to make sure the environment is set up to find src
sys.path.append(str(Path.cwd()))

from src.api.server import get_history

async def test():
    print("Testing get_history...")
    
    # Test 1: No filters
    res = await get_history()
    print(f"Total trades: {len(res) if isinstance(res, list) else res}")
    
    # Test 2: Start date filter (Today)
    today = datetime.now().strftime("%Y-%m-%d")
    res_today = await get_history(start_date=today)
    print(f"Trades today ({today}): {len(res_today) if isinstance(res_today, list) else res_today}")
    
    # Test 3: Invalid date
    res_err = await get_history(start_date="invalid")
    print(f"Invalid date response: {res_err}")

if __name__ == "__main__":
    asyncio.run(test())
