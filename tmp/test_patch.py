import time
import requests
import upstox_client
from upstox_client.feeder import MarketDataStreamerV3
from upstox_client.feeder.feeder import Feeder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWJjYzNhOGFlYjRmYjE5NWE3NDQ0MjkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mzk3ODUzNiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0MDQ0MDAwfQ.CbUhLk5GoybY2qZlJlSvD9fQXrFmFwCXQDH9AHVDG_4"

def on_open(): print("SUCCESS: Connection Opened!")
def on_error(error): print(f"ERROR: {error}")

def test_monkey_patch():
    print("1. Fetching Authorized Redirect URI...")
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"FAILED Auth: {res.text}")
        return
    
    ws_uri = res.json()["data"]["authorizedRedirectUri"]
    print(f"SUCCESS: {ws_uri[:60]}...")

    print("\n2. Initializing SDK with Monkey-Patch...")
    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)

    # Monkey-patch the internal _get_authorized_url of the streamer's base class
    original_get_auth = MarketDataStreamerV3._get_authorized_url
    MarketDataStreamerV3._get_authorized_url = lambda self: ws_uri

    try:
        instrument_keys = ["NSE_EQ|INE062A01020"]
        streamer = MarketDataStreamerV3(api_client, instrument_keys, mode="ltpc")
        streamer.on("open", on_open); streamer.on("error", on_error)

        print("Connecting...")
        streamer.connect()
        time.sleep(5)
        streamer.disconnect()
    finally:
        # Restore original for safety
        MarketDataStreamerV3._get_authorized_url = original_get_auth

if __name__ == "__main__":
    test_monkey_patch()
