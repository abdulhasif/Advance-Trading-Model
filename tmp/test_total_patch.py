import time
import requests
import upstox_client
from upstox_client.feeder.market_data_streamer_v3 import MarketDataStreamerV3
from upstox_client.feeder.market_data_feeder_v3 import MarketDataFeederV3
import websocket
import threading
import ssl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWJjY2E5NTAyYWIzNTEyZTllNGRjNzAiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mzk4MDMwOSwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0MDQ0MDAwfQ.Gn9KWmBBcsR_o4rpvQA_VHKZOHVgD8RVFETKeLenNxA"

def on_open(ws): print("SUCCESS: Connection Opened!")
def on_error(ws, error): print(f"ERROR: {error}")

def patched_connect(self):
    """Replacement for MarketDataFeederV3.connect that uses the modern Redirect URI."""
    print("PEFORMING PATCHED CONNECT...")
    auth_url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {self.api_client.configuration.access_token}"}
    
    res = requests.get(auth_url, headers=headers)
    if res.status_code != 200:
        print(f"FAILED Auth: {res.text}")
        return
    
    ws_url = res.json()["data"]["authorizedRedirectUri"]
    print(f"Patched URI: {ws_url[:50]}...")

    sslopt = {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
    
    # WebSocketApp handlers in SDK are (ws, ...)
    self.ws = websocket.WebSocketApp(ws_url,
                                     on_open=self.on_open,
                                     on_message=self.on_message,
                                     on_error=self.on_error,
                                     on_close=self.on_close)

    threading.Thread(target=self.ws.run_forever, kwargs={"sslopt": sslopt}).start()

def test_total_patch():
    # Apply global patch
    MarketDataFeederV3.connect = patched_connect

    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    api_client = upstox_client.ApiClient(configuration)

    instrument_keys = ["NSE_EQ|INE062A01020"]
    streamer = MarketDataStreamerV3(api_client, instrument_keys, mode="ltpc")
    
    # Streamer handlers are (event, ...)
    def on_streamer_open(): print("STREAMER OPEN EVENT FIRED!")
    streamer.on("open", on_streamer_open)

    print("Connecting...")
    streamer.connect()
    time.sleep(10)
    streamer.disconnect()

if __name__ == "__main__":
    test_total_patch()
