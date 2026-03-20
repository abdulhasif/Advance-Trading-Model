import requests
import websocket
import ssl
import json
import time

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWJjYzNhOGFlYjRmYjE5NWE3NDQ0MjkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3Mzk3ODUzNiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0MDQ0MDAwfQ.CbUhLk5GoybY2qZlJlSvD9fQXrFmFwCXQDH9AHVDG_4"

def test_handshake():
    print("1. Fetching Authorized Redirect URI...")
    url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"FAILED: {res.text}")
        return
    
    ws_uri = res.json()["data"]["authorizedRedirectUri"]
    print(f"SUCCESS: {ws_uri[:60]}...")

    print("\n2. Attempting WebSocket Handshake...")
    try:
        ws = websocket.create_connection(ws_uri, sslopt={"cert_reqs": ssl.CERT_NONE})
        print("SUCCESS: Handshake complete! Connection open.")
        ws.close()
    except Exception as e:
        print(f"FAILED Handshake: {e}")

if __name__ == "__main__":
    test_handshake()
