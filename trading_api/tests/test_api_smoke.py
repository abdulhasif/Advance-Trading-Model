import pytest
try:
    from src.main import app
except ImportError:
    from trading_api.src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    # Note: main.py root is not defined, but it has /health
    response = client.get("/health")
    assert response.status_code == 200
    assert "online" in response.json().get("status", "")

def test_get_history():
    response = client.get("/api/history")
    # Might be error if csv missing, but 200/404/etc.
    assert response.status_code in [200, 404]
