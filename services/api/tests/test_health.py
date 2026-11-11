from api.main import app
from fastapi.testclient import TestClient


def test_liveness() -> None:
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
