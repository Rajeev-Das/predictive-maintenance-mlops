from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "temperature": 100, "pressure": 1.5, "rpm": 3000, "vibration": 0.5
    })
    assert response.status_code == 200
    assert "predicted_rul" in response.json()
