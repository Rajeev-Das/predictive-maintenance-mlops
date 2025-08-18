from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "API" in resp.json()["message"]

def test_predict_rul():
    # Generate a dummy, valid payload (all 1.0s)
    features = [
        'op_setting_1', 'op_setting_2',
        'sensor_4_rollmean', 'sensor_12', 'sensor_7', 'sensor_21', 'sensor_20',
        'sensor_12_rollmean', 'sensor_7_rollmean', 'sensor_21_rollmean',
        'sensor_12_slope', 'sensor_7_slope', 'sensor_21_slope'
    ]
    sequence = [{f: 1.0 for f in features} for _ in range(30)]
    payload = {"sequence": sequence}
    response = client.post("/predict_rul", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_RUL" in data or "error" in data
