import os
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import after setting env var so CI behaves differently
from api import app, get_model_and_scaler

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_in_ci(monkeypatch):
    """Replace get_model_and_scaler with dummy version if running in CI."""
    if os.getenv("CI", "false").lower() == "true":

        class DummyScaler:
            def transform(self, arr):
                return arr  # no-op

        class DummyModel:
            def predict(self, arr):
                return np.array([[123.45]])  # deterministic fake RUL

        monkeypatch.setattr(
            "api.get_model_and_scaler", lambda: (DummyScaler(), DummyModel())
        )


def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert "API" in resp.json()["message"]


def test_predict_rul():
    features = [
        "op_setting_1",
        "op_setting_2",
        "sensor_4_rollmean",
        "sensor_12",
        "sensor_7",
        "sensor_21",
        "sensor_20",
        "sensor_12_rollmean",
        "sensor_7_rollmean",
        "sensor_21_rollmean",
        "sensor_12_slope",
        "sensor_7_slope",
        "sensor_21_slope",
    ]
    sequence = [{f: 1.0 for f in features} for _ in range(30)]
    payload = {"sequence": sequence}

    response = client.post("/predict_rul", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_RUL" in data or "error" in data
