import joblib
import numpy as np
from tensorflow.keras.models import load_model

def test_model_load_and_infer():
    scaler = joblib.load('model/trained_lstm_scaler_tuned.joblib')
    model = load_model('model/trained_lstm_rul_model_tuned.keras')
    # Dummy sequence: shape (1, sequence_length, n_features)
    arr = np.ones((1, 30, 13), dtype=np.float32)
    arr_flat = arr.reshape(-1, 13)
    arr_flat_scaled = scaler.transform(arr_flat)
    arr_scaled = arr_flat_scaled.reshape(1, 30, 13)
    y_pred = model.predict(arr_scaled)
    assert y_pred.shape == (1,1)
    assert np.isfinite(y_pred).all()
