"""
FastAPI service for Remaining Useful Life (RUL) prediction using a trained LSTM model.

This API receives a sequence of sensor and operational data, preprocesses it,
applies the trained scaler, and returns the predicted RUL for the input sequence.
Intended for use in predictive maintenance applications with the CMAPSS dataset.
"""

# api.py
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from tensorflow.keras.models import load_model

# Settings: update paths as needed

MODEL_PATH = os.environ.get("MODEL_PATH", "model/trained_lstm_rul_model_tuned.keras")
SCALER_PATH = os.environ.get("SCALER_PATH", "model/trained_lstm_scaler_tuned.joblib")
SEQUENCE_LENGTH = 30

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


# Pydantic model for input validation
class RULSequence(BaseModel):
    """
    Pydantic model representing a sequence of input data for RUL prediction.

    Each sequence should be a list of dictionaries, where each dictionary contains
    all required feature values for a single timestep. 
    The sequence must be of length SEQUENCE_LENGTH,
    and each dictionary must include all fields specified in the 'features' list.
    """
    sequence: list  # should be a list of dicts, each dict one timestep
    # Each dict must have all fields in 'features' (see below)


def preprocess_sequence(seq_dicts):
    """
    Expects a list of length SEQUENCE_LENGTH, each item a dict with all features.
    Returns: np array of shape (1, SEQUENCE_LENGTH, n_features)
    """
    if len(seq_dicts) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected input sequence of length {SEQUENCE_LENGTH}, got {len(seq_dicts)}."
        )
    arr = np.zeros((SEQUENCE_LENGTH, len(features)), dtype=np.float32)
    for i, feat_dict in enumerate(seq_dicts):
        arr[i] = [feat_dict[f] for f in features]
    return arr[np.newaxis, :, :]  # shape (1, SEQUENCE_LENGTH, n_features)


# Load model and scaler once at startup
_cache = {"model": None, "scaler": None}


def get_model_and_scaler():
    """
    Loads and returns the scaler and model objects for RUL prediction.

    This function caches the loaded model and scaler to avoid reloading them on every request.
    If the model or scaler is not already loaded, it loads them from disk using the specified
    MODEL_PATH and SCALER_PATH. If the files are not found, it raises a FileNotFoundError.

    Returns:
        tuple: (scaler, model) where scaler is the fitted scaler object and model is the trained LSTM model.

    Raises:
        FileNotFoundError: If the model or scaler files are not found at the specified paths.
    """
    if _cache["model"] is None or _cache["scaler"] is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            _cache["scaler"] = joblib.load(SCALER_PATH)
            _cache["model"] = load_model(MODEL_PATH)
        else:
            raise FileNotFoundError("Model/Scaler not found")
    return _cache["scaler"], _cache["model"]


app = FastAPI(title="Predictive Maintenance RUL Inference API")


@app.get("/")
def read_root():
    """
    Root endpoint for the Predictive Maintenance RUL API.

    Returns a welcome message indicating that the API is running.
    """
    return {"message": "Welcome to the Predictive Maintenance RUL API!"}

@app.post("/predict_rul")
def predict_rul_rnn(data: RULSequence):
    """
    Predict the Remaining Useful Life (RUL) for a given input sequence using the trained LSTM model.

    Args:
        data (RULSequence): Input data validated by the RULSequence Pydantic model. 
            Should be a sequence (list) of dictionaries, each containing all required feature values 
            for a single timestep. The sequence must be of length SEQUENCE_LENGTH.

    Returns:
        dict: A dictionary containing the predicted RUL value under the key 'predicted_RUL', 
            or an error message under the key 'error' if an exception occurs.
    """
    try:
        scaler, model = get_model_and_scaler()
    except FileNotFoundError:
        return {"error": "Model files not available"}
    try:
        arr = preprocess_sequence(data.sequence)
        nsamples, nsteps, nfeatures = arr.shape
        arr_flat = arr.reshape(-1, nfeatures)
        arr_flat_scaled = scaler.transform(arr_flat)
        arr_scaled = arr_flat_scaled.reshape(nsamples, nsteps, nfeatures)
        pred = float(model.predict(arr_scaled)[0])
        return {"predicted_RUL": pred}
    except (ValueError, KeyError, RuntimeError) as e:
        return {"error": str(e)}


# (Optional) Health check endpoint
@app.get("/ping")
def ping():
    """
    Health check endpoint for the Predictive Maintenance RUL API.

    Returns a simple message indicating that the API is alive and responsive.
    """
    return {"message": "API is alive!"}
