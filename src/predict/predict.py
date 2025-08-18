"""
Prediction utilities for LSTM-based Remaining Useful Life (RUL) inference.

This module provides functions to load a trained LSTM model and scaler,
preprocess input sequences, and generate RUL predictions for new data windows.
"""
# src/predict/predict.py
import joblib
from tensorflow.keras.models import load_model

def predict_rul_window(X_seq, lstm_model_path, scaler_path):
    """
    Predict Remaining Useful Life (RUL) for a window of input sequences using a trained LSTM model.

    Args:
        X_seq (np.ndarray): Input sequence data of shape (num_samples, sequence_length, num_features).
        lstm_model_path (str): Path to the trained LSTM model file (.keras or .h5).
        scaler_path (str): Path to the fitted scaler file (e.g., .joblib).

    Returns:
        np.ndarray: Predicted RUL values as a 1D array.
    """
    scaler = joblib.load(scaler_path)
    nsamples, nsteps, nfeatures = X_seq.shape
    X_seq_flat = X_seq.reshape(-1, nfeatures)
    X_seq_flat_scaled = scaler.transform(X_seq_flat)
    X_seq_scaled = X_seq_flat_scaled.reshape(nsamples, nsteps, nfeatures)
    model = load_model(lstm_model_path)
    return model.predict(X_seq_scaled).flatten()
