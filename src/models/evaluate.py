"""
Evaluation script for LSTM model on CMAPSS test set.

This module loads the trained LSTM model and scaler, preprocesses the CMAPSS test data,
applies feature engineering, creates test sequences, and evaluates the model's performance
by computing MAE and RMSE for Remaining Useful Life (RUL) prediction.
"""

# src/models/evaluate.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from src.data.preprocess import load_and_clean_cmapss, add_rul
from src.features.build_features import add_rolling_mean, add_rolling_slope
from src.utils.utils import create_test_sequences


def main():
    """
    Main function to evaluate the trained LSTM model on the CMAPSS test set.

    This function loads and preprocesses the test data, applies feature engineering,
    creates sequences for each test unit, scales the features using the saved scaler,
    loads the trained LSTM model, and computes predictions for Remaining Useful Life (RUL).
    It then compares the predictions to the true RUL values and prints the MAE and RMSE metrics.
    """
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
    # Load and preprocess test data
    test = load_and_clean_cmapss("data/CMAPSSData/test_FD001.txt")
    test = add_rul(test)
    test = add_rolling_mean(test, ["sensor_4", "sensor_12", "sensor_7", "sensor_21"], 5)
    test = add_rolling_slope(test, ["sensor_12", "sensor_7", "sensor_21"], 5)
    test = test.dropna(subset=features)
    # Last sequences/engine
    X_test_seq, test_unit_numbers = create_test_sequences(
        test, SEQUENCE_LENGTH, features
    )
    # The path "../../model_artifacts/lstm_scaler_tuned.joblib" is correct if you
    # are running this script from the "src/models" directory,
    # If you run from the project root, you may need to use
    # "model_artifacts/lstm_scaler_tuned.joblib" instead.
    scaler = joblib.load("model/trained_lstm_scaler_tuned.joblib")
    nsamples, nsteps, nfeatures = X_test_seq.shape
    X_test_seq_flat = X_test_seq.reshape(-1, nfeatures)
    X_test_scaled_flat = scaler.transform(X_test_seq_flat)
    X_test_scaled = X_test_scaled_flat.reshape(nsamples, nsteps, nfeatures)
    model = load_model("model/trained_lstm_rul_model_tuned.keras")
    y_pred_lstm = model.predict(X_test_scaled).flatten()
    true_rul = pd.read_csv(
        "data/CMAPSSData/RUL_FD001.txt", sep="\\s+", names=["true_RUL"]
    )
    y_true = true_rul["true_RUL"].values[: len(y_pred_lstm)]
    mae = mean_absolute_error(y_true, y_pred_lstm)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_lstm))
    print(f"Tuned LSTM Test set MAE: {mae:.2f}, RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
