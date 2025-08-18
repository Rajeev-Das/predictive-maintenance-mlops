"""
LSTM model training script for CMAPSS predictive maintenance.

This module loads and preprocesses the CMAPSS dataset, performs feature engineering,
creates sequences for time series modeling, and trains an LSTM model to predict
the Remaining Useful Life (RUL) of turbofan engines. Includes hyperparameter tuning
using Keras Tuner.
"""
# src/models/train_lstm.py
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Hyperparameter tuning
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.data.preprocess import load_and_clean_cmapss, add_rul
from src.features.build_features import add_rolling_mean, add_rolling_slope
from src.utils.utils import create_sequences, SEQUENCE_LENGTH, features

def build_lstm_model(hp):
    """
    Build and compile an LSTM model for Remaining Useful Life (RUL) prediction with hyperparameter tuning.

    This function constructs a Keras Sequential LSTM model using hyperparameters provided by Keras Tuner.
    The model architecture includes one or two LSTM layers (with optional dropout), followed by a dense output layer.
    The input shape is determined by SEQUENCE_LENGTH and the number of features.

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameter search space object from Keras Tuner.

    Returns:
        tensorflow.keras.Model: Compiled LSTM model ready for training.
    """
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, len(features))))
    model.add(
        LSTM(
            units=hp.Int("units1", min_value=32, max_value=128, step=32),
            activation="tanh",
            return_sequences=True,
        )
    )
    model.add(Dropout(hp.Float("dropout1", 0.1, 0.5, step=0.1)))
    if hp.Boolean("second_lstm"):
        model.add(
            LSTM(
                units=hp.Int("units2", min_value=16, max_value=64, step=16),
                activation="tanh",
                return_sequences=False,
            )
        )
        model.add(Dropout(hp.Float("dropout2", 0.1, 0.5, step=0.1)))
    else:
        model.add(
            LSTM(
                units=hp.Int("units2_flat", min_value=16, max_value=64, step=16),
                activation="tanh",
                return_sequences=False,
            )
        )
    model.add(Dense(1))
    model.compile(loss="mae", optimizer=Adam(), metrics=["mae"])
    return model


def main():
    """
    Main function to load data, perform feature engineering, create sequences, scale features,
    split data, and run hyperparameter tuning for the LSTM model for Remaining Useful Life (RUL) prediction.

    This function orchestrates the end-to-end workflow for preparing the CMAPSS dataset,
    building and tuning an LSTM model using Keras Tuner, and preparing the data for model training and validation.
    """
    # Load & prepare data
    df = load_and_clean_cmapss('data/CMAPSSData/train_FD001.txt')
    df = add_rul(df)
    df = add_rolling_mean(df, ["sensor_4", "sensor_12", "sensor_7", "sensor_21"], 5)
    df = add_rolling_slope(df, ["sensor_12", "sensor_7", "sensor_21"], 5)
    df = df.dropna(subset=features)
    # Sequence data
    X_seq, y_seq = create_sequences(df, SEQUENCE_LENGTH, features)
    scaler = StandardScaler()
    nsamples, nsteps, nfeatures = X_seq.shape
    X_seq_flat = X_seq.reshape(-1, nfeatures)
    X_seq_flat_scaled = scaler.fit_transform(X_seq_flat)
    X_seq_scaled = X_seq_flat_scaled.reshape(nsamples, nsteps, nfeatures)
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq_scaled, y_seq, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning
    tuner = kt.RandomSearch(
        build_lstm_model,
        objective="val_mae",
        max_trials=10,
        executions_per_trial=1,
        directory="../../model/lstm_tuning",
        project_name="fd001_rul",
    )
    tuner.search(
        X_train,
        y_train,
        epochs=15,
        validation_data=(X_val, y_val),
        batch_size=64,
        verbose=2,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best LSTM hyperparameters:", best_hp.values)
    # Retrain with best parameters on full data
    model = build_lstm_model(best_hp)
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(
        X_full, y_full, epochs=15, batch_size=64, callbacks=[early_stop], verbose=2
    )
    model.save("model/trained_lstm_rul_model_tuned.keras")
    joblib.dump(scaler, "model/trained_lstm_scaler_tuned.joblib")


if __name__ == "__main__":
    main()
