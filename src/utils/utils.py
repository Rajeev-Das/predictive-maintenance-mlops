"""
Utility functions for sequence creation and time series processing.

This module provides helper functions for preparing data sequences for
predictive maintenance models, such as generating sliding windows of features
and targets for training and test sets.
"""
import numpy as np

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


def create_sequences(df, seq_length, feature_list, rul_col="RUL"):
    """
    Create sequences of features and corresponding RUL targets for time series modeling.

    For each unique unit in the dataframe, this function generates sliding windows of length `seq_length`
    over the specified `feature_list` columns, and collects the corresponding RUL value at the end of each window.

    Args:
        df (pd.DataFrame): DataFrame containing the data with 'unit' and 'cycle' columns.
        seq_length (int): The length of the sequence window.
        feature_list (list): List of feature column names to include in the sequence.
        rul_col (str, optional): Name of the column containing the RUL target. Defaults to 'RUL'.

    Returns:
        tuple: (X, y)
            X (np.ndarray): Array of shape (num_samples, seq_length, num_features) containing the feature sequences.
            y (np.ndarray): Array of shape (num_samples,) containing the RUL targets.
    """
    X, y = [], []
    for unit_num in df["unit"].unique():
        sub = df[df["unit"] == unit_num].sort_values("cycle")
        feature_array = sub[feature_list].values
        rul_array = sub[rul_col].values
        if len(sub) >= seq_length:
            for i in range(seq_length - 1, len(sub)):
                X.append(feature_array[i - seq_length + 1 : i + 1, :])
                y.append(rul_array[i])
    return np.array(X), np.array(y)


def create_test_sequences(df, seq_length, feature_list):
    """
    Create test sequences for each unit in the dataframe.

    For each unique unit, extracts the last `seq_length` rows (ordered by 'cycle')
    and collects the features specified in `feature_list` to form a sequence.
    Returns a numpy array of shape (num_units, seq_length, num_features) and a list of unit numbers.

    Args:
        df (pd.DataFrame): DataFrame containing the test data with 'unit' and 'cycle' columns.
        seq_length (int): The length of the sequence window.
        feature_list (list): List of feature column names to include in the sequence.

    Returns:
        tuple: (X_test_seq, unit_list)
            X_test_seq (np.ndarray): Array of shape (num_units, seq_length, num_features).
            unit_list (list): List of unit numbers corresponding to each sequence.
    """
    X_test_seq = []
    unit_list = []
    for unit_num in df["unit"].unique():
        sub = df[df["unit"] == unit_num].sort_values("cycle")
        if len(sub) >= seq_length:
            x = sub[feature_list].values[-seq_length:]  # last window
            X_test_seq.append(x)
            unit_list.append(unit_num)
    return np.array(X_test_seq), unit_list
