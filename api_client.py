"""
API client utilities for Predictive Maintenance RUL FastAPI service.

This module provides helper functions to interact with the FastAPI RUL prediction endpoint,
including sequence formatting and HTTP request handling for inference.
Intended for use in testing, batch inference, or integration with other systems.
"""

import pandas as pd
import numpy as np
import json
import requests

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


def add_rolling_mean(df, sensors, window):
    """
    Add rolling mean features for specified sensors to the DataFrame.

    For each sensor in the provided list, this function computes the rolling mean
    over a specified window size for each unit, and adds the result as a new column
    named '{sensor}_rollmean' to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'unit' and sensor columns.
        sensors (list of str): List of sensor column names to compute rolling means for.
        window (int): Size of the rolling window for mean calculation.

    Returns:
        pd.DataFrame: DataFrame with new '{sensor}_rollmean' columns added.
    """
    for sensor in sensors:
        df[f"{sensor}_rollmean"] = (
            df.groupby("unit")[sensor]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def add_rolling_slope(df, sensors, window):
    """
    Add rolling slope features for specified sensors to the DataFrame.

    For each sensor in the provided list, this function computes the slope of a linear fit
    over a rolling window (of specified size) for each unit, and adds the result as a new column
    named '{sensor}_slope' to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'unit' and sensor columns.
        sensors (list of str): List of sensor column names to compute rolling slopes for.
        window (int): Size of the rolling window for slope calculation.

    Returns:
        pd.DataFrame: DataFrame with new '{sensor}_slope' columns added.
    """
    def rolling_slope(x):
        def linreg(x):
            if len(x) < 2:
                return np.nan
            idx = np.arange(len(x))
            return np.polyfit(idx, x, 1)[0]

        return x.rolling(window, min_periods=2).apply(linreg, raw=True)

    for sensor in sensors:
        df[f"{sensor}_slope"] = (
            df.groupby("unit")[sensor]
            .apply(rolling_slope)
            .reset_index(level=0, drop=True)
        )
    return df


def get_sample_payload(unit_id=1):
    """
    Generate a sample payload for the RUL prediction API using data from a specified unit.

    This function loads the CMAPSS test dataset, applies necessary feature engineering
    (rolling means and slopes), and extracts the most recent SEQUENCE_LENGTH timesteps
    for the given unit_id. It returns a payload dictionary formatted for the API,
    where each timestep is represented as a dictionary of feature values.

    Args:
        unit_id (int, optional): The unit number to extract the sequence from. Defaults to 1.

    Returns:
        dict: A payload dictionary with a "sequence" key containing a list of feature dictionaries,
              ready to be sent to the RUL prediction API.

    Raises:
        ValueError: If the specified unit does not have enough cycles for a full sequence.
    """
    # Load your test set or use live/latest data here!
    df = pd.read_csv(
        "data/CMAPSSData/test_FD001.txt", sep="\\s+", header=None, index_col=None
    )
    col_names = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    df.columns = col_names
    drop_cols = [
        "op_setting_3",
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]
    df = df.drop(columns=drop_cols)
    df = add_rolling_mean(df, ["sensor_4", "sensor_12", "sensor_7", "sensor_21"], 5)
    df = add_rolling_slope(df, ["sensor_12", "sensor_7", "sensor_21"], 5)
    df = df.dropna(subset=features)
    seq_df = df[df["unit"] == unit_id].sort_values("cycle").tail(SEQUENCE_LENGTH)
    if len(seq_df) < SEQUENCE_LENGTH:
        raise ValueError(f"Unit {unit_id} does not have enough cycles for a sequence.")
    payload = {
        "sequence": [
            {f: float(seq_df.iloc[i][f]) for f in features}
            for i in range(SEQUENCE_LENGTH)
        ]
    }
    print(json.dumps(payload, indent=2))
    return payload


def call_api(payload, url="http://127.0.0.1:8000/predict_rul"):
    """
    Send a POST request to the specified RUL prediction API endpoint with the given payload.

    Args:
        payload (dict): The JSON-serializable payload containing the input sequence for RUL prediction.
        url (str, optional): The API endpoint URL. Defaults to "http://127.0.0.1:8000/predict_rul".

    Returns:
        dict: The JSON response from the API, containing either the predicted RUL or an error message.
    """
    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    print("Status code:", res.status_code)
    print("Returned JSON:", res.json())
    return res.json()


if __name__ == "__main__":
    import sys

    # Default unit_id is 30 if not provided
    if len(sys.argv) > 1:
        UNIT_ID = sys.argv[1]
        try:
            unit_id = int(UNIT_ID)
        except ValueError:
            print(f"Invalid unit_id '{UNIT_ID}', must be an integer.")
            sys.exit(1)
    else:
        UNIT_ID = 30

    payload = get_sample_payload(unit_id=UNIT_ID)
    call_api(payload)
