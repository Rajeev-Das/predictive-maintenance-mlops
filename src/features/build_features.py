"""
Feature engineering utilities for CMAPSS dataset.

This module provides functions to add rolling mean and rolling slope features
to the CMAPSS turbofan engine degradation data, grouped by unit.
"""
import numpy as np


def add_rolling_mean(df, sensors, window):
    """
    Add rolling mean features for specified sensors to the DataFrame.

    For each sensor in the provided list, computes the rolling mean over a specified window,
    grouped by 'unit', and adds the result as a new column named '{sensor}_rollmean'.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'unit' and sensor columns.
        sensors (list): List of sensor column names to compute rolling means for.
        window (int): Window size for the rolling mean.

    Returns:
        pd.DataFrame: DataFrame with new rolling mean feature columns added.
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

    For each sensor in the provided list, computes the rolling slope (linear regression slope)
    over a specified window, grouped by 'unit', and adds the result as a new column named '{sensor}_slope'.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'unit' and sensor columns.
        sensors (list): List of sensor column names to compute rolling slopes for.
        window (int): Window size for the rolling slope calculation.

    Returns:
        pd.DataFrame: DataFrame with new rolling slope feature columns added.
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


if __name__ == "__main__":
    from src.data.preprocess import load_and_clean_cmapss, add_rul

    df = load_and_clean_cmapss("data/CMAPSSData/train_FD001.txt")
    df = add_rul(df)
    rolling_sensors = ["sensor_4", "sensor_12", "sensor_7", "sensor_21"]
    slope_sensors = ["sensor_12", "sensor_7", "sensor_21"]
    df = add_rolling_mean(df, rolling_sensors, window=5)
    df = add_rolling_slope(df, slope_sensors, window=5)
    print(df.head())
