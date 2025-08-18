"""
Preprocessing utilities for CMAPSS dataset.

This module provides functions to load, clean, and preprocess the CMAPSS turbofan engine degradation data,
including RUL (Remaining Useful Life) calculation.
"""

import pandas as pd


def load_and_clean_cmapss(filepath):
    """
    Load and clean the CMAPSS dataset from the given file path.

    Reads the raw CMAPSS data file, assigns appropriate column names, and drops
    columns that are not useful for modeling (as per domain knowledge).

    Args:
        filepath (str): Path to the raw CMAPSS data file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with selected columns.
    """
    col_names = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
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

    df = pd.read_csv(filepath, sep="\\s+", header=None, index_col=None)
    df.columns = col_names
    df = df.drop(columns=drop_cols)
    return df


def add_rul(df):
    """
    Add Remaining Useful Life (RUL) column to the DataFrame.

    For each unit, calculates the maximum cycle and computes RUL as the difference
    between the maximum cycle and the current cycle for each row.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'unit' and 'cycle' columns.

    Returns:
        pd.DataFrame: DataFrame with an added 'RUL' column.
    """
    df["max_cycle"] = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])
    return df


if __name__ == "__main__":
    df = load_and_clean_cmapss("data/CMAPSSData/train_FD001.txt")
    df = add_rul(df)
    print(df.head())
