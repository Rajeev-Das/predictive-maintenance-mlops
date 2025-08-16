import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    # Load raw data CSV
    return pd.read_csv(path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering logic here
    # e.g., rolling means, diffs, normalization
    return df

def save_processed(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
