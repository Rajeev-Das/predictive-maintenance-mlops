import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    X = df.drop(['RUL'], axis=1)
    y = df['RUL']
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, model_path)
