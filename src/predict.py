import joblib
import numpy as np

def load_model(model_path: str):
    return joblib.load(model_path)

def predict_rul(model, features: np.ndarray) -> float:
    prediction = model.predict(features.reshape(1, -1))
    return prediction[0]
