from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.predict import load_model, predict_rul

app = FastAPI()
model = load_model("model/model.joblib")

class SensorInput(BaseModel):
    # Define all fields you need based on your features
    temperature: float
    pressure: float
    rpm: float
    vibration: float

@app.post("/predict")
def predict(sensor: SensorInput):
    features = np.array([sensor.temperature, sensor.pressure, sensor.rpm, sensor.vibration])
    rul = predict_rul(model, features)
    return {"predicted_rul": rul}
