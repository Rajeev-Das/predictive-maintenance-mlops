# Predictive Maintenance: Remaining Useful Life (RUL) Prediction with LSTM

## predictive-maintenance-mlops

A robust MLOps pipeline for predictive maintenance, focused on estimating the Remaining Useful Life (RUL) of marine engine components using time-series sensor data. The system leverages the NASA CMAPSS Turbofan Engine Degradation Simulation Dataset—adapted here to represent sensor readings from ship engines—to train and serve deep learning models for real-time RUL prediction.

## 🚀 Overview
This repository delivers an end-to-end solution for RUL prediction on marine engine sensor data (NASA CMAPSS), including:
- Automated feature engineering (rolling means, slopes)
- Deep LSTM regression with hyperparameter tuning (KerasTuner)
- Full training, evaluation, and model serving lifecycle
- Modern MLOps project structure with FastAPI RESTful inference

---

## 📁 Project Structure

```text
.
├── main.py                # Entry point: train/evaluate orchestrator
├── api.py                 # FastAPI app for real-time RUL prediction
├── api_client.py          # Example client for API testing
├── src/
│   ├── data/
│   │   └── preprocess.py  # Data loading & cleaning functions
│   ├── features/          # (Optional) Additional feature engineering scripts
│   ├── models/
│   │   └── train_lstm.py  # Model training, tuning
|   |   └── evaluate.py    # Model Evaluation
│   ├── predict/
│   │   └── predict.py     # Batch/production prediction logic
│   └── utils/             # (Optional) Common helper utilities
├── model/                 # Saved models and scalers (artifacts)
│   ├── trained_lstm_rul_model_tuned.keras
│   └── trained_lstm_scaler_tuned.joblib
├── data/                  # Raw and processed data files
│   └── CMAPSSData/
├── notebooks/             # EDA, research, experiment tracking
├── requirements.txt
├── Makefile               # Helper CLI commands
```
---

## 🧑‍💻 How To Run

### 1. **Setup Environment**

```text
python -m venv venv
source venv/bin/activate # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 2. **Train Model**

```text
python main.py --mode train
```

### 3. **Evaluate Model**

```text
python main.py --mode evaluate
```

### 4. **Run API for Inference**

```text
uvicorn api:app --reload
```

or, if your api.py is in src/api/: 
```text
uvicorn src.api.api:app --reload
```

- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. **Test API with Automation Script**

```text
python api_client.py
```

---

## 🔎 Model Performance

| Model        | Test MAE | Test RMSE |
|--------------|----------|-----------|
| LSTM (tuned) | **15.48**| **20.07** |
| LSTM         | 16.82    | 21.84     |
| RF           | 25.31    | 33.44     |

See `/notebooks/` for error analysis and plots.

---

## 🚦 API Usage (Example)

POST to `/predict_rul` with JSON body:
```JSON
{
  "sequence": [
    {
      "op_setting_1": 0.53,
      "op_setting_2": 0.12,
      "sensor_4_rollmean": 519.2,
      "sensor_12": 0.235,
      "sensor_7": 126.2,
      "sensor_21": 39.6,
      "sensor_20": 101.2,
      "sensor_12_rollmean": 0.225,
      "sensor_7_rollmean": 125.9,
      "sensor_21_rollmean": 39.4,
      "sensor_12_slope": -0.0003,
      "sensor_7_slope": 0.10,
      "sensor_21_slope": 0.01
    },
    ... (29 more)
  ]
}

```
Returns:  
```text
{"predicted_RUL": 123.4}
```

| Feature Name        | Type   | Description                      |
|---------------------|--------|----------------------------------|
| op_setting_1        | float  | Engine operational setting 1     |
| op_setting_2        | float  | Engine operational setting 2     |
| sensor_4_rollmean   | float  | Rolling mean of sensor 4         |
| sensor_12           | float  | Raw sensor 12 value              |
| sensor_7            | float  | Raw sensor 7 value               |
| sensor_21           | float  | Raw sensor 21 value              |
| sensor_20           | float  | Raw sensor 20 value              |
| sensor_12_rollmean  | float  | Rolling mean of sensor 12        |
| sensor_7_rollmean   | float  | Rolling mean of sensor 7         |
| sensor_21_rollmean  | float  | Rolling mean of sensor 21        |
| sensor_12_slope     | float  | Rolling slope of sensor 12       |
| sensor_7_slope      | float  | Rolling slope of sensor 7        |
| sensor_21_slope     | float  | Rolling slope of sensor 21       |

**Important Notes:**

* You must fill all 13 variables (keys) for each timestep.

* The list must have 30 timesteps, ordered from oldest to most recent (unless your API expects newest first—your pipeline expects "oldest to newest").

* All values must be floats.
---

## 🛠️ Maintenance & Extensibility

- Modular MLOps design (easy to add new models, preprocessors)
- Ready to ship in Docker/container cloud
- Production-grade API with batch and streaming extensions possible

---

## 📑 License

MIT License – see [LICENSE](LICENSE)
