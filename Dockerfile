# ---------- Base Python image ----------
    FROM python:3.10-slim

    # ---------- Set working directory ----------
    WORKDIR /app
    
    # ---------- Set environment variables ----------
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    ENV MODEL_PATH=/app/model/trained_lstm_rul_model_tuned.keras
    ENV SCALER_PATH=/app/model/trained_lstm_scaler_tuned.joblib
    
    # ---------- Install system dependencies (if needed, e.g., for numpy/pandas) ----------
    RUN apt-get update && \
        apt-get install -y build-essential gcc && \
        rm -rf /var/lib/apt/lists/*
    
    # ---------- Copy requirements first (for cache efficiency) ----------
    COPY requirements.txt /app/
    
    # ---------- Install Python dependencies ----------
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt
    
    # ---------- Copy the rest of the app ----------
    COPY . /app/
    
    # ---------- Expose the FastAPI port ----------
    EXPOSE 8000
    
    # ---------- Run FastAPI server ----------
    CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
    