# src/app.py
import os
import joblib
import pandas as pd
import seaborn as sns
import time
import psutil
import json
from datetime import datetime

from pydantic import BaseModel, Field

from prometheus_client import start_http_server, Gauge, Histogram, Counter
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.middleware.wsgi import WSGIMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import make_wsgi_app, generate_latest, CONTENT_TYPE_LATEST

from src.feature_engineering import get_finance_df, get_xx_dropna
from src.model_functions import make_predictions

import mlflow
import mlflow.sklearn

COMPANY = 'PETR4.SA'
STOCK_VAR = 'Adj Close'

SCALER_PATH = 'src/artifacts/transformers/scaler'
MODEL_PATH = 'src/artifacts/models_tf/best_models'

batch_size = 30

class Request(BaseModel):
    end_date: str = Field(default='2024-11-15', min_length=10, max_length=10)
    start_date: str = Field(default='2024-06-01', min_length=10, max_length=10)
    seq_length: int = Field(default=20, gt=0)
    horizon: int = Field(default=1, gt=0)

app = FastAPI()

# Load the model pipeline
try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print("Model loaded successfully")
    scaler = mlflow.sklearn.load_model(SCALER_PATH)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading model /scaler: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    print(f"Files in src directory: {os.listdir('./src')}")
    model = None

# Prometheus metrics for performance monitoring
response_time_histogram = Histogram(
    "predict_response_time_seconds",
    "Response time for predict endpoint in seconds",
    buckets=[0.1,0.3,0.6,0.9,1.2]  # Adjust these buckets as needed
)
cpu_usage_gauge = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_usage_gauge = Gauge("memory_usage_percent", "Memory usage percentage")
query_counter = Counter("predict_queries_total", "Total number of queries to the predict endpoint")

# Create Prometheus metrics for drift monitoring
data_drift_gauge = Gauge("data_drift", "Data Drift Score")
concept_drift_gauge = Gauge("concept_drift", "Concept Drift Score")

# Prometheus Gauge for predictions
prediction_gauge = Gauge("stock_prediction", "Predicted stock prices", ["day"])

@app.get("/health")
async def health_check():
    return {"status":"OK"}

@app.post("/predict")
async def predict(request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    # Increment query counter for QPM calculation
    query_counter.inc()

    # Record resource usage before processing
    cpu_usage_gauge.set(psutil.cpu_percent())
    memory_usage_gauge.set(psutil.virtual_memory().percent)

    # Start timer for response time
    start_time = time.time()

    # data = await request.json()
    df = get_finance_df(COMPANY, request.start_date, request.end_date, STOCK_VAR)

    if (len(df) <= request.seq_length+request.horizon+1):
        raise Exception("Your dataset 'df' has less samples than the defined window size for data transformation, given by 'seq_length'")

    X, _ = get_xx_dropna(df, COMPANY)

    y_pred = make_predictions(X, X, request.seq_length, batch_size, scaler, model).tolist()

    time.sleep(0.1)
    
    # Calculate response time in milliseconds and observe it in the histogram
    response_time_s = (time.time() - start_time)
    response_time_histogram.observe(response_time_s)
    print(y_pred[0])
    offset = request.seq_length+request.horizon-1
    keys = list(range(offset,offset+len(y_pred[0])))
    print(keys[0])

    df_dict = df.to_dict()

    # Expor as previsões no Prometheus
    for key, prediction in dict(zip(keys, y_pred[0])).items():
        date_label = df_dict["Datetime"][key].strftime("%Y-%m-%d")
        prediction_gauge.labels(day=date_label).set(prediction)
    
    return {"prediction": dict(zip(keys, y_pred[0])),
            "input df": df_dict}


# Mount Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)

    # Run FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
