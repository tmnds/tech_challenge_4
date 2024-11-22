# src/app.py
import os
import joblib
import pandas as pd
import seaborn as sns
import time
import psutil
import json

from pydantic import BaseModel, Field

from prometheus_client import start_http_server, Gauge, Histogram, Counter
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.middleware.wsgi import WSGIMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import make_wsgi_app, generate_latest, CONTENT_TYPE_LATEST

from src.data_drift import detect_data_drift
from src.concept_drift import detect_concept_drift
from src.feature_engineering import get_finance_df, shift_drop_na_in_xy
from src.model_functions import make_predictions

import mlflow
import mlflow.sklearn

COMPANY = 'PETR4.SA'
STOCK_VAR = 'Adj Close'

SCALER_PATH = 'artifacts/transformers/scalerX'
MODEL_PATH = 'artifacts/models_tf/best_models'

batch_size = 30

class Request(BaseModel):
    start_date: str
    end_date: str
    seq_length: int
    horizon: int

app = FastAPI()

# Load the model pipeline
try:
    scaler = mlflow.sklearn.load_model(SCALER_PATH)
    print("Scaler loaded successfully")
    model = mlflow.sklearn.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model /scaler: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    model_pipeline = None

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

# Load reference data
# diamonds = sns.load_dataset("diamonds")
# X_reference = diamonds[["carat", "cut", "color", "clarity", "depth", "table"]]
# y_reference = diamonds["price"]

@app.get("/health")
async def health_check():
    return {"status":"OK"}

@app.post("/predict")
async def predict(request: Request):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded properly")

    # Increment query counter for QPM calculation
    query_counter.inc()

    # Record resource usage before processing
    cpu_usage_gauge.set(psutil.cpu_percent())
    memory_usage_gauge.set(psutil.virtual_memory().percent)

    # Start timer for response time
    start_time = time.time()

    data = await request.json()
    df = get_finance_df(COMPANY, request.start_date, request.end_date, STOCK_VAR)

    if (len(df) <= request.seq_length+request.horizon+1):
        raise Exception("Your dataset 'df' has less samples than the defined window size for data transformation, given by 'seq_length'")

    X, _ = shift_drop_na_in_xy(df, COMPANY, COMPANY, horizon_pred=request.horizon)

    y_pred = make_predictions(X, X, request.seq_length, batch_size, scaler)

    time.sleep(0.1)

    # prediction = model_pipeline.predict(df)
    
    # Calculate response time in milliseconds and observe it in the histogram
    response_time_s = (time.time() - start_time)
    response_time_histogram.observe(response_time_s)

    # return {"prediction": prediction[0]}
    return {"prediction": json.dump(y_pred)}

def monitor_drifts():
    # Simulating new data (in a real scenario, this would be actual new data)
    new_diamonds = sns.load_dataset("diamonds").sample(n=1000, replace=True)
    X_current = new_diamonds[["carat", "cut", "color", "clarity", "depth", "table"]]
    y_current = new_diamonds["price"]

    # Data drift detection
    is_data_drift, _, data_drift_score = detect_data_drift(X_reference, X_current)
    data_drift_gauge.set(data_drift_score)

    # Concept drift detection
    is_concept_drift, concept_drift_score = detect_concept_drift(
        model_pipeline,
        X_reference,
        y_reference,
        X_current,
        y_current,
    )
    concept_drift_gauge.set(concept_drift_score)

    if is_data_drift:
        print("Data drift detected!")
    if is_concept_drift:
        print("Concept drift detected!")

# Mount Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)


# Schedule drift monitoring
# scheduler = BackgroundScheduler()
# scheduler.add_job(monitor_drifts, "interval", minutes=1)
# scheduler.start()

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)

    # Run FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
