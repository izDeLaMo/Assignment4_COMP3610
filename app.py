from duckdb import df
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import pandas as pd
import uuid
import os

# -------------------------
# Load model + preprocessor once
# -------------------------
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

model_version = "v1"

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Taxi Tip Prediction API", version="1.0")

# -------------------------
# Input schema (Pydantic)
# -------------------------
class TripInput(BaseModel):
    passenger_count: int = Field(ge=1, le=8)
    trip_distance: float = Field(gt=0)
    fare_amount: float = Field(ge=0)
    total_amount: float = Field(ge=0)
    payment_type: int = Field(ge=1, le=5)
    trip_duration_minutes: float = Field(gt=0)
    trip_speed_mph: float = Field(ge=0)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day_of_week: str
    is_weekend: bool


class BatchInput(BaseModel):
    trips: List[TripInput]


# -------------------------
# Helper
# -------------------------
def preprocess_input(data: pd.DataFrame):
    return preprocessor.transform(data)

def add_features(df: pd.DataFrame):
    # log_trip_distance
    df["log_trip_distance"] = np.log1p(df["trip_distance"])

    # fare_per_mile
    df["fare_per_mile"] = df.apply(
        lambda x: x["fare_amount"] / x["trip_distance"] if x["trip_distance"] != 0 else 0,
        axis=1
    )

    # fare_per_minute
    df["fare_per_minute"] = df.apply(
        lambda x: x["fare_amount"] / x["trip_duration_minutes"] if x["trip_duration_minutes"] != 0 else 0,
        axis=1
    )

    return df
# -------------------------
# Routes
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name": "taxi-tip-regressor",
        "model_version": model_version,
        "features": [
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "payment_type",
            "trip_duration_minutes",
            "trip_speed_mph",
            "pickup_hour",
            "pickup_day_of_week",
            "is_weekend"
        ],
        "metrics": {
            "note": "Use MLflow UI screenshots for MAE/RMSE/R2"
        }
    }


@app.post("/predict")
def predict(trip: TripInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([trip.model_dump()])  # FIXED (pydantic v2)

        # Apply preprocessing
        df = add_features(df)
        X = preprocessor.transform(df)

        # Predict
        pred = model.predict(X)[0]

        return {
            "prediction": round(float(pred), 2),
            "model_version": "v1",
            "prediction_id": str(uuid.uuid4())
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed"}
        )


@app.post("/predict/batch")
def predict_batch(trips: list[TripInput]):
    try:
        df = pd.DataFrame([t.model_dump() for t in trips])

        df = add_features(df)
        X = preprocessor.transform(df)
        preds = model.predict(X)

        results = []
        for p in preds:
            results.append({
                "prediction": round(float(p), 2),
                "model_version": "v1",
                "prediction_id": str(uuid.uuid4())
            })

        return results

    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Batch prediction failed"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))