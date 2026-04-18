import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# -----------------------------
# Sample valid input
# -----------------------------
valid_payload = {
    "passenger_count": 1,
    "trip_distance": 1.5,
    "fare_amount": 10.0,
    "total_amount": 12.0,
    "payment_type": 1,
    "trip_duration_minutes": 8.0,
    "trip_speed_mph": 10.0,
    "pickup_hour": 14,
    "pickup_day_of_week": "Monday",
    "is_weekend": False
}

# -----------------------------
# 1. Successful single prediction
# -----------------------------
def test_single_prediction_success():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
    assert "prediction_id" in data

# -----------------------------
# 2. Successful batch prediction
# -----------------------------
def test_batch_prediction_success():
    batch_payload = {
        "trips": [valid_payload, valid_payload]
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert "prediction" in item
        assert "model_version" in item
        assert "prediction_id" in item

# -----------------------------
# 3. Invalid input (missing field)
# -----------------------------
def test_invalid_input_missing_field():
    bad_payload = {
        "trip_distance": 2.5,
        "fare_amount": 15.0
        # missing fields
    }

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

# -----------------------------
# 4. Health check endpoint
# -----------------------------
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_version" in data

# -----------------------------
# 5. Edge case: zero distance
# -----------------------------
def test_edge_case_zero_distance():
    edge_payload = valid_payload.copy()
    edge_payload["trip_distance"] = 0

    response = client.post("/predict", json=edge_payload)
    assert response.status_code == 422

def test_batch_limit_exceeded():
    batch_payload = {
        "trips": [valid_payload] * 101
    }

    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 400