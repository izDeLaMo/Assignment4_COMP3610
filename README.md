# 📌 ML Taxi Tip Prediction – MLOps Project

## 👥 Project Overview
This project builds an end-to-end machine learning system to predict taxi tipping behavior using NYC Yellow Taxi trip data (January 2024). It includes data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and deployment using FastAPI and Docker.

The goal is to demonstrate a complete **MLOps pipeline** from raw data to production API.

---

## 🚀 Features

### 📊 Data Pipeline
- NYC Yellow Taxi dataset (Jan 2024)
- Data cleaning:
  - Removed null values
  - Removed invalid trips (negative/zero distance, invalid fares)
  - Removed inconsistent timestamps
- Feature engineering:
  - Trip duration (minutes)
  - Trip speed (mph)
  - Log trip distance
  - Fare per mile
  - Fare per minute
  - Pickup hour
  - Pickup day of week
  - Weekend indicator

---

### 🎯 Target Variable
- **Regression:** `tip_amount`
- **Classification:** `high_tip` (1 if tip > 20% of fare, else 0)

---

## 🤖 Machine Learning Models

### Regression Models
- Linear Regression (baseline)
- Random Forest Regressor

### Classification Models
- Logistic Regression (baseline)
- Random Forest Classifier
- ExtraTrees Classifier (tuned model)
- Neural Network (PyTorch, 2 hidden layers)

---

## ⚙️ Hyperparameter Tuning

- Model: ExtraTreesClassifier
- Method: RandomizedSearchCV
- Cross-validation: 5-fold StratifiedKFold
- Sample size: 200,000 rows
- Tuned parameters:
  - n_estimators
  - max_depth
  - min_samples_split
  - min_samples_leaf
  - max_features

---

## 📈 Evaluation Metrics

### Regression:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Classification:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

---

## 🌐 API (FastAPI Deployment)

### Base URL

http://127.0.0.1:8000


---

### Endpoints

#### Health Check

GET /health


Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1"
}
Single Prediction
POST /predict

Request Body:

{
  "passenger_count": 1,
  "trip_distance": 1,
  "fare_amount": 10,
  "total_amount": 12,
  "payment_type": 1,
  "trip_duration_minutes": 5,
  "trip_speed_mph": 12,
  "pickup_hour": 23,
  "pickup_day_of_week": "Monday",
  "is_weekend": true
}

Response:

{
  "prediction": 0.24,
  "model_version": "v1",
  "prediction_id": "uuid"
}
Batch Prediction
POST /predict/batch

Request Body:

[
  {
    "passenger_count": 1,
    "trip_distance": 1,
    "fare_amount": 10,
    "total_amount": 12,
    "payment_type": 1,
    "trip_duration_minutes": 5,
    "trip_speed_mph": 12,
    "pickup_hour": 23,
    "pickup_day_of_week": "Monday",
    "is_weekend": true
  },
  {
    "passenger_count": 2,
    "trip_distance": 5,
    "fare_amount": 30,
    "total_amount": 35,
    "payment_type": 1,
    "trip_duration_minutes": 15,
    "trip_speed_mph": 20,
    "pickup_hour": 18,
    "pickup_day_of_week": "Friday",
    "is_weekend": false
  }
]
🐳 Docker Setup
Build and Run
docker compose build --no-cache
docker compose up
Access API Docs
http://127.0.0.1:8000/docs
🧪 Testing

Run unit tests using pytest:

pytest -v

Expected output:

5 passed
📦 Project Structure
mlops/
│
├── app.py                  # FastAPI application
├── model.pkl              # Trained ML model
├── preprocessor.pkl       # Feature preprocessing pipeline
├── test_app.py            # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── notebook.ipynb        # Data analysis & training
│
└── data/
    └── raw/              # Dataset files