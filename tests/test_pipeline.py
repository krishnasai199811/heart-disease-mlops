import os
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.preprocess import build_preprocessor


# Use TestClient with lifespan support
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_patient_dict():
    return {
        "age": 63.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 0.0,
        "ca": 0.0,
        "thal": 1.0
    }


# --- 1. Preprocessing Tests ---
def test_preprocessor_transformation(sample_patient_dict):
    """Test that ColumnTransformer transforms raw input into expected numerical array without NaN."""
    preprocessor = build_preprocessor()
    df = pd.DataFrame([sample_patient_dict])
    transformed = preprocessor.fit_transform(df)
    assert transformed is not None
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any(), "Transformed output must not contain NaN"


def test_preprocessor_handles_missing_values(sample_patient_dict):
    """Test that imputation handles missing values gracefully when fit on training-like data."""
    preprocessor = build_preprocessor()
    patient_missing = sample_patient_dict.copy()
    patient_missing["ca"] = np.nan
    patient_missing["thal"] = np.nan
    # Preprocessor requires at least one non-null per column to learn imputation values
    df = pd.DataFrame([sample_patient_dict, patient_missing])
    transformed = preprocessor.fit_transform(df)
    assert not np.isnan(transformed).any(), "Imputer failed to resolve NaNs"


# --- 2. Model Persistence Tests ---
def test_saved_model_exists_and_predicts(sample_patient_dict):
    """Test that the trained best_model.joblib exists and produces valid predictions."""
    model_path = os.path.join("models", "best_model.joblib")
    assert os.path.exists(model_path), "Serialized model artifact best_model.joblib missing"

    model = joblib.load(model_path)
    df = pd.DataFrame([sample_patient_dict])
    pred = model.predict(df)
    proba = model.predict_proba(df)

    assert pred[0] in [0, 1], "Prediction should be binary (0 or 1)"
    assert len(proba[0]) == 2, "Probabilities should have 2 classes"
    assert np.isclose(proba[0].sum(), 1.0), "Probabilities must sum to 1.0"


# --- 3. API Endpoint Tests ---
def test_api_health_endpoint(client):
    """Test GET /health returns 200 OK and model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["model_loaded"] is True


def test_api_predict_endpoint_valid(client, sample_patient_dict):
    """Test POST /predict returns valid schema with confidence score."""
    response = client.post("/predict", json=sample_patient_dict)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "prediction_label" in data


def test_api_predict_endpoint_invalid_payload(client):
    """Test POST /predict with invalid data types returns 422 Unprocessable Entity."""
    invalid_payload = {"age": "not-a-number", "sex": 1}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
