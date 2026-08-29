import os
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app

MODEL_PATH = "models/best_model.joblib"


@pytest.fixture(scope="module")
def client():
    """
    Pytest fixture providing a FastAPI TestClient instance.
    Triggers FastAPI lifespan events (model loading).
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_model():
    """
    Pytest fixture loading the serialized best_model joblib artifact.
    """
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file missing at '{MODEL_PATH}'. Please run train.py first.")
    return joblib.load(MODEL_PATH)


@pytest.fixture
def fraud_sample_features():
    """
    Sample feature dictionary using Member 3 canonical 12-feature schema representing a high-risk transaction.
    """
    return {
        "customer_txn_count_60m": 12,
        "customer_amount_mean_prior": 150.0,
        "amount_deviation_ratio": 4.2,
        "is_new_device": 1,
        "is_new_merchant": 1,
        "location_shift": 1,
        "customer_device_degree": 4,
        "customer_merchant_degree": 5,
        "device_customer_degree": 6,
        "merchant_customer_degree": 8,
        "shared_device_customer_count": 6,
        "relationship_risk_score": 0.85
    }


@pytest.fixture
def legit_sample_features():
    """
    Sample feature dictionary using Member 3 canonical 12-feature schema representing a normal transaction.
    """
    return {
        "customer_txn_count_60m": 1,
        "customer_amount_mean_prior": 50.0,
        "amount_deviation_ratio": 0.1,
        "is_new_device": 0,
        "is_new_merchant": 0,
        "location_shift": 0,
        "customer_device_degree": 1,
        "customer_merchant_degree": 2,
        "device_customer_degree": 1,
        "merchant_customer_degree": 2,
        "shared_device_customer_count": 0,
        "relationship_risk_score": 0.05
    }


@pytest.fixture
def legacy_fraud_sample_features():
    """
    Legacy feature dictionary representing a high-risk transaction.
    """
    return {
        "velocity_1h": 12,
        "customer_amount_mean_prior": 150.0,
        "amount_deviation": 4.2,
        "new_device": 1,
        "is_new_merchant": 1,
        "location_shift": 1,
        "customer_device_degree": 4,
        "customer_merchant_degree": 5,
        "device_customer_degree": 6,
        "merchant_customer_degree": 8,
        "shared_device_count": 6,
        "relationship_risk_score": 0.85
    }


# ==============================================================================
# 1. MODEL LOADING TESTS
# ==============================================================================

def test_best_model_artifact_exists():
    """
    Verify that models/best_model.joblib file exists on disk.
    """
    assert os.path.exists(MODEL_PATH), f"Model artifact missing at '{MODEL_PATH}'."


def test_best_model_loading(sample_model):
    """
    Verify that models/best_model.joblib loads successfully without error.
    """
    assert sample_model is not None


def test_model_has_predict_proba(sample_model):
    """
    Verify that the loaded model object exposes a callable 'predict_proba' method.
    """
    assert hasattr(sample_model, "predict_proba"), "Loaded model missing 'predict_proba' method."
    assert callable(getattr(sample_model, "predict_proba")), "'predict_proba' is not callable."


# ==============================================================================
# 2. PREDICTION SHAPE AND BOUNDS TESTS
# ==============================================================================

def test_predict_proba_bounds_fraud_sample(sample_model, fraud_sample_features):
    """
    Verify predict_proba output shape and probability range [0.0, 1.0] for a fraud sample.
    """
    df = pd.DataFrame([fraud_sample_features])
    probs = sample_model.predict_proba(df)

    assert probs.shape == (1, 2), f"Expected prediction shape (1, 2), got {probs.shape}."
    fraud_prob = probs[0, 1]
    assert 0.0 <= fraud_prob <= 1.0, f"Fraud probability {fraud_prob} out of bounds [0.0, 1.0]."


def test_predict_proba_bounds_legit_sample(sample_model, legit_sample_features):
    """
    Verify predict_proba output shape and probability range [0.0, 1.0] for a legit sample.
    """
    df = pd.DataFrame([legit_sample_features])
    probs = sample_model.predict_proba(df)

    assert probs.shape == (1, 2), f"Expected prediction shape (1, 2), got {probs.shape}."
    fraud_prob = probs[0, 1]
    assert 0.0 <= fraud_prob <= 1.0, f"Fraud probability {fraud_prob} out of bounds [0.0, 1.0]."
    assert fraud_prob < 0.5, f"Legitimate sample yielded unexpectedly high fraud probability ({fraud_prob})."


# ==============================================================================
# 3. API CONTRACT TESTS (FastAPI TestClient)
# ==============================================================================

def test_api_health(client):
    """
    Verify GET /api/v1/health returns HTTP 200 and status 'ok'.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_model_info(client):
    """
    Verify GET /api/v1/model/info returns HTTP 200 and keys 'model', 'version', 'source'.
    """
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "version" in data
    assert "source" in data
    assert isinstance(data["model"], str)
    assert isinstance(data["version"], str)
    assert data["source"] in {"registry", "local_fallback"}


def test_api_predict_member3_direct_payload(client, fraud_sample_features):
    """
    Verify POST /api/v1/model/predict with Member 3 direct feature payload.
    """
    payload = {"features": fraud_sample_features}
    response = client.post("/api/v1/model/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    expected_keys = {"fraud_probability", "risk_score", "prediction", "model_version"}
    assert expected_keys.issubset(set(data.keys())), f"Missing keys in predict response: {expected_keys - set(data.keys())}"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0 <= data["risk_score"] <= 100
    assert data["prediction"] in {"HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"}


def test_api_predict_member3_envelope_payload(client, fraud_sample_features):
    """
    Verify POST /api/v1/model/predict with Member 3 envelope payload ('data' key).
    """
    envelope = {
        "status": "success",
        "request_id": "REQ_TEST_12345",
        "data": fraud_sample_features
    }
    response = client.post("/api/v1/model/predict", json=envelope)

    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert data["prediction"] in {"HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"}


def test_api_predict_via_feature_service_mock(client, fraud_sample_features, monkeypatch):
    """
    Verify POST /api/v1/model/predict_via_feature_service with mocked Member 3 HTTP response.
    """
    import json
    from io import BytesIO

    mock_m3_response = json.dumps({
        "status": "success",
        "request_id": "REQ_MOCK_999",
        "data": fraud_sample_features
    }).encode("utf-8")

    class MockHTTPResponse:
        def read(self):
            return mock_m3_response
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_urlopen(req, timeout=5):
        return MockHTTPResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    payload = {
        "transaction": {"transaction_id": "TX_1001", "amount": 250.0},
        "feature_service_url": "http://localhost:8003/api/v1/features/extract"
    }

    response = client.post("/api/v1/model/predict_via_feature_service", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert data["prediction"] in {"HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"}


def test_api_predict_legacy_payload(client, legacy_fraud_sample_features):
    """
    Verify POST /api/v1/model/predict with legacy feature names.
    """
    payload = {"features": legacy_fraud_sample_features}
    response = client.post("/api/v1/model/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data


def test_api_predict_missing_fields(client):
    """
    Verify POST /api/v1/model/predict with missing required fields returns HTTP 422.
    """
    incomplete_payload = {
        "features": {
            "customer_txn_count_60m": 8
        }
    }
    response = client.post("/api/v1/model/predict", json=incomplete_payload)
    assert response.status_code == 422, f"Expected status code 422, got {response.status_code}."


def test_api_explain_success(client):
    """
    Verify GET /api/v1/model/explain/{id} with valid query params returns 200 and expected keys.
    """
    tx_id = "TX_TEST_999"
    url = f"/api/v1/model/explain/{tx_id}?customer_txn_count_60m=8&amount_deviation_ratio=3.0&is_new_device=1&shared_device_customer_count=4"
    response = client.get(url)

    assert response.status_code == 200
    data = response.json()

    expected_keys = {"transaction_id", "fraud_probability", "top_factors"}
    assert expected_keys.issubset(set(data.keys())), f"Missing keys in explain response: {expected_keys - set(data.keys())}"
    assert data["transaction_id"] == tx_id
    assert isinstance(data["top_factors"], list)
    assert len(data["top_factors"]) in {4, 12}




# ==============================================================================
# 4. BUSINESS LOGIC TESTS
# ==============================================================================

def test_risk_score_calculation(client, fraud_sample_features, legit_sample_features):
    """
    Verify that risk_score equals round(fraud_probability * 100).
    """
    for features in [fraud_sample_features, legit_sample_features]:
        response = client.post("/api/v1/model/predict", json={"features": features})
        assert response.status_code == 200
        data = response.json()

        prob = data["fraud_probability"]
        expected_score = int(round(prob * 100))
        assert data["risk_score"] == expected_score, f"Expected risk score {expected_score}, got {data['risk_score']}."


def test_prediction_risk_categories(client, fraud_sample_features, legit_sample_features):
    """
    Verify prediction category threshold assignment:
    - HIGH_RISK when fraud_probability >= 0.7
    - MEDIUM_RISK when fraud_probability >= 0.4 and < 0.7
    - LOW_RISK when fraud_probability < 0.4
    """
    for features in [fraud_sample_features, legit_sample_features]:
        response = client.post("/api/v1/model/predict", json={"features": features})
        assert response.status_code == 200
        data = response.json()

        prob = data["fraud_probability"]
        pred = data["prediction"]

        if prob >= 0.7:
            assert pred == "HIGH_RISK", f"Expected HIGH_RISK for prob {prob}, got {pred}."
        elif prob >= 0.4:
            assert pred == "MEDIUM_RISK", f"Expected MEDIUM_RISK for prob {prob}, got {pred}."
        else:
            assert pred == "LOW_RISK", f"Expected LOW_RISK for prob {prob}, got {pred}."

