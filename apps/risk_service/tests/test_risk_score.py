"""
Integration tests for POST /api/v1/risk/score.

Key constraint: This endpoint MUST NOT return a `decision` field.
"""
from __future__ import annotations

import httpx
import respx
from apps.risk_service.tests.conftest import (
    M3_EXTRACT_URL,
    M4_PREDICT_URL,
    M3_RESPONSE_HIGH,
    M3_RESPONSE_LOW,
    M4_RESPONSE_HIGH,
    M4_RESPONSE_LOW,
    SAMPLE_TRANSACTION,
    make_envelope,
)


def test_score_full_flow_high_risk(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        
        body = response.json()
        assert body["status"] == "success"
        data = body["data"]
        
        assert data["risk_score"] == 87
        assert data["ml_prediction"] == "HIGH_RISK"
        assert len(data["features_used"]) == 12
        assert "rule_signals" in data
        assert "latency_ms" in data


def test_score_no_decision_field_in_response(test_client):
    """The /risk/score endpoint must not return a final decision."""
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert "decision" not in data


def test_score_full_flow_low_risk(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_LOW)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_LOW)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["risk_score"] == 10


def test_score_invalid_channel_422(test_client):
    bad_tx = dict(SAMPLE_TRANSACTION, channel="PIGEON")
    response = test_client.post("/api/v1/risk/score", json=make_envelope(bad_tx))
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_score_missing_transaction_id_422(test_client):
    bad_tx = dict(SAMPLE_TRANSACTION)
    del bad_tx["transaction_id"]
    response = test_client.post("/api/v1/risk/score", json=make_envelope(bad_tx))
    assert response.status_code == 422


def test_score_m3_unavailable_503(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(500)
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "FEATURE_SERVICE_ERROR"


def test_score_m4_unavailable_503(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        # Ensure exhaustion of retries
        mock.post(M4_PREDICT_URL).respond(500)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_SERVICE_UNAVAILABLE"


def test_score_request_id_echoed(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION, "REQ_123456"))
        assert response.status_code == 200
        assert response.json()["request_id"] == "REQ_123456"


def test_score_includes_rule_signals(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        data = response.json()["data"]
        # HIGH_RISK features trigger VELOCITY_BURST, HIGH_AMOUNT_NEW_DEVICE, NEW_MERCHANT_DEVIATION, MULTI_DEVICE, LOCATION_SHIFT
        triggered = [s for s in data["rule_signals"] if s["triggered"]]
        assert len(triggered) >= 1


def test_score_includes_latency_breakdown(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        response = test_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        data = response.json()["data"]
        lat = data["latency_ms"]
        assert "feature_service_ms" in lat
        assert "model_service_ms" in lat
        assert "rule_engine_ms" in lat
        assert "total_ms" in lat
