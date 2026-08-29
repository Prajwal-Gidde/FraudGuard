"""
Integration tests for GET /api/v1/health and degraded state recovery.
"""
from __future__ import annotations

import httpx
import respx
from apps.risk_service.tests.conftest import (
    M3_BASE,
    M4_BASE,
    M3_EXTRACT_URL,
    M4_PREDICT_URL,
    M3_RESPONSE_HIGH,
    M4_RESPONSE_HIGH,
    SAMPLE_TRANSACTION,
    make_envelope,
)

M3_HEALTH = f"{M3_BASE}/health"
M4_HEALTH = f"{M4_BASE}/api/v1/health"


def test_health_all_ok(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.get(M3_HEALTH).respond(200)
        mock.get(M4_HEALTH).respond(200)
        
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()["data"]
        
        assert data["status"] == "ok"
        assert data["metadata_status"] == "ok"
        assert data["upstream"]["feature_service"] == "ok"
        assert data["upstream"]["model_service"] == "ok"


def test_health_m3_down(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.get(M3_HEALTH).respond(500)
        mock.get(M4_HEALTH).respond(200)
        
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()["data"]
        
        assert data["status"] == "degraded"
        assert data["upstream"]["feature_service"] == "degraded"


def test_health_m4_down(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.get(M3_HEALTH).respond(200)
        mock.get(M4_HEALTH).respond(500)
        
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()["data"]
        
        assert data["status"] == "degraded"
        assert data["upstream"]["model_service"] == "degraded"


def test_health_includes_versions(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.get(M3_HEALTH).respond(200)
        mock.get(M4_HEALTH).respond(200)
        
        response = test_client.get("/api/v1/health")
        data = response.json()["data"]
        
        assert "policy_version" in data
        assert "model_version" in data
        assert "feature_schema_version" in data


def test_health_degraded_when_model_unknown(degraded_client):
    """⭐ Tests degraded metadata status when M4 was down at startup."""
    with respx.mock(assert_all_mocked=False) as mock:
        mock.get(M3_HEALTH).respond(200)
        mock.get(M4_HEALTH).respond(200)
        
        response = degraded_client.get("/api/v1/health")
        data = response.json()["data"]
        
        assert data["metadata_status"] == "degraded"
        assert data["model_version"] == "unknown"
        assert data["status"] == "degraded"  # Overall status is degraded if metadata is


def test_health_recovered_when_metadata_restored(degraded_client):
    """⭐ Tests live recovery of model version via /risk/score endpoint."""
    with respx.mock(assert_all_mocked=False) as mock:
        # First, ensure we start degraded
        mock.get(M3_HEALTH).respond(200)
        mock.get(M4_HEALTH).respond(200)
        
        res_initial = degraded_client.get("/api/v1/health")
        assert res_initial.json()["data"]["metadata_status"] == "degraded"
        
        # Now make a successful predict call
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        score_res = degraded_client.post("/api/v1/risk/score", json=make_envelope(SAMPLE_TRANSACTION))
        assert score_res.status_code == 200
        
        # Verify health reflects the recovered state
        res_after = degraded_client.get("/api/v1/health")
        data = res_after.json()["data"]
        
        assert data["metadata_status"] == "recovered"
        assert data["model_version"] == "xgb-1.0"
        assert data["status"] == "ok"  # Overall status is back to ok
