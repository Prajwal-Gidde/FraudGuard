"""
Integration tests for POST /api/v1/decision and GET /api/v1/risk/{id}.
"""
from __future__ import annotations
import pytest

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


@pytest.mark.asyncio
async def test_decision_block_on_high_risk(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_HIGH)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_HIGH)
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["decision"] == "BLOCK"


@pytest.mark.asyncio
async def test_decision_allow_on_low_risk(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_LOW)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_LOW)
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["decision"] == "ALLOW"


@pytest.mark.asyncio
async def test_decision_step_up_on_medium_risk_new_device(test_client):
    mid_features = dict(M3_RESPONSE_LOW["data"]["features"])
    mid_features["is_new_device"] = True
    m3_resp = dict(M3_RESPONSE_LOW)
    m3_resp["data"]["features"] = mid_features
    
    m4_resp = dict(M4_RESPONSE_LOW)
    m4_resp["fraud_probability"] = 0.25  # Baseline ALLOW, but NEW_DEVICE rule should escalate to STEP_UP
    
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=m3_resp)
        mock.post(M4_PREDICT_URL).respond(200, json=m4_resp)
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["decision"] == "STEP_UP"


@pytest.mark.asyncio
async def test_decision_includes_policy_version(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_LOW)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_LOW)
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        data = response.json()["data"]
        assert data["policy_version"] == "v1"


@pytest.mark.asyncio
async def test_decision_rule_signals_in_response(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_LOW)
        mock.post(M4_PREDICT_URL).respond(200, json=M4_RESPONSE_LOW)
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        data = response.json()["data"]
        assert "rule_signals" in data
        assert isinstance(data["rule_signals"], list)


@pytest.mark.asyncio
async def test_get_risk_stub_404(test_client):
    response = await test_client.get("/api/v1/risk/txn_123")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "NOT_FOUND"
