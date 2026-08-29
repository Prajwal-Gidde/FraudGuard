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
        from copy import deepcopy
        m3_resp = deepcopy(M3_RESPONSE_LOW)
        m4_resp = deepcopy(M4_RESPONSE_LOW)
        mock.post(M3_EXTRACT_URL).respond(200, json=m3_resp)
        mock.post(M4_PREDICT_URL).respond(200, json=m4_resp)
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["decision"] == "ALLOW"


@pytest.mark.asyncio
async def test_decision_step_up_on_medium_risk_new_device(test_client):
    from copy import deepcopy
    m3_resp = deepcopy(M3_RESPONSE_LOW)
    m3_resp["data"]["features"]["is_new_device"] = True
    
    m4_resp = deepcopy(M4_RESPONSE_LOW)
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


@pytest.mark.asyncio
async def test_decision_fallback_no_rules(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        from copy import deepcopy
        m3_resp = deepcopy(M3_RESPONSE_LOW)
        mock.post(M3_EXTRACT_URL).respond(200, json=m3_resp) # Clean rules
        mock.post(M4_PREDICT_URL).respond(503, json={"error": "model down"})
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["fallback_used"] is True
        assert "MODEL_UNAVAILABLE" in data["reason_codes"]
        assert data["decision"] == "ALLOW"
        assert data["fraud_probability"] is None
        assert data["risk_score"] is None
        assert response.json()["request_id"] == "REQ_test_001"

@pytest.mark.asyncio
async def test_decision_fallback_medium_rule(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        # High amount + new device -> MEDIUM rule
        from copy import deepcopy
        tx = deepcopy(SAMPLE_TRANSACTION)
        tx["amount"] = 9999.0
        
        m3_med = deepcopy(M3_RESPONSE_LOW)
        m3_med["data"]["features"]["is_new_device"] = True
        mock.post(M3_EXTRACT_URL).respond(200, json=m3_med)
        mock.post(M4_PREDICT_URL).respond(503, json={"error": "model down"})
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(tx))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["fallback_used"] is True
        assert "MODEL_UNAVAILABLE" in data["reason_codes"]
        assert data["decision"] == "STEP_UP"

@pytest.mark.asyncio
async def test_decision_fallback_high_rule(test_client):
    with respx.mock(assert_all_mocked=False) as mock:
        from copy import deepcopy
        m3_high = deepcopy(M3_RESPONSE_HIGH) # Triggers velocity burst (HIGH)
        mock.post(M3_EXTRACT_URL).respond(200, json=m3_high)
        mock.post(M4_PREDICT_URL).respond(500, json={"error": "fatal"})
        
        response = await test_client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["fallback_used"] is True
        assert data["decision"] == "BLOCK"
