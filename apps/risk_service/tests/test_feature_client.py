"""
Unit tests for feature_client (M3 HTTP client).

Key constraint tested: M3 POST is NEVER retried.
test_m3_not_retried_after_timeout asserts exactly 1 call on timeout.
"""
from __future__ import annotations

import pytest
import httpx
import respx

from apps.risk_service.clients.feature_client import FeatureServiceError, extract
from apps.risk_service.tests.conftest import M3_BASE, M3_RESPONSE_HIGH, fast_settings

M3_EXTRACT_URL = f"{M3_BASE}/api/v1/features/extract"


@pytest.fixture
def settings():
    return fast_settings()


@pytest.fixture
def tx():
    return {
        "transaction_id": "txn_001",
        "customer_id": "cust_1",
        "merchant_id": "merch_1",
        "amount": 100.0,
        "currency": "INR",
        "timestamp": "2026-08-29T17:00:00Z",
        "channel": "CARD",
        "device_id": "dev_1",
        "location": {"country": "IN", "city": "Delhi"},
        "merchant_category": "retail",
    }


# ── Payload shape ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_builds_envelope_request_correctly(settings, tx):
    with respx.mock() as mock:
        route = mock.post(M3_EXTRACT_URL).mock(
            return_value=httpx.Response(200, json=M3_RESPONSE_HIGH)
        )
        await extract(tx, settings, request_id="REQ_envelope_test")
        assert route.called
        body = route.calls[0].request.content
        import json
        payload = json.loads(body)
        assert payload["request_id"] == "REQ_envelope_test"
        assert payload["schema_version"] == "1.0"
        assert "timestamp" in payload
        assert payload["data"]["transaction_id"] == "txn_001"


@pytest.mark.asyncio
async def test_extracts_12_features_from_response(settings, tx):
    with respx.mock() as mock:
        mock.post(M3_EXTRACT_URL).mock(
            return_value=httpx.Response(200, json=M3_RESPONSE_HIGH)
        )
        result = await extract(tx, settings)
        assert len(result.features) == 12
        assert "customer_txn_count_60m" in result.features
        assert "relationship_risk_score" in result.features


@pytest.mark.asyncio
async def test_extracts_feature_schema_version(settings, tx):
    with respx.mock() as mock:
        mock.post(M3_EXTRACT_URL).mock(
            return_value=httpx.Response(200, json=M3_RESPONSE_HIGH)
        )
        result = await extract(tx, settings)
        assert result.feature_schema_version == "1.0"


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_raises_feature_service_error_on_500(settings, tx):
    with respx.mock() as mock:
        mock.post(M3_EXTRACT_URL).respond(500)
        with pytest.raises(FeatureServiceError) as exc_info:
            await extract(tx, settings)
        assert exc_info.value.code == "FEATURE_SERVICE_ERROR"


@pytest.mark.asyncio
async def test_raises_feature_service_error_on_connect(settings, tx):
    with respx.mock() as mock:
        mock.post(M3_EXTRACT_URL).mock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(FeatureServiceError) as exc_info:
            await extract(tx, settings)
        assert exc_info.value.code == "FEATURE_SERVICE_UNAVAILABLE"


# ── ⭐ NO-RETRY GUARANTEE ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_m3_not_retried_after_timeout(settings, tx):
    """
    CRITICAL: M3 POST mutates history state. A retry on timeout would
    double-register the transaction. This test asserts that exactly ONE
    HTTP call is made, even when the request times out.
    """
    with respx.mock() as mock:
        route = mock.post(M3_EXTRACT_URL).mock(
            side_effect=httpx.TimeoutException("simulated timeout")
        )
        with pytest.raises(FeatureServiceError) as exc_info:
            await extract(tx, settings)

        # The critical assertion: exactly ONE call made — no retry
        assert route.call_count == 1, (
            f"Expected exactly 1 call to M3 after timeout, got {route.call_count}. "
            "M3 POST must never be retried."
        )
        assert exc_info.value.code == "FEATURE_SERVICE_UNAVAILABLE"
        assert "timeout" in exc_info.value.message.lower() or "respond" in exc_info.value.message.lower()
