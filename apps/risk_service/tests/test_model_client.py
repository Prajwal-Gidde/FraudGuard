"""
Unit tests for model_client (M4 HTTP client).

Key constraints tested:
- bool→int coercion for is_new_device, is_new_merchant, location_shift
- Flat JSON response parsing (M4 does NOT return EnvelopeResponse)
- ⭐ M4 retries exactly MODEL_SERVICE_MAX_RETRIES times for transient 5xx
- ⭐ M4 does NOT retry on 422 (deterministic client error)
- ⭐ model_version returned from successful predict response
"""
from __future__ import annotations

import json
import pytest
import httpx
import respx

from apps.risk_service.clients.model_client import (
    ModelServiceError,
    ModelPrediction,
    get_model_info,
    predict,
)
from apps.risk_service.tests.conftest import M4_BASE, M4_RESPONSE_HIGH, M4_INFO_RESPONSE, fast_settings

M4_PREDICT_URL = f"{M4_BASE}/api/v1/model/predict"
M4_INFO_URL = f"{M4_BASE}/api/v1/model/info"

SAMPLE_FEATURES = {
    "customer_txn_count_60m": 5,
    "customer_amount_mean_prior": 200.0,
    "amount_deviation_ratio": 1.5,
    "is_new_device": True,
    "is_new_merchant": False,
    "location_shift": True,
    "customer_device_degree": 2,
    "customer_merchant_degree": 3,
    "device_customer_degree": 1,
    "merchant_customer_degree": 4,
    "shared_device_customer_count": 1,
    "relationship_risk_score": 0.3,
}


@pytest.fixture
def settings():
    return fast_settings(model_service_max_retries=0)


@pytest.fixture
def retry_settings():
    """Settings with max_retries=2 and zero backoff for fast tests."""
    return fast_settings(model_service_max_retries=2, model_service_retry_backoff_seconds=0.0)


# ── Payload shape ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sends_all_12_canonical_features(settings):
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).mock(
            return_value=httpx.Response(200, json=M4_RESPONSE_HIGH)
        )
        await predict(SAMPLE_FEATURES, settings)
        payload = json.loads(route.calls[0].request.content)
        features_sent = payload["features"]
        assert len(features_sent) == 12
        expected = {
            "customer_txn_count_60m", "customer_amount_mean_prior",
            "amount_deviation_ratio", "is_new_device", "is_new_merchant",
            "location_shift", "customer_device_degree", "customer_merchant_degree",
            "device_customer_degree", "merchant_customer_degree",
            "shared_device_customer_count", "relationship_risk_score",
        }
        assert set(features_sent.keys()) == expected


# ── bool→int coercion ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bool_true_coerced_to_1(settings):
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).mock(
            return_value=httpx.Response(200, json=M4_RESPONSE_HIGH)
        )
        await predict({**SAMPLE_FEATURES, "is_new_device": True}, settings)
        payload = json.loads(route.calls[0].request.content)
        assert payload["features"]["is_new_device"] == 1
        assert isinstance(payload["features"]["is_new_device"], int)


@pytest.mark.asyncio
async def test_bool_false_coerced_to_0(settings):
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).mock(
            return_value=httpx.Response(200, json=M4_RESPONSE_HIGH)
        )
        await predict({**SAMPLE_FEATURES, "is_new_device": False}, settings)
        payload = json.loads(route.calls[0].request.content)
        assert payload["features"]["is_new_device"] == 0


@pytest.mark.asyncio
async def test_all_three_bool_fields_coerced(settings):
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).mock(
            return_value=httpx.Response(200, json=M4_RESPONSE_HIGH)
        )
        feats = {**SAMPLE_FEATURES, "is_new_device": True, "is_new_merchant": True, "location_shift": False}
        await predict(feats, settings)
        payload = json.loads(route.calls[0].request.content)
        f = payload["features"]
        assert f["is_new_device"] == 1
        assert f["is_new_merchant"] == 1
        assert f["location_shift"] == 0


# ── Response parsing ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parses_flat_predict_response(settings):
    """M4 returns flat JSON — NOT EnvelopeResponse."""
    with respx.mock() as mock:
        mock.post(M4_PREDICT_URL).mock(
            return_value=httpx.Response(200, json=M4_RESPONSE_HIGH)
        )
        result = await predict(SAMPLE_FEATURES, settings)
        assert isinstance(result, ModelPrediction)
        assert result.fraud_probability == pytest.approx(0.87)
        assert result.risk_score == 87
        assert result.ml_prediction == "HIGH_RISK"
        assert result.model_version == "xgb-1.0"


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_raises_on_m4_500(settings):
    with respx.mock() as mock:
        mock.post(M4_PREDICT_URL).respond(500)
        with pytest.raises(ModelServiceError) as exc_info:
            await predict(SAMPLE_FEATURES, settings)
        assert exc_info.value.code in ("MODEL_SERVICE_UNAVAILABLE", "MODEL_SERVICE_CLIENT_ERROR")


@pytest.mark.asyncio
async def test_raises_on_m4_422_immediately(settings):
    """422 is a client error — raise immediately, never retry."""
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).respond(422, json={"detail": "bad features"})
        with pytest.raises(ModelServiceError) as exc_info:
            await predict(SAMPLE_FEATURES, settings)
        assert exc_info.value.code == "MODEL_SERVICE_CLIENT_ERROR"
        # Even with max_retries=0, 422 must never be retried
        assert route.call_count == 1


# ── ⭐ RETRY POLICY ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_m4_retry_respects_configured_max(retry_settings):
    """
    With max_retries=2, a persistent 503 should result in exactly 3 total
    HTTP calls (1 initial attempt + 2 retries) before raising ModelServiceError.
    """
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).mock(
            return_value=httpx.Response(503)
        )
        with pytest.raises(ModelServiceError) as exc_info:
            await predict(SAMPLE_FEATURES, retry_settings)

        assert route.call_count == 3, (
            f"Expected 3 total calls (1 + 2 retries), got {route.call_count}"
        )
        assert exc_info.value.code == "MODEL_SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_m4_422_not_retried(retry_settings):
    """422 must never be retried, even with max_retries=2."""
    with respx.mock() as mock:
        route = mock.post(M4_PREDICT_URL).respond(422)
        with pytest.raises(ModelServiceError):
            await predict(SAMPLE_FEATURES, retry_settings)
        assert route.call_count == 1, (
            f"422 must not be retried. Expected 1 call, got {route.call_count}"
        )


@pytest.mark.asyncio
async def test_m4_succeeds_on_second_attempt(retry_settings):
    """503 on first attempt, 200 on second — should succeed."""
    responses = [httpx.Response(503), httpx.Response(200, json=M4_RESPONSE_HIGH)]
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        r = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return r

    with respx.mock() as mock:
        mock.post(M4_PREDICT_URL).mock(side_effect=side_effect)
        result = await predict(SAMPLE_FEATURES, retry_settings)
        assert result.fraud_probability == pytest.approx(0.87)
        assert call_count == 2


# ── model/info endpoint ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_info_returns_version(settings):
    with respx.mock() as mock:
        mock.get(M4_INFO_URL).mock(
            return_value=httpx.Response(200, json=M4_INFO_RESPONSE)
        )
        info = await get_model_info(settings)
        assert info.model_version == "xgb-1.0"
        assert info.model_name == "XGBClassifier"
        assert info.source == "local_fallback"
