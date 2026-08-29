"""
Shared fixtures for risk_service tests.

Integration test strategy:
  - TestClient triggers the lifespan. Since M4 is not running in test env,
    model_metadata starts as degraded. Fixtures override state as needed.
  - Use respx to mock all httpx calls within each test.
  - respx is used with assert_all_mocked=False in fixtures so unmatched
    routes from the startup attempt (which fails gracefully) don't break setup.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
import respx
from fastapi.testclient import TestClient

from apps.risk_service.main import app
from apps.risk_service.schemas.risk import ModelMetadata
from apps.risk_service.config import Settings, get_settings

# ── Service base URLs (must match Settings defaults) ──────────────────────────
M3_BASE = "http://feature_service:8000"
M4_BASE = "http://model_service:8000"
M3_EXTRACT_URL = f"{M3_BASE}/api/v1/features/extract"
M4_PREDICT_URL = f"{M4_BASE}/api/v1/model/predict"

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_TRANSACTION = {
    "transaction_id": "txn_test_001",
    "customer_id": "cust_42",
    "merchant_id": "merch_7",
    "amount": 4999.99,
    "currency": "INR",
    "timestamp": "2026-08-29T17:59:55Z",
    "channel": "CARD",
    "device_id": "dev_88",
    "location": {"country": "IN", "city": "Mumbai"},
    "merchant_category": "electronics",
}

FEATURES_HIGH_RISK = {
    "customer_txn_count_60m": 12,
    "customer_amount_mean_prior": 150.0,
    "amount_deviation_ratio": 4.2,
    "is_new_device": True,
    "is_new_merchant": True,
    "location_shift": True,
    "customer_device_degree": 6,
    "customer_merchant_degree": 5,
    "device_customer_degree": 6,
    "merchant_customer_degree": 8,
    "shared_device_customer_count": 6,
    "relationship_risk_score": 0.85,
}

FEATURES_LOW_RISK = {
    "customer_txn_count_60m": 2,
    "customer_amount_mean_prior": 100.0,
    "amount_deviation_ratio": 0.1,
    "is_new_device": False,
    "is_new_merchant": False,
    "location_shift": False,
    "customer_device_degree": 1,
    "customer_merchant_degree": 2,
    "device_customer_degree": 1,
    "merchant_customer_degree": 3,
    "shared_device_customer_count": 0,
    "relationship_risk_score": 0.05,
}

M3_RESPONSE_HIGH = {
    "request_id": "REQ_test",
    "timestamp": "2026-08-29T18:00:00Z",
    "schema_version": "1.0",
    "status": "success",
    "data": {
        "transaction_id": "txn_test_001",
        "feature_schema_version": "1.0",
        "features": FEATURES_HIGH_RISK,
    },
}

M3_RESPONSE_LOW = {
    **M3_RESPONSE_HIGH,
    "data": {
        "transaction_id": "txn_test_001",
        "feature_schema_version": "1.0",
        "features": FEATURES_LOW_RISK,
    },
}

M4_RESPONSE_HIGH = {
    "fraud_probability": 0.87,
    "risk_score": 87,
    "prediction": "HIGH_RISK",
    "model_version": "xgb-1.0",
}

M4_RESPONSE_LOW = {
    "fraud_probability": 0.10,
    "risk_score": 10,
    "prediction": "LOW_RISK",
    "model_version": "xgb-1.0",
}

M4_INFO_RESPONSE = {
    "model": "XGBClassifier",
    "version": "xgb-1.0",
    "source": "local_fallback",
}


def make_envelope(data: dict, request_id: str = "REQ_test_001") -> dict:
    return {
        "request_id": request_id,
        "timestamp": "2026-08-29T18:00:00Z",
        "schema_version": "1.0",
        "data": data,
    }


# ── Settings for client unit tests ────────────────────────────────────────────

def fast_settings(**overrides) -> Settings:
    """Settings with zero retries and fast timeouts — suitable for unit tests."""
    defaults = dict(
        feature_service_url=M3_BASE,
        model_service_url=M4_BASE,
        feature_service_timeout_seconds=1.0,
        model_service_timeout_seconds=1.0,
        model_service_max_retries=0,
        model_service_retry_backoff_seconds=0.0,
        threshold_allow_max=0.30,
        threshold_step_up_max=0.50,
        threshold_review_max=0.70,
        rule_velocity_threshold=10,
        rule_high_amount_threshold=2000.0,
        rule_deviation_threshold=3.0,
        rule_multi_device_threshold=5,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ── Integration test fixtures ─────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="function")
async def test_client():
    # Run lifespan manually or use AsyncClient with ASGITransport
    from httpx import AsyncClient, ASGITransport
    app.state.model_metadata = ModelMetadata(
        model_version="xgb-1.0",
        model_name="XGBClassifier",
        source="test_fixture",
        metadata_status="ok",
    )
    # Ensure settings exist as lifespan is bypassed
    app.state.settings = get_settings()
    app.state.feature_schema_version = "1.0"
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def degraded_client():
    from httpx import AsyncClient, ASGITransport
    app.state.model_metadata = ModelMetadata(
        model_version="unknown",
        model_name="unknown",
        source="unknown",
        metadata_status="degraded",
    )
    app.state.settings = get_settings()
    app.state.feature_schema_version = "1.0"
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client
