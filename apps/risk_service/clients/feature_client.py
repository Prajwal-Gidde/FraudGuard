"""
M3 Feature Service HTTP client.

CRITICAL — ZERO RETRIES BY DESIGN:
  POST /api/v1/features/extract mutates Member 3's in-memory history store.
  A retry on a timed-out POST could double-register the transaction and
  corrupt subsequent feature extraction for the same customer window.

  This module makes EXACTLY ONE request attempt. On timeout or connection
  failure, FeatureServiceError is raised immediately without any retry.
  No retry configuration is exposed — the absence of retries is intentional
  and must be preserved.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from apps.risk_service.config import Settings


class FeatureServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


@dataclass
class FeatureResult:
    features: dict
    feature_schema_version: str
    transaction_id: str


# Module-level connection pool to eliminate TCP/DNS setup latency per request
_http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100))

async def extract(
    transaction_data: dict,
    settings: Settings,
    request_id: str | None = None,
) -> FeatureResult:
    """
    POST to M3 /api/v1/features/extract — ONE attempt only, no retries.

    Args:
        transaction_data: dict representation of the Transaction model.
        settings: service configuration.
        request_id: request_id to propagate in the envelope.

    Raises:
        FeatureServiceError: on timeout, connection failure, or HTTP error.
    """
    req_id = request_id or f"REQ_{uuid.uuid4().hex[:12]}"
    payload = {
        "request_id": req_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "data": transaction_data,
    }
    url = f"{settings.feature_service_url}/api/v1/features/extract"

    # ONE attempt — no retry loop, no fallback
    try:
        response = await _http_client.post(
            url, 
            json=payload, 
            timeout=settings.feature_service_timeout_seconds
        )
    except httpx.TimeoutException as exc:
        raise FeatureServiceError(
            "FEATURE_SERVICE_UNAVAILABLE",
            f"feature_service did not respond within {settings.feature_service_timeout_seconds}s",
        ) from exc
    except httpx.ConnectError as exc:
        raise FeatureServiceError(
            "FEATURE_SERVICE_UNAVAILABLE",
            f"feature_service connection failed: {exc}",
        ) from exc

    if response.status_code >= 400:
        raise FeatureServiceError(
            "FEATURE_SERVICE_ERROR",
            f"feature_service returned HTTP {response.status_code}: {response.text[:200]}",
        )

    body = response.json()
    data = body.get("data") or {}
    features = data.get("features", {})
    feature_schema_version = data.get("feature_schema_version", "1.0")
    transaction_id = data.get("transaction_id", "")

    if not features:
        raise FeatureServiceError(
            "FEATURE_SERVICE_BAD_RESPONSE",
            "feature_service returned an empty features dict",
        )

    return FeatureResult(
        features=features,
        feature_schema_version=feature_schema_version,
        transaction_id=transaction_id,
    )
