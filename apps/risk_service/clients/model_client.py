"""
M4 Model Service HTTP client.

Retry policy:
  POST /api/v1/model/predict is read-only (inference only). Retries are safe.
  Retries are performed for transient server errors:
    - HTTP 500, 502, 503, 504 (configurable via MODEL_SERVICE_RETRY_ON_STATUS)
    - Connection errors (httpx.ConnectError)
    - Timeout errors (httpx.TimeoutException)

  Deterministic 4xx errors (especially 422 validation errors) are NEVER retried —
  the request is malformed and retrying would produce the same result.

  Maximum retry count is configurable via MODEL_SERVICE_MAX_RETRIES.
  Backoff between retries is configurable via MODEL_SERVICE_RETRY_BACKOFF_SECONDS.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from apps.risk_service.config import Settings


class ModelServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


@dataclass
class ModelPrediction:
    fraud_probability: float
    risk_score: int
    ml_prediction: str      # M4's label: LOW_RISK / MEDIUM_RISK / HIGH_RISK
    model_version: str      # Dynamic — derived from loaded model type at M4 startup


@dataclass
class ModelInfo:
    model_name: str
    model_version: str
    source: str


_BOOL_FIELDS = frozenset({"is_new_device", "is_new_merchant", "location_shift"})


def _coerce_bools(features: dict) -> dict:
    """
    Coerce bool feature values to int before sending to M4.
    M3 returns bools (True/False); M4 TransactionFeatures expects int (1/0).
    M4 also handles this internally via map_aliases(), but M5 coerces
    defensively to document and enforce the M3→M4 boundary explicitly.
    """
    result = dict(features)
    for field in _BOOL_FIELDS:
        if field in result and isinstance(result[field], bool):
            result[field] = int(result[field])
    return result


# Module-level connection pool to eliminate TCP/DNS setup latency per request
_http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100))


async def predict(features: dict, settings: Settings) -> ModelPrediction:
    """
    POST to M4 /api/v1/model/predict with retry on transient failures.
    4xx responses are raised immediately without retry.
    """
    coerced = _coerce_bools(features)
    payload = {"features": coerced}
    url = f"{settings.model_service_url}/api/v1/model/predict"
    retry_statuses = settings.retry_on_status_codes
    max_attempts = settings.model_service_max_retries + 1

    for attempt in range(max_attempts):
        is_last = attempt == max_attempts - 1
        try:
            response = await _http_client.post(
                url, 
                json=payload, 
                timeout=settings.model_service_timeout_seconds
            )

            # 4xx → client error, never retry
            if 400 <= response.status_code < 500:
                raise ModelServiceError(
                    "MODEL_SERVICE_CLIENT_ERROR",
                    f"model_service returned HTTP {response.status_code}: {response.text[:300]}",
                )

            # Retryable server error
            if response.status_code in retry_statuses:
                if not is_last:
                    await asyncio.sleep(settings.model_service_retry_backoff_seconds)
                    continue
                raise ModelServiceError(
                    "MODEL_SERVICE_UNAVAILABLE",
                    f"model_service returned HTTP {response.status_code} "
                    f"after {max_attempts} attempt(s)",
                )

            if response.status_code >= 500:
                if not is_last:
                    await asyncio.sleep(settings.model_service_retry_backoff_seconds)
                    continue
                raise ModelServiceError(
                    "MODEL_SERVICE_UNAVAILABLE",
                    f"model_service returned HTTP {response.status_code}",
                )

            # Success
            body = response.json()
            return ModelPrediction(
                fraud_probability=float(body["fraud_probability"]),
                risk_score=int(body["risk_score"]),
                ml_prediction=body.get("prediction", "UNKNOWN"),
                model_version=body.get("model_version", "unknown"),
            )

        except ModelServiceError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if not is_last:
                await asyncio.sleep(settings.model_service_retry_backoff_seconds)
                continue
            raise ModelServiceError(
                "MODEL_SERVICE_UNAVAILABLE",
                f"model_service did not respond after {max_attempts} attempt(s): {exc}",
            ) from exc

    # Should not reach here, but guard
    raise ModelServiceError(
        "MODEL_SERVICE_UNAVAILABLE",
        f"model_service exhausted {max_attempts} attempt(s)",
    )


async def get_model_info(settings: Settings) -> ModelInfo:
    """
    GET /api/v1/model/info — retrieves model metadata (version, name, source).
    Used at startup to populate app.state.model_metadata.
    Retries on transient failures.
    """
    url = f"{settings.model_service_url}/api/v1/model/info"
    retry_statuses = settings.retry_on_status_codes
    max_attempts = settings.model_service_max_retries + 1

    for attempt in range(max_attempts):
        is_last = attempt == max_attempts - 1
        try:
            response = await _http_client.get(
                url, 
                timeout=settings.model_service_timeout_seconds
            )

            if 400 <= response.status_code < 500:
                raise ModelServiceError(
                    "MODEL_INFO_CLIENT_ERROR",
                    f"model_service /info returned HTTP {response.status_code}",
                )

            if response.status_code in retry_statuses:
                if not is_last:
                    await asyncio.sleep(settings.model_service_retry_backoff_seconds)
                    continue
                raise ModelServiceError(
                    "MODEL_INFO_UNAVAILABLE",
                    f"model_service /info returned HTTP {response.status_code}",
                )

            if response.status_code >= 500:
                if not is_last:
                    await asyncio.sleep(settings.model_service_retry_backoff_seconds)
                    continue
                raise ModelServiceError(
                    "MODEL_INFO_UNAVAILABLE",
                    f"model_service /info returned HTTP {response.status_code}",
                )

            body = response.json()
            return ModelInfo(
                model_name=body.get("model", "unknown"),
                model_version=body.get("version", "unknown"),
                source=body.get("source", "unknown"),
            )

        except ModelServiceError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if not is_last:
                await asyncio.sleep(settings.model_service_retry_backoff_seconds)
                continue
            raise ModelServiceError(
                "MODEL_INFO_UNAVAILABLE",
                f"model_service /info did not respond: {exc}",
            ) from exc

    raise ModelServiceError("MODEL_INFO_UNAVAILABLE", "Exhausted retries")
