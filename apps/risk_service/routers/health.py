"""GET /api/v1/health — pings upstreams and reports model metadata status."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from apps.risk_service.schemas.risk import HealthData, HealthUpstream
from shared.schemas.envelope import EnvelopeResponse, StatusEnum

router = APIRouter()


async def _ping(url: str, timeout: float) -> str:
    """Returns 'ok' if the upstream responds with < 500, else 'degraded'."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
        return "ok" if resp.status_code < 500 else "degraded"
    except Exception:
        return "degraded"


from apps.risk_service.clients import redis_client, pg_client

@router.get("/api/v1/health")
async def health_check(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    model_meta = request.app.state.model_metadata

    m3_status = await _ping(
        f"{settings.feature_service_url}/health",
        settings.feature_service_timeout_seconds,
    )
    m4_status = await _ping(
        f"{settings.model_service_url}/api/v1/health",
        settings.model_service_timeout_seconds,
    )
    redis_status = "ok" if await redis_client.ping() else "degraded"
    pg_status = "ok" if await pg_client.ping() else "degraded"

    metadata_status = model_meta.metadata_status  # "ok" | "degraded" | "recovered"
    overall = (
        "degraded"
        if "degraded" in (metadata_status, m3_status, m4_status, redis_status, pg_status)
        else "ok"
    )

    data = HealthData(
        status=overall,
        policy_version=settings.policy_version,
        model_version=model_meta.model_version,
        feature_schema_version=request.app.state.feature_schema_version,
        metadata_status=metadata_status,
        upstream=HealthUpstream(
            feature_service=m3_status, 
            model_service=m4_status,
            redis=redis_status,
            postgres=pg_status
        ),
    )

    req_id = f"REQ_{uuid.uuid4().hex[:12]}"
    return JSONResponse(
        status_code=200,
        content=EnvelopeResponse(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.SUCCESS,
            data=data.model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
