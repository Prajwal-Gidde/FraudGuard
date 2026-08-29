"""
FraudGuard 360 — Risk & Decision Service (Member 5)

Startup behaviour:
  - Attempts GET /api/v1/model/info from M4 to cache model metadata.
  - If M4 is unavailable, starts in degraded state (model_version="unknown").
  - Degraded state is exposed by GET /api/v1/health.
  - Model metadata is updated automatically when a later successful M4
    prediction returns a model_version (live recovery path in routers).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from apps.risk_service.clients.model_client import ModelServiceError, get_model_info
from apps.risk_service.config import get_settings
from apps.risk_service.routers import decision, feedback, health, risk
from apps.risk_service.schemas.risk import ModelMetadata
from shared.schemas.envelope import EnvelopeResponse, ErrorDetail, StatusEnum


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str = "unknown",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=EnvelopeResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.ERROR,
            error=ErrorDetail(code=code, message=message),
        ).model_dump(mode="json"),
    )


from apps.risk_service.clients import redis_client, pg_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.feature_schema_version = "1.0"
    app.state.model_metadata = ModelMetadata()  # default: degraded / unknown

    # Initialize Slice 2 dependencies
    redis_client.init_redis(settings)
    await pg_client.init_pg(settings)

    # Attempt to load model metadata at startup.
    # If M4 is unavailable, start in degraded state — do not block startup.
    try:
        info = await get_model_info(settings)
        app.state.model_metadata = ModelMetadata(
            model_version=info.model_version if info.model_version not in ["unknown", ""] else "unknown",
            model_name=info.model_name if info.model_name not in ["unknown", ""] else "unknown",
            source=info.source,
            metadata_status="ok",
        )
        print(
            f"[risk_service] Model metadata loaded: "
            f"version={info.model_version} source={info.source}"
        )
    except Exception as exc:
        print(
            f"[risk_service] WARNING: M4 unavailable at startup ({type(exc).__name__}: {exc}). "
            "Starting in degraded state — model_version=unknown."
        )

    print(
        f"[risk_service] Ready | policy={settings.policy_version} "
        f"| model={app.state.model_metadata.model_version} "
        f"| metadata_status={app.state.model_metadata.metadata_status}"
    )
    yield
    # Shutdown
    await redis_client.close_redis()
    await pg_client.close_pg()


app = FastAPI(
    title="FraudGuard 360 - Risk & Decision Service",
    description=(
        "Real-time risk scoring and deterministic policy engine. "
        "Member 5 module — orchestrates M3 (features) → M4 (ML) → policy → decision."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Global exception handlers ──────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(422, "VALIDATION_ERROR", str(exc))


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    return _error_response(422, "VALIDATION_ERROR", str(exc))


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(risk.router)
app.include_router(decision.router)
app.include_router(health.router)
app.include_router(feedback.router)


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "apps.risk_service.main:app",
        host=s.risk_service_host,
        port=s.risk_service_port,
        reload=True,
    )
