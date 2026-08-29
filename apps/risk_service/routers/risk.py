"""
POST /api/v1/risk/score

Scoring and evidence endpoint. Calls M3 → M4 → rule engine.
Returns ML output + rule signals as evidence.

IMPORTANT: This endpoint MUST NOT include a `decision` field in its response.
Final business decisions (ALLOW/STEP_UP/REVIEW/BLOCK) are produced only by
POST /api/v1/decision.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from apps.risk_service.clients.feature_client import FeatureServiceError, extract
from apps.risk_service.clients.model_client import ModelServiceError, predict
from apps.risk_service.engine.rules import evaluate_all
from apps.risk_service.schemas.risk import LatencyBreakdown, RiskScoreData
from shared.schemas.envelope import EnvelopeRequest, EnvelopeResponse, ErrorDetail, StatusEnum
from shared.schemas.transaction import Transaction

router = APIRouter()


def _error(request_id: str, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=EnvelopeResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.ERROR,
            error=ErrorDetail(code=code, message=message),
        ).model_dump(mode="json"),
    )


@router.post("/api/v1/risk/score")
async def risk_score(payload: EnvelopeRequest, request: Request) -> JSONResponse:
    t_start = time.monotonic()
    req_id = payload.request_id or f"REQ_{uuid.uuid4().hex[:12]}"
    settings = request.app.state.settings
    model_meta = request.app.state.model_metadata

    # Validate transaction from envelope data
    try:
        transaction = Transaction(**(payload.data or {}))
    except (ValidationError, TypeError, Exception) as exc:
        return _error(req_id, 422, "VALIDATION_ERROR", str(exc))

    # ── Step 1: M3 feature extraction (NO retries) ─────────────────────────────
    t_m3 = time.monotonic()
    try:
        feat_result = await extract(
            transaction_data=transaction.model_dump(mode="json"),
            settings=settings,
            request_id=req_id,
        )
    except FeatureServiceError as exc:
        return _error(req_id, 503, exc.code, exc.message)
    t_m3_ms = int((time.monotonic() - t_m3) * 1000)

    # ── Step 2: M4 prediction (retries allowed) ────────────────────────────────
    t_m4 = time.monotonic()
    try:
        prediction = await predict(features=feat_result.features, settings=settings)
    except ModelServiceError as exc:
        return _error(req_id, 503, exc.code, exc.message)
    t_m4_ms = int((time.monotonic() - t_m4) * 1000)

    # Live recovery: update cached model_version if startup was degraded
    if model_meta.metadata_status == "degraded" and prediction.model_version not in ("", "unknown"):
        model_meta.model_version = prediction.model_version
        model_meta.metadata_status = "recovered"
        print(
            f"[risk_service] Model metadata recovered mid-flight: "
            f"version={prediction.model_version}"
        )

    # ── Step 3: Rule engine ────────────────────────────────────────────────────
    t_rules = time.monotonic()
    rule_signals = evaluate_all(
        features=feat_result.features,
        amount=float(transaction.amount),
        config=settings,
    )
    t_rules_ms = int((time.monotonic() - t_rules) * 1000)

    total_ms = int((time.monotonic() - t_start) * 1000)

    # Build response — NO `decision` field (this is /risk/score, not /decision)
    data = RiskScoreData(
        transaction_id=transaction.transaction_id,
        fraud_probability=prediction.fraud_probability,
        risk_score=prediction.risk_score,
        ml_prediction=prediction.ml_prediction,
        model_version=prediction.model_version,
        feature_schema_version=feat_result.feature_schema_version,
        features_used=feat_result.features,
        rule_signals=rule_signals,
        latency_ms=LatencyBreakdown(
            feature_service_ms=t_m3_ms,
            model_service_ms=t_m4_ms,
            rule_engine_ms=t_rules_ms,
            total_ms=total_ms,
        ),
    )

    return JSONResponse(
        status_code=200,
        content=EnvelopeResponse(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.SUCCESS,
            data=data.model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
