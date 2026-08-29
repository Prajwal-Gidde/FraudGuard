"""
POST /api/v1/decision  — full pipeline + policy engine → final business decision.
GET  /api/v1/risk/{transaction_id}  — Slice 1 stub (always 404).
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
from apps.risk_service.engine.decision import decide, decide_fallback
from apps.risk_service.engine.rules import evaluate_all
from apps.risk_service.schemas.risk import DecisionData, LatencyBreakdown
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


@router.post("/api/v1/decision")
async def make_decision(payload: EnvelopeRequest, request: Request) -> JSONResponse:
    t_start = time.monotonic()
    req_id = payload.request_id or f"REQ_{uuid.uuid4().hex[:12]}"
    settings = request.app.state.settings
    model_meta = request.app.state.model_metadata

    # Validate transaction
    try:
        transaction = Transaction(**(payload.data or {}))
    except (ValidationError, TypeError, Exception) as exc:
        return _error(req_id, 422, "VALIDATION_ERROR", str(exc))

    # ── M3 feature extraction (NO retries) ────────────────────────────────────
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

    # ── M4 prediction (retries allowed) ──────────────────────────────────────
    t_m4 = time.monotonic()
    fallback_used = False
    reason_codes = []
    prediction = None
    
    try:
        prediction = await predict(features=feat_result.features, settings=settings)
    except ModelServiceError as exc:
        fallback_used = True
        reason_codes.append("MODEL_UNAVAILABLE")
    t_m4_ms = int((time.monotonic() - t_m4) * 1000)

    # Live recovery: update cached model_version if startup was degraded
    if prediction and model_meta.metadata_status == "degraded" and prediction.model_version not in ("", "unknown"):
        model_meta.model_version = prediction.model_version
        model_meta.metadata_status = "recovered"
        print(
            f"[risk_service] Model metadata recovered: version={prediction.model_version}"
        )

    # ── Rule engine ───────────────────────────────────────────────────────────
    t_rules = time.monotonic()
    rule_signals = evaluate_all(
        features=feat_result.features,
        amount=float(transaction.amount),
        config=settings,
    )
    t_rules_ms = int((time.monotonic() - t_rules) * 1000)

    # ── Policy engine ─────────────────────────────────────────────────────────
    t_policy = time.monotonic()
    if fallback_used:
        final_decision = decide_fallback(rule_signals=rule_signals)
    else:
        final_decision = decide(
            fraud_probability=prediction.fraud_probability,
            rule_signals=rule_signals,
            config=settings,
        )
    t_policy_ms = int((time.monotonic() - t_policy) * 1000)

    total_ms = int((time.monotonic() - t_start) * 1000)

    data = DecisionData(
        transaction_id=transaction.transaction_id,
        decision=final_decision,
        policy_version=settings.policy_version,
        fraud_probability=prediction.fraud_probability if prediction else None,
        risk_score=prediction.risk_score if prediction else None,
        model_version=prediction.model_version if prediction else "unavailable",
        feature_schema_version=feat_result.feature_schema_version,
        rule_signals=rule_signals,
        fallback_used=fallback_used,
        reason_codes=reason_codes,
        latency_ms=LatencyBreakdown(
            feature_service_ms=t_m3_ms,
            model_service_ms=t_m4_ms,
            rule_engine_ms=t_rules_ms,
            policy_engine_ms=t_policy_ms,
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


@router.get("/api/v1/risk/{transaction_id}")
async def get_risk_result(transaction_id: str) -> JSONResponse:
    """
    Slice 1: always returns 404. Persistence (Redis) added in Slice 2.
    Endpoint is stubbed to preserve the public API contract.
    """
    req_id = f"REQ_{uuid.uuid4().hex[:12]}"
    return JSONResponse(
        status_code=404,
        content=EnvelopeResponse(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.ERROR,
            error=ErrorDetail(
                code="NOT_FOUND",
                message=(
                    f"No cached result for transaction '{transaction_id}'. "
                    "Persistence not yet enabled — coming in Slice 2."
                ),
            ),
        ).model_dump(mode="json"),
    )
