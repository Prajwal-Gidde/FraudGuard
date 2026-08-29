import asyncio
import time
import uuid
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from apps.risk_service.clients.feature_client import FeatureServiceError, extract
from apps.risk_service.clients.model_client import ModelServiceError, predict
from apps.risk_service.clients import redis_client, pg_client
from apps.risk_service.engine.decision import decide, decide_fallback
from apps.risk_service.engine.rules import evaluate_all
from apps.risk_service.schemas.risk import DecisionData, LatencyBreakdown
from shared.schemas.envelope import EnvelopeRequest, EnvelopeResponse, ErrorDetail, StatusEnum
from shared.schemas.transaction import Transaction

logger = logging.getLogger(__name__)
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

    txn_id = transaction.transaction_id
    is_redis_up = await redis_client.ping()
    lock_token = uuid.uuid4().hex
    lock_acquired = False

    if is_redis_up:
        # Check cache
        cached = await redis_client.get_decision(txn_id)
        if cached:
            cached["source"] = "redis"
            return JSONResponse(
                status_code=200,
                content=EnvelopeResponse(
                    request_id=req_id,
                    timestamp=datetime.now(timezone.utc),
                    status=StatusEnum.SUCCESS,
                    data=cached,
                ).model_dump(mode="json")
            )
        
        lock_acquired = await redis_client.acquire_lock(txn_id, lock_token, settings.redis_lock_timeout_ms)
        
        if not lock_acquired:
            # Loser polling
            max_polls = int(settings.redis_lock_timeout_ms / 100)
            for _ in range(max_polls):
                await asyncio.sleep(0.1)
                cached = await redis_client.get_decision(txn_id)
                if cached:
                    cached["source"] = "redis"
                    return JSONResponse(
                        status_code=200,
                        content=EnvelopeResponse(
                            request_id=req_id,
                            timestamp=datetime.now(timezone.utc),
                            status=StatusEnum.SUCCESS,
                            data=cached,
                        ).model_dump(mode="json")
                    )
                
                if not await redis_client.lock_exists(txn_id):
                    # Lock disappeared, check PG
                    pg_result = await pg_client.get_decision(txn_id)
                    if pg_result:
                        pg_result["source"] = "postgres"
                        # Set to redis for future
                        await redis_client.set_decision(txn_id, pg_result, settings.redis_cache_ttl_seconds)
                        return JSONResponse(
                            status_code=200,
                            content=EnvelopeResponse(
                                request_id=req_id,
                                timestamp=datetime.now(timezone.utc),
                                status=StatusEnum.SUCCESS,
                                data=pg_result,
                            ).model_dump(mode="json")
                        )
                    
                    # Try to acquire new lock
                    lock_acquired = await redis_client.acquire_lock(txn_id, lock_token, settings.redis_lock_timeout_ms)
                    if lock_acquired:
                        break
            
            if not lock_acquired:
                return _error(req_id, 409, "CONCURRENCY_ERROR", "Transaction is already being processed by another request.")

    # Winner computes
    try:
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

        t_rules = time.monotonic()
        rule_signals = evaluate_all(
            features=feat_result.features,
            amount=float(transaction.amount),
            config=settings,
        )
        t_rules_ms = int((time.monotonic() - t_rules) * 1000)

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
            transaction_id=txn_id,
            decision=final_decision,
            policy_version=settings.policy_version,
            fraud_probability=prediction.fraud_probability if prediction else None,
            risk_score=prediction.risk_score if prediction else None,
            model_version=model_meta.model_version,
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

        # Persistence to PostgreSQL
        saved_db = await pg_client.insert_decision(
            decision_data={
                **data.model_dump(mode="json"),
                "request_id": req_id,
                "registered_model_name": model_meta.registered_model_name,
                "registered_model_version": model_meta.registered_model_version,
                "run_id": model_meta.run_id,
                "artifact_sha256": model_meta.artifact_sha256
            }
        )
        
        if not saved_db:
            logger.error(f"Failed to persist decision audit for {txn_id} to PostgreSQL")
            data.audit_persisted = False
        else:
            data.audit_persisted = True

        response_data = data.model_dump(mode="json")
        
        # Save to Redis
        if is_redis_up:
            # Inject metadata for cache
            cache_payload = dict(response_data)
            cache_payload["registered_model_name"] = model_meta.registered_model_name
            cache_payload["registered_model_version"] = model_meta.registered_model_version
            cache_payload["run_id"] = model_meta.run_id
            cache_payload["artifact_sha256"] = model_meta.artifact_sha256
            cache_payload["cached_at"] = datetime.now(timezone.utc).isoformat()
            await redis_client.set_decision(txn_id, cache_payload, settings.redis_cache_ttl_seconds)

        return JSONResponse(
            status_code=200,
            content=EnvelopeResponse(
                request_id=req_id,
                timestamp=datetime.now(timezone.utc),
                status=StatusEnum.SUCCESS,
                data=response_data,
            ).model_dump(mode="json"),
        )
    finally:
        if lock_acquired:
            await redis_client.release_lock(txn_id, lock_token)

@router.get("/api/v1/risk/{transaction_id}")
async def get_risk_result(transaction_id: str) -> JSONResponse:
    req_id = f"REQ_{uuid.uuid4().hex[:12]}"
    
    # 1. Try Redis
    cached = await redis_client.get_decision(transaction_id)
    if cached:
        cached["source"] = "redis"
        return JSONResponse(
            status_code=200,
            content=EnvelopeResponse(
                request_id=req_id,
                timestamp=datetime.now(timezone.utc),
                status=StatusEnum.SUCCESS,
                data=cached,
            ).model_dump(mode="json")
        )
        
    # 2. Try Postgres
    pg_result = await pg_client.get_decision(transaction_id)
    if pg_result:
        pg_result["source"] = "postgres"
        return JSONResponse(
            status_code=200,
            content=EnvelopeResponse(
                request_id=req_id,
                timestamp=datetime.now(timezone.utc),
                status=StatusEnum.SUCCESS,
                data=pg_result,
            ).model_dump(mode="json")
        )
        
    return JSONResponse(
        status_code=404,
        content=EnvelopeResponse(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.ERROR,
            error=ErrorDetail(
                code="NOT_FOUND",
                message=f"No cached result for transaction '{transaction_id}'."
            ),
        ).model_dump(mode="json"),
    )
