"""
POST /api/v1/feedback

Slice 1: log-only stub. Returns acknowledgement with a feedback_id.
DB write and monitoring pipeline integration deferred to Slice 2.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from apps.risk_service.schemas.risk import FeedbackData, FeedbackResponseData
from shared.schemas.envelope import EnvelopeRequest, EnvelopeResponse, ErrorDetail, StatusEnum

router = APIRouter()


@router.post("/api/v1/feedback")
async def submit_feedback(payload: EnvelopeRequest) -> JSONResponse:
    req_id = payload.request_id or f"REQ_{uuid.uuid4().hex[:12]}"

    try:
        fb = FeedbackData(**(payload.data or {}))
    except (ValidationError, TypeError, Exception) as exc:
        return JSONResponse(
            status_code=422,
            content=EnvelopeResponse(
                request_id=req_id,
                timestamp=datetime.now(timezone.utc),
                status=StatusEnum.ERROR,
                error=ErrorDetail(code="VALIDATION_ERROR", message=str(exc)),
            ).model_dump(mode="json"),
        )

    feedback_id = f"FB_{uuid.uuid4().hex[:8]}"
    # Slice 1: structured log only — DB write deferred to Slice 2
    print(
        f"[feedback] feedback_id={feedback_id} "
        f"transaction_id={fb.transaction_id} "
        f"original_decision={fb.original_decision} "
        f"feedback={fb.feedback} "
        f"reviewer={fb.reviewer}"
    )

    return JSONResponse(
        status_code=200,
        content=EnvelopeResponse(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc),
            status=StatusEnum.SUCCESS,
            data=FeedbackResponseData(
                transaction_id=fb.transaction_id,
                feedback_id=feedback_id,
                acknowledged=True,
            ).model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
