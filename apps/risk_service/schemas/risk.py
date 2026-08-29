"""
Pydantic models for risk_service public API.

Decision enum: ALLOW | STEP_UP | REVIEW | BLOCK — the only valid values.
APPROVE and CHALLENGE are not valid and must never appear.

RuleSignal: evidence returned by rule functions.
Rules return evidence only — they never return a Decision.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Decision(str, Enum):
    ALLOW = "ALLOW"        # Proceed, no friction
    STEP_UP = "STEP_UP"    # Require step-up authentication (OTP / biometric)
    REVIEW = "REVIEW"      # Route to human review queue
    BLOCK = "BLOCK"        # Reject transaction


class RuleSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ── Rule evidence ──────────────────────────────────────────────────────────────

class RuleSignal(BaseModel):
    """Evidence returned by a rule function. Never contains a Decision."""
    rule_id: str
    triggered: bool
    severity: RuleSeverity
    detail: str


# ── Latency ───────────────────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    feature_service_ms: int = 0
    model_service_ms: int = 0
    rule_engine_ms: int = 0
    policy_engine_ms: int | None = None
    total_ms: int = 0


# ── /risk/score response data ──────────────────────────────────────────────────

class RiskScoreData(BaseModel):
    """
    Payload for POST /api/v1/risk/score.
    Does NOT contain a `decision` field — that is /decision's responsibility.
    """
    transaction_id: str
    fraud_probability: float
    risk_score: int
    ml_prediction: str             # M4's label: LOW_RISK / MEDIUM_RISK / HIGH_RISK
    model_version: str
    feature_schema_version: str
    features_used: dict[str, Any]
    rule_signals: list[RuleSignal]
    latency_ms: LatencyBreakdown


# ── /decision response data ────────────────────────────────────────────────────

class DecisionData(BaseModel):
    transaction_id: str
    decision: Decision             # Final business decision
    policy_version: str
    fraud_probability: float | None = None
    risk_score: int | None = None
    model_version: str
    feature_schema_version: str
    rule_signals: list[RuleSignal]
    latency_ms: LatencyBreakdown
    fallback_used: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    # Slice 2 additions
    audit_persisted: bool = False
    source: str | None = None      # Used for GET /risk/{id}


# ── /feedback ─────────────────────────────────────────────────────────────────

class FeedbackData(BaseModel):
    transaction_id: str
    original_decision: str
    feedback_type: str
    reviewer: str | None = None
    notes: str | None = None


class FeedbackResponseData(BaseModel):
    transaction_id: str
    feedback_id: str
    acknowledged: bool


# ── /health ───────────────────────────────────────────────────────────────────

class HealthUpstream(BaseModel):
    feature_service: str
    model_service: str
    redis: str = "ok"              # Slice 2
    postgres: str = "ok"           # Slice 2


class HealthData(BaseModel):
    status: str                    # "ok" | "degraded"
    policy_version: str
    model_version: str
    feature_schema_version: str
    metadata_status: str           # "ok" | "degraded" | "recovered"
    upstream: HealthUpstream


# ── Model metadata (stored in app.state) ──────────────────────────────────────

class ModelMetadata(BaseModel):
    model_version: str = "logreg-1.0-v6"
    model_name: str = "fraudguard360-detector"
    source: str = "unknown"
    metadata_status: str = "degraded"   # "ok" | "degraded" | "recovered"
    registered_model_name: str = "fraudguard360-detector"
    registered_model_version: str = "6"
    run_id: str = "793b4dd5b8064cb29a8208264fca5811"
    artifact_sha256: str = "76966a027c034fdc210fac690ad2877fb004c12c99569b10d4b4df11ca3bb688"
