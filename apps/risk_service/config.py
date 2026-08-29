"""
risk_service configuration — all settings sourced from environment variables.

THRESHOLD NOTE:
  Values 0.30 / 0.50 / 0.70 are TEMPORARY DEVELOPMENT DEFAULTS ONLY.
  They are NOT empirically validated production thresholds.
  Final operating points will be set after Member 4 delivers the threshold
  sweep (precision-recall / F1 / FPR curves) and M4/M5 jointly calibrate
  the decision boundary.

RETRY NOTE:
  feature_service (M3) client has ZERO retries by design — see feature_client.py.
  model_service (M4) client retries are configurable here.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service ────────────────────────────────────────────────────────────────
    risk_service_host: str = "0.0.0.0"
    risk_service_port: int = 8005
    risk_service_log_level: str = "INFO"

    # ── Policy ─────────────────────────────────────────────────────────────────
    policy_version: str = "v1"

    # ── Upstream service URLs ──────────────────────────────────────────────────
    feature_service_url: str = "http://feature_service:8000"
    model_service_url: str = "http://model_service:8000"

    # ── M3 client — NO RETRIES BY DESIGN ──────────────────────────────────────
    # POST /api/v1/features/extract mutates M3's in-memory history store.
    # A retry on a timed-out POST could double-register the transaction and
    # corrupt subsequent feature extraction for the same customer.
    # This timeout is the only configurable parameter — retries are not exposed.
    feature_service_timeout_seconds: float = 2.0

    # ── M4 client — retries allowed (prediction is read-only) ─────────────────
    model_service_timeout_seconds: float = 2.0
    model_service_max_retries: int = 2
    model_service_retry_backoff_seconds: float = Field(0.1, env="MODEL_SERVICE_RETRY_BACKOFF")

    # Slice 2: Redis and DB config
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")
    database_url: str = Field("postgresql://localhost:5432/fraudguard", alias="DATABASE_URL")
    redis_lock_timeout_ms: int = Field(15000, env="REDIS_LOCK_TIMEOUT_MS")
    redis_cache_ttl_seconds: int = Field(86400, env="REDIS_CACHE_TTL_SECONDS")

    # Comma-separated HTTP status codes eligible for retry (transient errors only)
    # 4xx responses (e.g. 422 validation errors) are NEVER retried.
    model_service_retry_on_status: str = "500,502,503,504"

    # ── ML baseline thresholds — TEMPORARY DEVELOPMENT DEFAULTS ONLY ──────────
    # NOT empirically validated production thresholds.
    # Calibration deferred to M4 threshold sweep + M4/M5 joint operating-point
    # selection. All values must be read from config — never hardcoded in logic.
    threshold_allow_max: float = 0.30
    threshold_step_up_max: float = 0.50
    threshold_review_max: float = 0.70

    # ── Rule evidence thresholds — TEMPORARY DEVELOPMENT DEFAULTS ONLY ────────
    # Subject to the same calibration process as ML thresholds.
    rule_velocity_threshold: int = 10
    rule_high_amount_threshold: float = 2000.0
    rule_deviation_threshold: float = 3.0
    rule_multi_device_threshold: int = 5

    @property
    def retry_on_status_codes(self) -> frozenset[int]:
        return frozenset(
            int(s.strip())
            for s in self.model_service_retry_on_status.split(",")
            if s.strip().isdigit()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
