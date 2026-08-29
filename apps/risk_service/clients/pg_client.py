import json
import logging
import asyncpg
from typing import Optional, Dict, Any

from apps.risk_service.config import Settings

logger = logging.getLogger(__name__)

_pg_pool: Optional[asyncpg.Pool] = None

async def init_pg(settings: Settings) -> None:
    global _pg_pool
    try:
        _pg_pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
            command_timeout=5.0
        )
        await _create_tables()
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL pool: {e}")

async def _create_tables() -> None:
    if not _pg_pool:
        return
    query = """
    CREATE TABLE IF NOT EXISTS risk_decisions (
        transaction_id VARCHAR PRIMARY KEY,
        request_id VARCHAR,
        decision VARCHAR,
        risk_score INT,
        fraud_probability NUMERIC,
        model_version VARCHAR,
        registered_model_name VARCHAR,
        registered_model_version VARCHAR,
        run_id VARCHAR,
        artifact_sha256 VARCHAR,
        feature_schema_version VARCHAR,
        policy_version VARCHAR,
        fallback_used BOOLEAN,
        rule_signals JSONB,
        latency_ms JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_risk_decisions_created_at ON risk_decisions(created_at);

    CREATE TABLE IF NOT EXISTS risk_feedback (
        feedback_id VARCHAR PRIMARY KEY,
        transaction_id VARCHAR,
        original_decision VARCHAR,
        feedback_type VARCHAR,
        reviewer VARCHAR,
        notes TEXT,
        request_id VARCHAR,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_risk_feedback_transaction_id ON risk_feedback(transaction_id);
    CREATE INDEX IF NOT EXISTS idx_risk_feedback_created_at ON risk_feedback(created_at);
    """
    async with _pg_pool.acquire() as conn:
        await conn.execute(query)

async def close_pg() -> None:
    global _pg_pool
    if _pg_pool:
        await _pg_pool.close()
        _pg_pool = None

async def ping() -> bool:
    if not _pg_pool:
        return False
    try:
        async with _pg_pool.acquire() as conn:
            await conn.execute("SELECT 1")
            return True
    except Exception:
        return False

async def insert_decision(decision_data: Dict[str, Any]) -> bool:
    if not _pg_pool:
        return False
    query = """
        INSERT INTO risk_decisions (
            transaction_id, request_id, decision, risk_score, fraud_probability,
            model_version, registered_model_name, registered_model_version, run_id, artifact_sha256,
            feature_schema_version, policy_version, fallback_used, rule_signals, latency_ms
        ) VALUES (
            , , , , ,
            , , , , ,
            , , , , 
        ) ON CONFLICT (transaction_id) DO NOTHING;
    """
    try:
        async with _pg_pool.acquire() as conn:
            await conn.execute(
                query,
                decision_data.get("transaction_id"),
                decision_data.get("request_id"),
                decision_data.get("decision"),
                decision_data.get("risk_score"),
                decision_data.get("fraud_probability"),
                decision_data.get("model_version"),
                decision_data.get("registered_model_name"),
                decision_data.get("registered_model_version"),
                decision_data.get("run_id"),
                decision_data.get("artifact_sha256"),
                decision_data.get("feature_schema_version"),
                decision_data.get("policy_version"),
                decision_data.get("fallback_used"),
                json.dumps(decision_data.get("rule_signals", [])),
                json.dumps(decision_data.get("latency_ms", {}))
            )
        return True
    except Exception as e:
        logger.error(f"Failed to insert decision into PostgreSQL: {e}")
        return False

async def get_decision(transaction_id: str) -> Optional[Dict[str, Any]]:
    if not _pg_pool:
        return None
    query = "SELECT * FROM risk_decisions WHERE transaction_id = ;"
    try:
        async with _pg_pool.acquire() as conn:
            row = await conn.fetchrow(query, transaction_id)
            if row:
                d = dict(row)
                d["rule_signals"] = json.loads(d["rule_signals"]) if d["rule_signals"] else []
                d["latency_ms"] = json.loads(d["latency_ms"]) if d["latency_ms"] else {}
                d["fraud_probability"] = float(d["fraud_probability"]) if d["fraud_probability"] is not None else None
                # Make sure the format matches the exact schema
                return d
    except Exception as e:
        logger.error(f"Failed to get decision from PostgreSQL: {e}")
    return None

async def insert_feedback(feedback_data: Dict[str, Any]) -> bool:
    if not _pg_pool:
        return False
    query = """
        INSERT INTO risk_feedback (
            feedback_id, transaction_id, original_decision, feedback_type,
            reviewer, notes, request_id
        ) VALUES (
            , , , , , , 
        ) ON CONFLICT (feedback_id) DO NOTHING;
    """
    try:
        async with _pg_pool.acquire() as conn:
            await conn.execute(
                query,
                feedback_data.get("feedback_id"),
                feedback_data.get("transaction_id"),
                feedback_data.get("original_decision"),
                feedback_data.get("feedback_type"),
                feedback_data.get("reviewer"),
                feedback_data.get("notes"),
                feedback_data.get("request_id")
            )
        return True
    except Exception as e:
        logger.error(f"Failed to insert feedback into PostgreSQL: {e}")
        return False
