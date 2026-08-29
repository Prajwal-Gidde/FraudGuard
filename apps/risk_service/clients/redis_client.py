import json
import logging
from typing import Optional, Dict, Any

from redis import asyncio as aioredis

from apps.risk_service.config import Settings

logger = logging.getLogger(__name__)

_redis_pool: Optional[aioredis.Redis] = None

# Lua script to release lock only if the token matches
RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""
_release_script = None

def init_redis(settings: Settings) -> None:
    global _redis_pool, _release_script
    _redis_pool = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0
    )
    _release_script = _redis_pool.register_script(RELEASE_LOCK_SCRIPT)

async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None

async def ping() -> bool:
    if not _redis_pool:
        return False
    try:
        return await _redis_pool.ping()
    except Exception:
        return False

async def get_decision(transaction_id: str) -> Optional[Dict[str, Any]]:
    if not _redis_pool:
        return None
    try:
        val = await _redis_pool.get(f"risk:decision:{transaction_id}")
        if val:
            return json.loads(val)
    except Exception as e:
        logger.warning(f"Failed to get decision from Redis: {e}")
    return None

async def set_decision(transaction_id: str, decision_data: Dict[str, Any], ttl: int) -> bool:
    if not _redis_pool:
        return False
    try:
        await _redis_pool.setex(
            f"risk:decision:{transaction_id}",
            ttl,
            json.dumps(decision_data)
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to set decision to Redis: {e}")
        return False

async def acquire_lock(transaction_id: str, token: str, timeout_ms: int) -> bool:
    if not _redis_pool:
        return False
    try:
        return await _redis_pool.set(
            f"risk:lock:{transaction_id}",
            token,
            nx=True,
            px=timeout_ms
        )
    except Exception as e:
        logger.warning(f"Failed to acquire lock from Redis: {e}")
        return False

async def release_lock(transaction_id: str, token: str) -> None:
    if not _redis_pool or not _release_script:
        return
    try:
        await _release_script(keys=[f"risk:lock:{transaction_id}"], args=[token])
    except Exception as e:
        logger.warning(f"Failed to release lock in Redis: {e}")

async def lock_exists(transaction_id: str) -> bool:
    if not _redis_pool:
        return False
    try:
        return await _redis_pool.exists(f"risk:lock:{transaction_id}") > 0
    except Exception as e:
        logger.warning(f"Failed to check lock in Redis: {e}")
        return False
