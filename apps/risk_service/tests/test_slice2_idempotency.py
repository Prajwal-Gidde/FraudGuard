import asyncio
import uuid
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from apps.risk_service.tests.conftest import SAMPLE_TRANSACTION, make_envelope
from copy import deepcopy

from apps.risk_service.main import app







@pytest.mark.asyncio
async def test_concurrent_requests_execute_m3_once(test_client):
    # This is the critical M3 safety test!
    
    # We will mock Redis to simulate a real Redis lock behavior locally in memory.
    _store = {}
    _locks = {}
    
    async def mock_ping(): return True
    async def mock_get(txn_id): return _store.get(txn_id)
    async def mock_set(txn_id, data, ttl):
        _store[txn_id] = data
        return True
    async def mock_acquire(txn_id, token, ttl):
        if txn_id in _locks:
            return False
        _locks[txn_id] = token
        return True
    async def mock_release(txn_id, token):
        if _locks.get(txn_id) == token:
            del _locks[txn_id]
    async def mock_exists(txn_id): return txn_id in _locks
    
    with patch("apps.risk_service.clients.redis_client.ping", side_effect=mock_ping), \
         patch("apps.risk_service.clients.redis_client.get_decision", side_effect=mock_get), \
         patch("apps.risk_service.clients.redis_client.set_decision", side_effect=mock_set), \
         patch("apps.risk_service.clients.redis_client.acquire_lock", side_effect=mock_acquire), \
         patch("apps.risk_service.clients.redis_client.release_lock", side_effect=mock_release), \
         patch("apps.risk_service.clients.redis_client.lock_exists", side_effect=mock_exists), \
         patch("apps.risk_service.clients.pg_client.insert_decision", return_value=True) as mock_pg_insert, \
         patch("apps.risk_service.clients.pg_client.get_decision", return_value=None), \
         patch("apps.risk_service.routers.decision.extract", return_value=AsyncMock(features={"f1": 1}, feature_schema_version="1.0")) as mock_m3, \
         patch("apps.risk_service.routers.decision.predict", return_value=AsyncMock(fraud_probability=0.1, risk_score=10, ml_prediction="LOW_RISK", model_version="1.0")) as mock_m4:
         
         # Force M3 to sleep slightly so the second request catches the lock
         async def slow_extract(*args, **kwargs):
             await asyncio.sleep(0.5)
             return AsyncMock(features={"f1": 1}, feature_schema_version="1.0")
         mock_m3.side_effect = slow_extract
         
         req1 = make_envelope(deepcopy(SAMPLE_TRANSACTION))
         req2 = make_envelope(deepcopy(SAMPLE_TRANSACTION))
         
         resps = await asyncio.gather(
             test_client.post("/api/v1/decision", json=req1),
             test_client.post("/api/v1/decision", json=req2)
         )
         
         assert resps[0].status_code == 200
         assert resps[1].status_code == 200
         
         # Assert M3 was called exactly ONCE!
         assert mock_m3.call_count == 1
         assert mock_m4.call_count == 1
         assert mock_pg_insert.call_count == 1
         
         # One of them should have source = "redis" (the loser that polled)
         d1 = resps[0].json()["data"]
         d2 = resps[1].json()["data"]
         assert "redis" in [d1.get("source"), d2.get("source")]

@pytest.mark.asyncio
async def test_redis_down_pg_down_degraded(test_client):
    with patch("apps.risk_service.clients.redis_client.ping", return_value=False), \
         patch("apps.risk_service.clients.pg_client.insert_decision", return_value=False), \
         patch("apps.risk_service.routers.decision.extract") as mock_m3, \
         patch("apps.risk_service.routers.decision.predict") as mock_m4:
         
         mock_m3.return_value = AsyncMock(features={"f1": 1}, feature_schema_version="1.0")
         mock_m4.return_value = AsyncMock(fraud_probability=0.8, risk_score=90, ml_prediction="HIGH_RISK", model_version="1.0")
         
         req = make_envelope(SAMPLE_TRANSACTION)
         resp = await test_client.post("/api/v1/decision", json=req)
         assert resp.status_code == 200
         data = resp.json()["data"]
         
         # Ensure audit_persisted is False
         assert data["audit_persisted"] is False
         assert data["decision"] == "BLOCK"
@pytest.mark.asyncio
async def test_concurrent_different_transactions_independent():
    # different transaction IDs execute independently
    _store = {}
    _locks = {}
    async def mock_ping(): return True
    async def mock_get(txn_id): return _store.get(txn_id)
    async def mock_set(txn_id, data, ttl):
        _store[txn_id] = data
        return True
    async def mock_acquire(txn_id, token, ttl):
        if txn_id in _locks: return False
        _locks[txn_id] = token
        return True
    async def mock_release(txn_id, token):
        if _locks.get(txn_id) == token: del _locks[txn_id]
    async def mock_exists(txn_id): return txn_id in _locks
    
    with patch("apps.risk_service.clients.redis_client.ping", side_effect=mock_ping), \
         patch("apps.risk_service.clients.redis_client.get_decision", side_effect=mock_get), \
         patch("apps.risk_service.clients.redis_client.set_decision", side_effect=mock_set), \
         patch("apps.risk_service.clients.redis_client.acquire_lock", side_effect=mock_acquire), \
         patch("apps.risk_service.clients.redis_client.release_lock", side_effect=mock_release), \
         patch("apps.risk_service.clients.redis_client.lock_exists", side_effect=mock_exists), \
         patch("apps.risk_service.clients.pg_client.insert_decision", return_value=True) as mock_pg_insert, \
         patch("apps.risk_service.clients.pg_client.get_decision", return_value=None), \
         patch("apps.risk_service.routers.decision.extract") as mock_m3, \
         patch("apps.risk_service.routers.decision.predict") as mock_m4:

         async def extract1(*args, **kwargs):
             await asyncio.sleep(0.1)
             return AsyncMock(features={"f1": 1}, feature_schema_version="1.0")
         mock_m3.side_effect = extract1
         mock_m4.return_value = AsyncMock(fraud_probability=0.1, risk_score=10, ml_prediction="LOW_RISK", model_version="1.0")

         async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
             req1 = make_envelope(deepcopy(SAMPLE_TRANSACTION))
             req2 = make_envelope(deepcopy(SAMPLE_TRANSACTION))
             req1["data"]["transaction_id"] = "TX_DIFF_1"
             req2["data"]["transaction_id"] = "TX_DIFF_2"
             
             resps = await asyncio.gather(
                 client.post("/api/v1/decision", json=req1),
                 client.post("/api/v1/decision", json=req2)
             )
             
             # Both must compute independently, so M3 is called twice
             assert mock_m3.call_count == 2
             assert resps[0].status_code == 200
             assert resps[1].status_code == 200
             
@pytest.mark.asyncio
async def test_winner_crash_lock_expiry_and_pg_result_exists():
    _store = {}
    _locks = {}
    async def mock_ping(): return True
    async def mock_get(txn_id): return _store.get(txn_id)
    async def mock_set(txn_id, data, ttl):
        _store[txn_id] = data
        return True
    async def mock_acquire(txn_id, token, ttl):
        if txn_id in _locks: return False
        _locks[txn_id] = token
        return True
    async def mock_release(txn_id, token):
        if _locks.get(txn_id) == token: del _locks[txn_id]
        
    # Simulate lock expiry immediately for polling loser
    async def mock_exists(txn_id): return False
    
    with patch("apps.risk_service.clients.redis_client.ping", side_effect=mock_ping), \
         patch("apps.risk_service.clients.redis_client.get_decision", side_effect=mock_get), \
         patch("apps.risk_service.clients.redis_client.set_decision", side_effect=mock_set), \
         patch("apps.risk_service.clients.redis_client.acquire_lock", side_effect=mock_acquire), \
         patch("apps.risk_service.clients.redis_client.release_lock", side_effect=mock_release), \
         patch("apps.risk_service.clients.redis_client.lock_exists", side_effect=mock_exists), \
         patch("apps.risk_service.clients.pg_client.insert_decision", return_value=True) as mock_pg_insert, \
         patch("apps.risk_service.clients.pg_client.get_decision", return_value={"transaction_id": "TX_CRASH", "decision": "REVIEW"}) as mock_pg_get, \
         patch("apps.risk_service.routers.decision.extract") as mock_m3, \
         patch("apps.risk_service.routers.decision.predict") as mock_m4:
         
         # Lock is artificially held by some crashed winner
         _locks["TX_CRASH"] = "dead_token"
         
         async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
             req = make_envelope(SAMPLE_TRANSACTION)
             req["data"]["transaction_id"] = "TX_CRASH"
             
             resp = await client.post("/api/v1/decision", json=req)
             
             assert resp.status_code == 200
             # Because lock_exists returns False, loser checks PG, finds result, returns it
             assert resp.json()["data"]["decision"] == "REVIEW"
             assert resp.json()["data"]["source"] == "postgres"
             assert mock_m3.call_count == 0
