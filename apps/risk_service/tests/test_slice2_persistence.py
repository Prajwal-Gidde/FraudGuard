
import asyncio
import uuid
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

from apps.risk_service.main import app
from apps.risk_service.clients import redis_client, pg_client





@pytest.mark.asyncio
async def test_get_risk_not_found(test_client):
    with patch("apps.risk_service.clients.redis_client.get_decision") as mock_redis, \
         patch("apps.risk_service.clients.pg_client.get_decision") as mock_pg:
         mock_redis.return_value = None
         mock_pg.return_value = None
         
         response = await test_client.get("/api/v1/risk/TX_UNKNOWN")
         assert response.status_code == 404
         assert response.json()["error"]["code"] == "NOT_FOUND"

@pytest.mark.asyncio
async def test_get_risk_redis_hit(test_client):
    with patch("apps.risk_service.clients.redis_client.get_decision") as mock_redis:
         mock_redis.return_value = {"transaction_id": "TX_REDIS", "decision": "ALLOW"}
         
         response = await test_client.get("/api/v1/risk/TX_REDIS")
         assert response.status_code == 200
         data = response.json()["data"]
         assert data["transaction_id"] == "TX_REDIS"
         assert data["source"] == "redis"
         assert data["decision"] == "ALLOW"

@pytest.mark.asyncio
async def test_get_risk_postgres_hit(test_client):
    with patch("apps.risk_service.clients.redis_client.get_decision") as mock_redis, \
         patch("apps.risk_service.clients.pg_client.get_decision") as mock_pg:
         mock_redis.return_value = None
         mock_pg.return_value = {"transaction_id": "TX_PG", "decision": "BLOCK"}
         
         response = await test_client.get("/api/v1/risk/TX_PG")
         assert response.status_code == 200
         data = response.json()["data"]
         assert data["transaction_id"] == "TX_PG"
         assert data["source"] == "postgres"
         assert data["decision"] == "BLOCK"

@pytest.mark.asyncio
async def test_health_degraded_if_redis_down(test_client):
    with patch("apps.risk_service.clients.redis_client.ping", return_value=False), \
         patch("apps.risk_service.routers.health._ping", return_value="ok"):
         response = await test_client.get("/api/v1/health")
         assert response.status_code == 200
         data = response.json()["data"]
         assert data["status"] == "degraded"
         assert data["upstream"]["redis"] == "degraded"
