import pytest
from httpx import AsyncClient, ASGITransport
from apps.risk_service.main import app
from apps.risk_service.config import get_settings
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_startup_redis_down_does_not_crash():
    with patch("apps.risk_service.clients.redis_client.ping", return_value=False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["upstream"]["redis"] == "degraded"

@pytest.mark.asyncio
async def test_startup_pg_down_does_not_crash():
    with patch("apps.risk_service.clients.pg_client.ping", return_value=False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["upstream"]["postgres"] == "degraded"

@pytest.mark.asyncio
async def test_startup_both_down_does_not_crash():
    with patch("apps.risk_service.clients.redis_client.ping", return_value=False), \
         patch("apps.risk_service.clients.pg_client.ping", return_value=False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["upstream"]["redis"] == "degraded"
            assert data["upstream"]["postgres"] == "degraded"
            assert data["status"] == "degraded"
