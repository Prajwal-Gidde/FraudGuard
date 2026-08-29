import asyncio
import httpx
from asgiref.sync import async_to_sync

async def main():
    from apps.risk_service.main import app
    from httpx import AsyncClient, ASGITransport
    import respx
    from apps.risk_service.tests.conftest import M3_EXTRACT_URL, M4_PREDICT_URL, M3_RESPONSE_LOW, SAMPLE_TRANSACTION, make_envelope

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with respx.mock(assert_all_mocked=False) as mock:
            mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_LOW)
            mock.post(M4_PREDICT_URL).respond(503, json={"error": "model down"})
            
            response = await client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
            print(response.json())

asyncio.run(main())
