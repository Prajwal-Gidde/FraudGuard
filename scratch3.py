import asyncio
from httpx import AsyncClient, ASGITransport
import respx
from apps.risk_service.main import app
from apps.risk_service.tests.conftest import M3_EXTRACT_URL, M4_PREDICT_URL, M3_RESPONSE_LOW, SAMPLE_TRANSACTION, make_envelope

async def main():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        with respx.mock(assert_all_mocked=False) as mock:
            mock.post(M3_EXTRACT_URL).respond(200, json=M3_RESPONSE_LOW)
            mock.post(M4_PREDICT_URL).respond(503, json={"error": "model down"})
            response = await client.post("/api/v1/decision", json=make_envelope(SAMPLE_TRANSACTION))
            print("FULL RESPONSE DATA:")
            import json
            print(json.dumps(response.json(), indent=2))

asyncio.run(main())
