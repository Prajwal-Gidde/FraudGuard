import asyncio
import httpx
import time
import json
import statistics

# Configuration
M3_URL = 'http://localhost:8003/api/v1/features/extract'
M4_URL = 'http://localhost:8000/api/v1/model/predict'
M5_URL = 'http://localhost:8005/api/v1/decision'

SAMPLE_TRANSACTION = {
    "request_id": "REQ_bench_001",
    "timestamp": "2026-08-29T18:00:00Z",
    "schema_version": "1.0",
    "data": {
        "transaction_id": "txn_12345",
        "customer_id": "cust_999",
        "merchant_id": "merch_555",
        "amount": 150.0,
        "currency": "USD",
        "timestamp": "2026-08-29T18:00:00Z",
        "device_id": "dev_abc123"
    }
}

async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(M5_URL, json=SAMPLE_TRANSACTION)
        print(json.dumps(resp.json(), indent=2))

if __name__ == '__main__':
    asyncio.run(main())
