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
        "device_id": "dev_abc123",
        "channel": "WEB"
    }
}

M4_PAYLOAD = {
    "request_id": "REQ_bench_001",
    "timestamp": "2026-08-29T18:00:00Z",
    "schema_version": "1.0",
    "data": {
        "transaction_id": "txn_12345",
        "features": {
            "amount_7d_sum": 500.0,
            "amount_30d_sum": 2000.0,
            "txn_count_7d": 5,
            "txn_count_30d": 20,
            "is_new_device": False,
            "is_new_merchant": False,
            "distance_from_home": 10.5,
            "time_since_last_txn": 3600.0,
            "velocity_60m": 1.0,
            "category_risk_score": 0.2,
            "relationship_risk_score": 0.1,
            "shared_device_customer_count": 1
        }
    }
}


async def measure_m3(client):
    start = time.monotonic()
    resp = await client.post(M3_URL, json=SAMPLE_TRANSACTION)
    return (time.monotonic() - start) * 1000, resp.status_code

async def measure_m4(client):
    start = time.monotonic()
    resp = await client.post(M4_URL, json=M4_PAYLOAD)
    return (time.monotonic() - start) * 1000, resp.status_code

async def measure_m5(client):
    start = time.monotonic()
    resp = await client.post(M5_URL, json=SAMPLE_TRANSACTION)
    data = resp.json().get("data", {}) if resp.status_code == 200 else {}
    latencies = data.get("latency_ms", {})
    return (time.monotonic() - start) * 1000, resp.status_code, latencies

async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        print('Warming up M3...')
        for _ in range(3): await client.post(M3_URL, json=SAMPLE_TRANSACTION)
        print('Warming up M4...')
        for _ in range(3): await client.post(M4_URL, json=M4_PAYLOAD)
        print('Warming up M5...')
        for _ in range(3): await client.post(M5_URL, json=SAMPLE_TRANSACTION)
        
        print('\nBenchmarking M3 (/extract)...')
        m3_times = []
        for _ in range(10):
            ms, status = await measure_m3(client)
            m3_times.append(ms)
        print(f'M3 Avg: {statistics.mean(m3_times):.2f} ms')
        
        print('\nBenchmarking M4 (/predict)...')
        m4_times = []
        for _ in range(10):
            ms, status = await measure_m4(client)
            m4_times.append(ms)
        print(f'M4 Avg: {statistics.mean(m4_times):.2f} ms')
        
        print('\nBenchmarking M5 (/decision)...')
        m5_times = []
        m3_internal = []
        m4_internal = []
        rules = []
        policy = []
        
        for _ in range(10):
            ms, status, latencies = await measure_m5(client)
            m5_times.append(ms)
            if latencies:
                m3_internal.append(latencies.get('feature_service_ms', 0))
                m4_internal.append(latencies.get('model_service_ms', 0))
                rules.append(latencies.get('rule_engine_ms', 0))
                policy.append(latencies.get('policy_engine_ms', 0))
                
        print(f'M5 Total Avg (HTTP): {statistics.mean(m5_times):.2f} ms')
        print(f'  - M3 internal (as reported by M5): {statistics.mean(m3_internal):.2f} ms')
        print(f'  - M4 internal (as reported by M5): {statistics.mean(m4_internal):.2f} ms')
        print(f'  - Rules internal: {statistics.mean(rules):.2f} ms')
        print(f'  - Policy internal: {statistics.mean(policy):.2f} ms')

if __name__ == '__main__':
    asyncio.run(main())
