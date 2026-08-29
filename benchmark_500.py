import asyncio
import httpx
import time
import json
import statistics

# Configuration
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
        "channel": "CARD"
    }
}

async def make_request(client):
    start = time.monotonic()
    try:
        resp = await client.post(M5_URL, json=SAMPLE_TRANSACTION)
        elapsed = (time.monotonic() - start) * 1000
        is_error = resp.status_code != 200
        return elapsed, is_error
    except Exception:
        return 0, True

async def main():
    # Warmup
    async with httpx.AsyncClient(timeout=15.0, limits=httpx.Limits(max_connections=50)) as client:
        print('Warming up...')
        for _ in range(10):
            await client.post(M5_URL, json=SAMPLE_TRANSACTION)
        
        print('\nStarting benchmark (500 requests)...')
        start_time = time.monotonic()
        
        # Sequentially or Concurrently? The user said "warm-load benchmark".
        # Let's do batches of 10 concurrently
        num_requests = 500
        batch_size = 20
        latencies = []
        errors = 0
        
        for i in range(0, num_requests, batch_size):
            batch = [make_request(client) for _ in range(min(batch_size, num_requests - i))]
            results = await asyncio.gather(*batch)
            for lat, is_err in results:
                if is_err:
                    errors += 1
                else:
                    latencies.append(lat)
        
        end_time = time.monotonic()
        total_time = end_time - start_time
        
        if not latencies:
            print("All requests failed!")
            return
            
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        mean = statistics.mean(latencies)
        min_lat = latencies[0]
        max_lat = latencies[-1]
        req_sec = num_requests / total_time
        error_rate = (errors / num_requests) * 100
        
        print(f"Results over {num_requests} requests:")
        print(f"Total time: {total_time:.2f} s")
        print(f"Requests/sec: {req_sec:.2f}")
        print(f"Error rate: {error_rate:.2f}%")
        print(f"Min: {min_lat:.2f} ms")
        print(f"Mean: {mean:.2f} ms")
        print(f"P50: {p50:.2f} ms")
        print(f"P95: {p95:.2f} ms")
        print(f"P99: {p99:.2f} ms")
        print(f"Max: {max_lat:.2f} ms")

if __name__ == '__main__':
    asyncio.run(main())
