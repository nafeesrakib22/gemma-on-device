import asyncio
import httpx
import json
import time
import statistics
import argparse

async def send_request(client, url, message, session_id, request_id):
    print(f"[Request {request_id}] Sending to Session {session_id}: {message[:30]}...")
    start_time = time.perf_counter()
    server_ttft = None
    client_ttft = None
    total_time = None
    full_response = ""
    
    try:
        async with client.stream("POST", url, json={"message": message, "session_id": session_id}, timeout=None) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                # Record the moment the first response chunk arrives
                if client_ttft is None:
                    client_ttft = time.perf_counter() - start_time
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Request {request_id}] WARNING: Failed to parse line: {line[:50]}...")
                    continue

                if data["type"] == "metrics":
                    if "ttft" in data:
                        server_ttft = data["ttft"]
                    if "total_time" in data:
                        total_time = data["total_time"]
                elif data["type"] == "content":
                    full_response += data["text"]
    except Exception as e:
        print(f"[Request {request_id}] Connection Error: {e}")
        return None

    actual_total_time = time.perf_counter() - start_time
    s_ttft_str = f"{server_ttft:.3f}s" if server_ttft is not None else "N/A"
    c_ttft_str = f"{client_ttft:.3f}s" if client_ttft is not None else "N/A"
    total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
    
    print(f"[Request {request_id}] TTFT (Server: {s_ttft_str} | Client: {c_ttft_str}) | Total: {total_time_str} (Actual: {actual_total_time:.3f}s)")
    
    return {
        "request_id": request_id,
        "server_ttft": server_ttft,
        "client_ttft": client_ttft,
        "total_time": total_time,
        "actual_total_time": actual_total_time,
        "response": full_response
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860/chat")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--json", default="convo_gemini/test-gemma-convo.json")
    args = parser.parse_args()

    # Load queries from JSON
    with open(args.json, "r") as f:
        data = json.load(f)
    
    queries = [item["text"] for item in data["conversation"] if item["speaker"] == "user"]
    if not queries:
        print("No user queries found in JSON.")
        return

    # Take first N queries or repeat if not enough
    test_queries = (queries * (args.concurrency // len(queries) + 1))[:args.concurrency]

    print(f"Starting benchmark with {args.concurrency} concurrent requests to {args.url}...")
    
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = []
        for i, query in enumerate(test_queries):
            # Assign a unique session ID to each concurrent request
            session_id = f"user_{i+1}"
            tasks.append(send_request(client, args.url, query, session_id, i+1))
        
        results = await asyncio.gather(*tasks)

    results = [r for r in results if r is not None]
    
    if not results:
        print("No successful results.")
        return

    server_ttfts = [r["server_ttft"] for r in results if r["server_ttft"] is not None]
    client_ttfts = [r["client_ttft"] for r in results if r["client_ttft"] is not None]
    total_times = [r["total_time"] for r in results if r["total_time"] is not None]

    print("\n--- Benchmark Results ---")
    print(f"Concurrency: {args.concurrency}")
    
    if server_ttfts:
        print(f"Avg Server TTFT: {statistics.mean(server_ttfts):.3f}s")
    if client_ttfts:
        print(f"Avg Client TTFT: {statistics.mean(client_ttfts):.3f}s (Inc. overhead)")
        print(f"Min/Max Client:  {min(client_ttfts):.3f}s / {max(client_ttfts):.3f}s")
    
    if total_times:
        print(f"Avg Generation Time: {statistics.mean(total_times):.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
