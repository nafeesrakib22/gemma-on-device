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
    
    print(f"[{request_id}] TTFT (Server: {s_ttft_str} | Client: {c_ttft_str}) | Total: {total_time_str} (Actual: {actual_total_time:.3f}s)")
    
    return {
        "server_ttft": server_ttft,
        "client_ttft": client_ttft,
        "total_time": total_time,
        "actual_total_time": actual_total_time
    }

async def run_session(client, url, queries, session_id):
    """Executes all turns in a conversation serially for a single session."""
    session_results = []
    for i, query in enumerate(queries):
        request_id = f"Session {session_id} Turn {i+1}"
        res = await send_request(client, url, query, session_id, request_id)
        if res:
            session_results.append(res)
        else:
            # If a turn fails, we might want to stop the session or continue.
            # Here we'll continue but the failure will be logged in send_request.
            pass
    return session_results

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

    print(f"Starting benchmark with {args.concurrency} concurrent sessions.")
    print(f"Each session will execute {len(queries)} turns from {args.json}.\n")
    
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = []
        for i in range(args.concurrency):
            session_id = f"user_{i+1}"
            tasks.append(run_session(client, args.url, queries, session_id))
        
        # results is a list of lists (one per session)
        session_results = await asyncio.gather(*tasks)

    # Flatten results from all sessions and turns
    results = [turn for session in session_results for turn in session]

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
