import asyncio
import httpx
import json
import time
import statistics
import argparse

async def send_request(client, url, message, session_id, request_id):
    print(f"[Request {request_id}] Sending to Session {session_id}: {message[:30]}...")
    start_time = time.perf_counter()
    server_metrics = {}
    client_ttft = None
    full_response = ""
    
    try:
        async with client.stream("POST", url, json={"message": message, "session_id": session_id}, timeout=None) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                if client_ttft is None:
                    client_ttft = time.perf_counter() - start_time
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data["type"] == "metrics":
                    server_metrics.update(data)
                elif data["type"] == "content":
                    full_response += data.get("text", "")
    except Exception as e:
        print(f"[Request {request_id}] Connection Error: {e}")
        return None

    actual_total_time = time.perf_counter() - start_time
    
    c_ttft_str = f"{client_ttft:.3f}s" if client_ttft is not None else "N/A"
    total_time_str = f"{server_metrics.get('total_time', 0):.3f}s"
    
    print(f"[{request_id}] TTFT (Client): {c_ttft_str} | Total: {total_time_str}")
    
    return {
        "metrics": {
            "client_ttft": client_ttft,
            "server_ttft": server_metrics.get("ttft"),
            "total_time": server_metrics.get("total_time"),
            "actual_total_time": actual_total_time
        },
        "response": full_response
    }

async def run_session(client, url, queries, session_id):
    """Executes all turns and returns history + metrics."""
    history = []
    performance = []
    for i, query in enumerate(queries):
        request_id = f"{session_id} Turn {i+1}"
        res = await send_request(client, url, query, session_id, request_id)
        if res:
            history.append({
                "turn": i + 1,
                "user": query,
                "assistant": res["response"]
            })
            performance.append(res["metrics"])
    return {"session_id": session_id, "history": history, "performance": performance}

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860/chat")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--json", default="convo_gemini/test-gemma-convo.json")
    parser.add_argument("--output", default="benchmark_log.json")
    args = parser.parse_args()

    # Load queries from JSON
    with open(args.json, "r") as f:
        data = json.load(f)
    
    queries = [item["text"] for item in data["conversation"] if item["speaker"] == "user"]
    
    print(f"Starting benchmark with {args.concurrency} concurrent sessions.")
    
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = []
        for i in range(args.concurrency):
            session_id = f"user_{i+1}"
            tasks.append(run_session(client, args.url, queries, session_id))
        
        all_sessions = await asyncio.gather(*tasks)

    # Stats Calculation
    stats = {
        "avg_turn1_ttft": None,
        "avg_steady_ttft": None,
        "avg_steady_gen_time": None
    }
    
    turn1_ttfts = []
    steady_ttfts = []
    steady_gen_times = []

    for session in all_sessions:
        perf = session["performance"]
        if not perf: continue
        
        # Turn 1
        if perf[0]["client_ttft"]:
            turn1_ttfts.append(perf[0]["client_ttft"])
        
        # Steady State (Turn 2+)
        for turn in perf[1:]:
            if turn["client_ttft"]: steady_ttfts.append(turn["client_ttft"])
            if turn["total_time"]: steady_gen_times.append(turn["total_time"])

    if turn1_ttfts:
        stats["avg_turn1_ttft"] = statistics.mean(turn1_ttfts)
    if steady_ttfts:
        stats["avg_steady_ttft"] = statistics.mean(steady_ttfts)
    if steady_gen_times:
        stats["avg_steady_gen_time"] = statistics.mean(steady_gen_times)

    # Export to JSON
    output_data = {
        "summary": stats,
        "sessions": all_sessions
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Full benchmark results (including stats) saved to: {args.output}")

    # Terminal Output
    print("\n--- Benchmark Results ---")
    if stats["avg_turn1_ttft"]:
        print(f"Avg Turn 1 TTFT:    {stats['avg_turn1_ttft']:.3f}s")
    if stats["avg_steady_ttft"]:
        print(f"Avg Steady TTFT:    {stats['avg_steady_ttft']:.3f}s")
    if stats["avg_steady_gen_time"]:
        print(f"Avg Steady Gen:     {stats['avg_steady_gen_time']:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
