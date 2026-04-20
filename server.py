import os
import json
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from prompts import PROMPTS

# Load .env file
load_dotenv()

# Configuration
INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",
    "http://localhost:8000/v1/chat/completions",
)
PROMPT_TYPE = os.environ.get("PROMPT_TYPE", "survey")
SYSTEM_INSTRUCTION = PROMPTS.get(PROMPT_TYPE, PROMPTS["survey"])

print(f"[INFO] Using Prompt Type: {PROMPT_TYPE}")
print(f"[INFO] Inference Backend: {INFERENCE_URL}")

# Store history as a list of message dicts: {session_id: [messages]}
session_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prepare shared client
    app.state.client = httpx.AsyncClient(timeout=None)
    
    # Pre-warming: Initialize sessions to prime the KV cache
    # We do this in a separate task so as not to block the main server startup
    async def warm_up():
        print("[INFO] Starting sequential pre-warming of 5 KV slots...")
        # Wait for inference backend to be ready
        retries = 15
        ready = False
        while retries > 0 and not ready:
            try:
                resp = await app.state.client.get(INFERENCE_URL.replace("/v1/chat/completions", "/health"))
                if resp.status_code == 200:
                    ready = True
                else:
                    await asyncio.sleep(3)
            except:
                await asyncio.sleep(3)
            retries -= 1
        
        if ready:
            for i in range(1, 6):
                # We only need to trigger the backend once. 
                # Since -sp is enabled, even a tiny request will pre-fill the global system prompt cache.
                payload = {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "max_tokens": 1
                }
                try:
                    await app.state.client.post(INFERENCE_URL, json=payload)
                    print(f"[INFO] Slot {i}/5 primed.", flush=True)
                except Exception as e:
                    print(f"[ERROR] Prime failed for slot {i}: {e}", flush=True)
            print("[INFO] PRE-WARMING COMPLETE. You can now run the benchmark.", flush=True)
        else:
            print("[WARNING] Pre-warming skipped: Inference backend not reachable.", flush=True)

    import asyncio
    asyncio.create_task(warm_up())
    
    yield
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    """
    Proxies requests to the optimized llama-cpp-python backend.
    """
    start_time = time.perf_counter()
    session_id = chat_req.session_id
    
    # Initialize session history if new
    if session_id not in session_store:
        session_store[session_id] = []
    
    # Add current user message
    session_store[session_id].append({
        "role": "user", 
        "content": chat_req.message
    })

    async def generate_response():
        first_token_time = None
        full_response = ""
        
        # Construct payload without the system prompt (it is handled by -sp on the backend)
        # This ensures the prefix cache is hit every time
        payload = {
            "model": "gemma",
            "messages": session_store[session_id],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1024
        }

        try:
            async with app.state.client.stream("POST", INFERENCE_URL, json=payload) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta:
                            text = delta['content']
                            
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                yield json.dumps({
                                    "type": "metrics", 
                                    "ttft": first_token_time - start_time
                                }) + "\n"
                            
                            full_response += text
                            yield json.dumps({"type": "content", "text": text}) + "\n"
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"[ERROR] Proxy failed for session {session_id}: {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        # Store the completion
        session_store[session_id].append({
            "role": "assistant",
            "content": full_response
        })
        
        yield json.dumps({
            "type": "metrics", 
            "total_time": time.perf_counter() - start_time
        }) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
