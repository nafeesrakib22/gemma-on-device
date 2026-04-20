from llama_cpp import Llama
import time
import os
import json
import anyio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from prompts import PROMPTS

# Load .env file
load_dotenv()

# Configuration
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "./models/google_gemma-4-E2B-it-Q4_K_M.gguf",
)
PROMPT_TYPE = os.environ.get("PROMPT_TYPE", "survey")
SYSTEM_INSTRUCTION = PROMPTS.get(PROMPT_TYPE, PROMPTS["survey"])

print(f"[INFO] Using Prompt Type: {PROMPT_TYPE}")
print(f"[INFO] Model Path: {MODEL_PATH}")

import asyncio

# Global state
llm = None
# llama-cpp-python has its own internal locking/queuing, 
# but for Gemma 2's specific architecture on CPU, we'll keep a simple lock 
# to ensure stability during high-concurrency batching.
engine_lock = asyncio.Lock()
# Store history as a list of message dicts: {session_id: [messages]}
session_store = {}

def _init_engine():
    """Initialize Llama engine."""
    global llm
    print(f"[INFO] Initializing Llama Engine with model: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,           # Sufficient context for multi-turn sessions
        n_threads=8,         # Match vCPUs on c6a.2xlarge
        n_batch=512,
        verbose=False
    )
    print("[INFO] Llama Engine initialized and ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize engine
    await anyio.to_thread.run_sync(_init_engine)
    yield
    # Shutdown
    print("[INFO] Shutting down Llama Engine.")

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    """
    Handles isolated sessions using llama-cpp-python's chat completion.
    """
    start_time = time.perf_counter()
    session_id = chat_req.session_id
    
    # Initialize session history if new
    if session_id not in session_store:
        session_store[session_id] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]
    
    # Add current user message to local history
    session_store[session_id].append({
        "role": "user", 
        "content": chat_req.message
    })

    async def generate_response():
        async with engine_lock:
            first_token_time = None
            full_response = ""
            
            # Use llama-cpp-python's high-level chat API
            # This is already optimized and handles prompting formats correctly
            def run_inference_sync():
                return llm.create_chat_completion(
                    messages=session_store[session_id],
                    stream=True,
                    temperature=0.7,
                    max_tokens=512
                )

            # Consuming a synchronous iterator from a thread and yielding back to FastAPI
            try:
                stream = await anyio.to_thread.run_sync(run_inference_sync)
                
                for chunk in stream:
                    # Extract delta content
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        text = delta['content']
                        
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            yield json.dumps({"type": "metrics", "ttft": first_token_time - start_time}) + "\n"
                        
                        full_response += text
                        yield json.dumps({"type": "content", "text": text}) + "\n"
                        
            except Exception as e:
                print(f"[ERROR] Inference failed for session {session_id}: {e}")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            
            # Store the assistant response in history
            session_store[session_id].append({
                "role": "assistant",
                "content": full_response
            })
            
            yield json.dumps({"type": "metrics", "total_time": time.perf_counter() - start_time}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
