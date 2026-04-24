import os
import json
import time
import asyncio
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
    "http://localhost:8080/v1/chat/completions",
)
PROMPT_TYPE = os.environ.get("PROMPT_TYPE", "survey")
SYSTEM_INSTRUCTION = PROMPTS.get(PROMPT_TYPE, PROMPTS["survey"])

# Number of KV cache slots configured in llama.cpp (-np flag).
# Must match the -np value in docker-compose.yml command.
NUM_SLOTS = int(os.environ.get("NUM_SLOTS", "4"))

# Derive the native /completion endpoint from INFERENCE_URL.
# Works whether INFERENCE_URL ends in /v1/chat/completions or the base URL.
_base_url = INFERENCE_URL.split("/v1/")[0].split("/completion")[0]
COMPLETION_URL = _base_url + "/completion"
HEALTH_URL = _base_url + "/health"

print(f"[INFO] Using Prompt Type:   {PROMPT_TYPE}")
print(f"[INFO] Inference Backend:   {INFERENCE_URL}")
print(f"[INFO] Native Completion:   {COMPLETION_URL}")
print(f"[INFO] KV Slots Available:  {NUM_SLOTS}")

# ─── Session State ────────────────────────────────────────────────────────────
# session_id -> list of message dicts {role, content}
session_store: dict[str, list] = {}

# session_id -> assigned slot_id (0 … NUM_SLOTS-1)
# Once assigned, a session always uses the same slot so the KV cache is reused.
slot_assignments: dict[str, int] = {}
_slot_counter = 0  # simple round-robin counter


def get_or_assign_slot(session_id: str) -> int:
    """Return the pinned slot for this session, creating one if needed."""
    global _slot_counter
    if session_id not in slot_assignments:
        slot_assignments[session_id] = _slot_counter % NUM_SLOTS
        _slot_counter += 1
        print(f"[INFO] Session '{session_id}' pinned to slot {slot_assignments[session_id]}")
    return slot_assignments[session_id]


# ─── Prompt Formatting ────────────────────────────────────────────────────────

# Regex for stripping any hallucinated turn-delimiter artifacts from model output.
# Gemma 4 uses <turn|> as EOG; the model sometimes emits partial/corrupted variants.
# Storing these in history creates a feedback loop, so we strip them out.
import re
_TURN_TOKEN_RE = re.compile(r'<[|]?turn[|]?>|<\\turn[|]?>')

def strip_end_tokens(text: str) -> str:
    """Remove <turn|> and any hallucinated turn-delimiter variants from model output."""
    return _TURN_TOKEN_RE.sub('', text).strip()


def format_gemma_prompt(messages: list) -> str:
    """
    Convert a list of {role, content} messages into Gemma 4's native chat format.

    Gemma 4 E2B template (from model metadata):
        <|turn>system
        {system_prompt}<turn|>
        <|turn>user
        {user_msg}<turn|>
        <|turn>model
        {assistant_msg}<turn|>
        ...
        <|turn>model\n      ← generation prompt (no trailing <turn|>)

    NOTE: We do NOT prepend <bos> here. llama.cpp automatically adds the BOS
    token as specified by the model's metadata (add_bos_token = true).
    Adding it manually would result in a double-BOS which corrupts generation.
    """
    prompt = ""
    system_content: str | None = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_content = content
            continue

        if role == "user":
            if system_content is not None:
                # Emit system turn first, then user turn
                prompt += f"<|turn>system\n{system_content}<turn|>\n"
                system_content = None
            prompt += f"<|turn>user\n{content}<turn|>\n"

        elif role == "assistant":
            prompt += f"<|turn>model\n{content}<turn|>\n"

    # Final model-turn prefix to trigger generation (no closing <turn|>)
    prompt += "<|turn>model\n"
    return prompt


# ─── Lifespan / Pre-warming ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=None)

    async def warm_up():
        """Prime every KV slot with the system prompt so Turn-1 prefill is minimal."""
        print(f"[INFO] Starting pre-warming of {NUM_SLOTS} KV slots…")
        retries = 15
        ready = False
        while retries > 0 and not ready:
            try:
                resp = await app.state.client.get(HEALTH_URL)
                if resp.status_code == 200:
                    ready = True
                else:
                    await asyncio.sleep(3)
            except Exception:
                await asyncio.sleep(3)
            retries -= 1

        if ready:
            warm_prompt = format_gemma_prompt([
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user",   "content": "হ্যালো"},
            ])
            for i in range(NUM_SLOTS):
                payload = {
                    "prompt":       warm_prompt,
                    "slot_id":      i,
                    "cache_prompt": True,
                    "n_predict":    1,
                    "stream":       False,
                }
                try:
                    await app.state.client.post(COMPLETION_URL, json=payload)
                    print(f"[INFO] Slot {i}/{NUM_SLOTS - 1} primed.", flush=True)
                except Exception as e:
                    print(f"[ERROR] Prime failed for slot {i}: {e}", flush=True)
            print("[INFO] PRE-WARMING COMPLETE. You can now run the benchmark.", flush=True)
        else:
            print("[WARNING] Pre-warming skipped: inference backend not reachable.", flush=True)

    asyncio.create_task(warm_up())
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


# ─── Request Model ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


# ─── Chat Endpoint ────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    """
    Proxies requests to the llama.cpp native /completion endpoint.
    Uses slot_id pinning + cache_prompt so the KV cache is reused across turns.
    """
    start_time = time.perf_counter()
    session_id = chat_req.session_id

    # Initialise session history
    if session_id not in session_store:
        session_store[session_id] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]

    # Append current user message
    session_store[session_id].append({
        "role": "user",
        "content": chat_req.message
    })

    # Build the full prompt and pin to the session's dedicated slot
    prompt = format_gemma_prompt(session_store[session_id])
    slot_id = get_or_assign_slot(session_id)

    async def generate_response():
        first_token_time = None
        full_response = ""

        payload = {
            # ── Native llama.cpp parameters ──────────────────────────────
            "prompt":        prompt,
            "slot_id":       slot_id,       # pin to this session's KV slot
            "cache_prompt":  True,          # reuse cached prefix tokens
            # NOTE: do NOT set add_bos_token here. llama.cpp adds exactly one
            # BOS per model config. Our prompt text has no <bos>, so we get
            # exactly one BOS — the correct state for Gemma.
            # ── Sampling ────────────────────────────────────────────────
            "n_predict":     512,
            "temperature":   1.0,
            "top_p":         0.95,
            "top_k":         64,
            "repeat_penalty": 1.2,
            # Gemma 4 E2B uses <turn|> as the end-of-generation token (token 106).
            # <|turn> as a stop string prevents the model from roleplaying extra turns.
            "stop": [
                "<turn|>",    # primary EOG token for Gemma 4
                "<|turn>",    # prevents roleplaying of next turn
                "<eos>",
                "</s>",
            ],
            # ── Streaming ───────────────────────────────────────────────
            "stream": True,
        }

        try:
            async with app.state.client.stream("POST", COMPLETION_URL, json=payload) as response:
                async for line in response.aiter_lines():
                    # Native /completion streams as: "data: {json}"
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    text = chunk.get("content", "")
                    stopped = chunk.get("stop", False)

                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            yield json.dumps({
                                "type": "metrics",
                                "ttft": first_token_time - start_time
                            }) + "\n"

                        full_response += text
                        yield json.dumps({"type": "content", "text": text}) + "\n"

                    if stopped:
                        break

        except Exception as e:
            print(f"[ERROR] Proxy failed for session {session_id} (slot {slot_id}): {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        # Strip any end-of-turn token artifacts before storing.
        # Storing corrupted variants (e.g. <end_of_off_turn>) would feed them
        # back into the prompt and cause a generation feedback loop.
        clean_response = strip_end_tokens(full_response)
        session_store[session_id].append({
            "role": "assistant",
            "content": clean_response
        })

        yield json.dumps({
            "type": "metrics",
            "total_time": time.perf_counter() - start_time
        }) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
