#!/bin/bash
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/models/google_gemma-4-E2B-it-Q4_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
NUM_SLOTS="${NUM_SLOTS:-4}"
PROMPT_TYPE="${PROMPT_TYPE:-survey}"

echo "[start.sh] Model:       $MODEL_PATH"
echo "[start.sh] Slots:       $NUM_SLOTS"
echo "[start.sh] Prompt type: $PROMPT_TYPE"

# ── Step 1: Start llama.cpp inference server ───────────────────────────────────
echo "[start.sh] Starting llama-server (CUDA)..."
"$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  -c 16384 \
  -np "$NUM_SLOTS" \
  -ngl 99 \
  -t 4 \
  -cb \
  --host 0.0.0.0 \
  --port 8080 \
  > /tmp/llama-server.log 2>&1 &

LLAMA_PID=$!
echo "[start.sh] llama-server PID: $LLAMA_PID"

# ── Step 2: Wait for inference server to be healthy ────────────────────────────
echo "[start.sh] Waiting for inference server to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "[start.sh] Inference server is ready!"
    break
  fi
  if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo "[ERROR] llama-server crashed! Check /tmp/llama-server.log"
    cat /tmp/llama-server.log
    exit 1
  fi
  echo "[start.sh] Waiting... ($i/30)"
  sleep 3
done

# ── Step 3: Start FastAPI proxy ────────────────────────────────────────────────
echo "[start.sh] Starting FastAPI server (server.py)..."
INFERENCE_URL="http://localhost:8080/v1/chat/completions" \
PROMPT_TYPE="$PROMPT_TYPE" \
NUM_SLOTS="$NUM_SLOTS" \
python server.py
