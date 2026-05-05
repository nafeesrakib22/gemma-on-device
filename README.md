# 🚀 Gemma-4-E2B-it: High-Performance On-Device Chat

A blazingly fast, multi-user chat and agent application powered by the **Gemma-4-E2B-it** model. Designed for low-latency interactions, this project leverages a highly optimized architecture combining `llama.cpp` for native inference, `FastAPI` for concurrent request handling, and `Gradio` for a premium user interface. 

It is built specifically to support complex survey and customer service agents while maintaining strict response formats and sub-second latencies.

## ✨ Key Features

- ⚡ **Ultra-Low Latency**: Optimized KV-cache management and slot pinning ensure near-instantaneous responses (TTFT < 100ms on RTX 4090).
- 🧠 **Pre-Warming System**: Automatically caches system prompts at startup, drastically reducing processing time for the first user interaction.
- 👥 **High Concurrency**: The asynchronous FastAPI proxy efficiently handles multiple simultaneous user sessions without bottlenecking the single-threaded inference engine.
- 🛠️ **Multi-Agent Support**: Easily switch between predefined system prompts (Survey Agent, Loan Reminder, Support Desk) via environment variables.
- 📊 **Built-in Benchmarking**: Comprehensive tools to measure Turn-1 TTFT, steady-state latency, and generation speeds under simulated concurrent loads.

---

## 🏗️ Architecture

The system is decoupled into two primary layers to maximize throughput:

1. **Inference Backend (`llama.cpp`)**: A bare-metal, highly optimized C++ server handling the heavy lifting of LLM generation. Supports both CPU (via Docker) and native GPU (CUDA) execution.
2. **API Proxy & State Manager (`FastAPI`)**: A Python middleware layer that isolates user sessions, manages KV cache slots intelligently, and enforces strict generation bounds to prevent model hallucination.

---

## 💻 Getting Started (Local CPU/Docker)

The fastest way to test the application locally is via Docker Compose.

### Prerequisites
- Docker & Docker Compose
- The quantized model file: `google_gemma-4-E2B-it-Q4_K_M.gguf`

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nafeesrakib22/gemma-on-device.git
   cd gemma-on-device
   ```

2. **Prepare the Model**:
   Place your `.gguf` model file inside the `models/` directory.

3. **Launch the Stack**:
   ```bash
   docker compose up -d
   ```
   *This spins up both the `llama.cpp` inference server and the `FastAPI` backend.*

---

## 🚀 Getting Started (RunPod / GPU Deployment)

For production performance or benchmarking, deploy directly on a GPU instance (e.g., RunPod RTX 4090).

1. **Clone & Setup Environment**:
   ```bash
   git clone https://github.com/nafeesrakib22/gemma-on-device.git
   cd gemma-on-device
   python3 -m venv gemma-venv
   source gemma-venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Build `llama.cpp` with CUDA** (Ensure CUDA toolkit is installed):
   ```bash
   cd ~
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j$(nproc)
   ```

3. **Download the Model** (Directly to the pod):
   ```bash
   pip install huggingface_hub -q
   python3 -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="bartowski/google_gemma-4-E2B-it-GGUF", filename="google_gemma-4-E2B-it-Q4_K_M.gguf", local_dir="/workspace/models")'
   ```

4. **Start the Services**:
   We provide a convenience script to launch the GPU inference server and the FastAPI proxy sequentially.
   ```bash
   # Set the path to your model and start
   MODEL_PATH=/workspace/models/google_gemma-4-E2B-it-Q4_K_M.gguf ./start.sh
   ```

---

## 📈 Benchmarking

The repository includes a robust benchmarking suite to test the system under load. It simulates multiple concurrent users and captures detailed latency metrics.

**To run the benchmark (e.g., testing 1, 5, and 10 concurrent users):**
```bash
python run_benchmark.py --levels 1,5,10
```

*Note: For accurate cold-start (Turn 1) measurements, launch the server with `SKIP_WARMUP=1 ./start.sh`.*

---

## 📁 Project Structure

- `server.py`: The core FastAPI proxy, managing sessions and KV cache slots.
- `start.sh`: Production startup script for GPU environments.
- `docker-compose.yml`: Local CPU deployment configuration.
- `prompts.py`: Defines the personas and system instructions for various agents.
- `run_benchmark.py`: Multi-concurrency benchmarking orchestrator.
- `benchmark.py`: Core request simulation and metrics gathering logic.
- `app.py`: (Optional) Gradio-based web interface for manual testing.
