# Exentec Survey Agent (Gemma-based)

An optimized inference architecture for a conversational survey agent powered by the Gemma model (e.g., `google_gemma-4-E2B-it-Q4_K_M.gguf`). This application features a robust backend proxy handling conversation states with persistent KV caching, a Gradio-based interface, and various benchmarking utilities.

## Features
- **Optimized Inference Backend**: Uses a `llama.cpp` server for high-performance generation.
- **Persistent Context (KV Pinning)**: The FastAPI proxy (`server.py`) pins user sessions to specific KV cache slots in `llama.cpp` to eliminate prompt reprocessing, drastically improving latency in multi-turn interactions.
- **Pre-warming**: Automatically pre-loads the system prompt into all cache slots on startup.
- **Premium UI**: Modern, responsive text-based chat interface built with Gradio (`app.py`).
- **Benchmarking & Analysis Tools**: Includes concurrency load testing, Khmer ASR Character Error Rate (CER) evaluation, and automated conversation analysis.

## Prerequisites
- **Python 3.12+**
- **Docker & Docker Compose** (highly recommended for backend)
- **Gemma Model File**: You need a `.gguf` file (e.g., `google_gemma-4-E2B-it-Q4_K_M.gguf`) placed in the `models/` directory for the `llama.cpp` server.

## Configuration

Before running the application, set up your environment variables by copying the example file:

```bash
cp .env.example .env
```

Edit the `.env` file to match your desired configuration.

## Running with Docker (Recommended)

The Docker configuration sets up the `llama.cpp` inference engine and the FastAPI proxy server.

1. **Prepare the Model**: Place your model file in the `models/` directory. By default, `docker-compose.yml` expects `models/google_gemma-4-E2B-it-Q4_K_M.gguf`.
   
2. **Run with Docker Compose**:
   ```bash
   docker compose up -d
   ```
   *Note: This starts the inference engine on port 8080 (internal) and the proxy API on port 7860.*

3. **Start the UI (Locally)**:
   You need to install `gradio` to run the frontend interface, as it is decoupled from the proxy API.
   ```bash
   pip install -r requirements.txt gradio
   python app.py
   ```
   Open `http://localhost:7861` in your browser to access the chat.

## Manual Installation (Local)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt gradio
   ```

2. **Start the Inference Server**:
   You must have a `llama.cpp` server running the model (typically on `http://localhost:8080`).

3. **Start the Proxy Server**:
   ```bash
   python server.py
   ```

4. **Start the Application UI**:
   ```bash
   python app.py
   ```

## Project Structure
- `server.py`: FastAPI proxy that interfaces with `llama.cpp`. Handles session caching, pre-warming, and Gemma 4 E2B specific prompt formatting (e.g., `<turn|>` handling).
- `app.py`: Main Gradio chat application frontend connecting to `server.py`.
- `docker-compose.yml`: Simplified deployment for the `llama.cpp` inference backend and the `server.py` proxy.
- `Dockerfile`: Container definition for the FastAPI proxy API.
- `prompts.py`: Defines the system instructions and behaviors for the survey agent (loaded by `server.py`).
- `benchmark.py`: Tool for executing concurrent requests to test Time to First Token (TTFT) and total generation time under load.
- `benchmark_khmer_asr_cer.py`: Evaluates Gemma's Khmer ASR transcription accuracy by calculating the Character Error Rate (CER) against ground truth, utilizing Gemini as a judge.
- `analyze_to_csv.py`: Analyzes conversation transcriptions and produces structured JSON/CSV reports.
