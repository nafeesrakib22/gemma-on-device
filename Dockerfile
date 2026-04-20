# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps required by llama-cpp and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Security ──────────────────────────────────────────────────────────────────
# Create a non-root user for security
RUN groupadd -r gemma && useradd -r -g gemma gemma

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py prompts.py ./

# Ensure the app directory is owned by the non-root user
RUN chown -R gemma:gemma /app
USER gemma

# Gradio listens on 7860 by default
EXPOSE 7860

# Set environment variables for llama-cpp build
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
ENV MODEL_PATH="/app/models/google_gemma-4-E2B-it-Q4_K_M.gguf"

# Healthcheck to verify Gradio is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "server.py"]
