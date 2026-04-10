# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps required by LiteRT-LM native libraries
# curl is added for the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Security ──────────────────────────────────────────────────────────────────
# Create a non-root user for security
RUN groupadd -r gemma && useradd -r -g gemma gemma

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Ensure the app directory is owned by the non-root user
RUN chown -R gemma:gemma /app
USER gemma

# Gradio listens on 7860 by default
EXPOSE 7860

# The model file is large — mount it from the host at runtime.
# Default path matches the host path used in app.py; override with -e MODEL_PATH=...
ENV MODEL_PATH=/models/gemma-e2b/model.litertlm

# Healthcheck to verify Gradio is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
