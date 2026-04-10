# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps required by LiteRT-LM native libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY tools/ tools/

# Gradio listens on 7860 by default
EXPOSE 7860

# The model file is large — mount it from the host at runtime (see README).
# Default path matches the host path used in app.py; override with -e MODEL_PATH=...
ENV MODEL_PATH=/models/gemma-e2b/model.litertlm

CMD ["python", "app.py"]
