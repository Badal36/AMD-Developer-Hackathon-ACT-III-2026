# HybridRouter v2 — Production Dockerfile
# -----------------------------------------------
# Targets: linux/amd64  (required by judging environment)
# IO:  reads /input/tasks.json  →  writes /output/results.json
# Pre-downloads all model weights at BUILD time so judges
# get zero-download, instant-start containers.

FROM python:3.11-slim

# Build-time platform assertion (helps catch wrong arch early)
ARG TARGETPLATFORM
RUN echo "Building for: ${TARGETPLATFORM:-linux/amd64}"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/root/.hybridrouter/config.json
# Tell sentence-transformers to use a baked-in cache dir
ENV SENTENCE_TRANSFORMERS_HOME=/app/.st_cache
ENV TRANSFORMERS_CACHE=/app/.hf_cache

# Work directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Install Python dependencies first (layer-cached) ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download sentence-transformers weights at BUILD time ──────────────
# This prevents runtime network timeouts during judge evaluation.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Bake Ollama local model pool into the image ───────────────────────────
# All weights are pre-pulled so judges get zero-download execution.
# Model pool (v2):
#   - smollm2:135m       ~270MB  | ultra-light, triage speed
#   - smollm2:360m       ~725MB  | fast factual routing
#   - qwen2.5:0.5b       ~397MB  | baseline factual/code
#   - openelm 270M       ~140MB  | best code routing (21.6% cal acc)
#   - qwen3:0.6b         ~523MB  | best overall accuracy (new gen)
#   - deepseek-r1:1.5b   ~1.1GB  | best math/reasoning
#   - llama3.2:1b        ~638MB  | reliable general-purpose fallback
RUN ollama serve > /var/log/ollama_build.log 2>&1 & \
    echo "Waiting for Ollama build daemon..." && \
    for i in {1..20}; do if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then break; fi; sleep 1; done && \
    echo "Pulling smollm2:135m..."    && ollama pull smollm2:135m    && \
    echo "Pulling smollm2:360m..."    && ollama pull smollm2:360m    && \
    echo "Pulling qwen2.5:0.5b..."    && ollama pull qwen2.5:0.5b    && \
    echo "Pulling openelm 270M..."    && ollama pull hf.co/mradermacher/OpenELM-270M-GGUF && \
    echo "Pulling qwen3:0.6b..."      && ollama pull qwen3:0.6b      && \
    echo "Pulling deepseek-r1:1.5b..." && ollama pull deepseek-r1:1.5b && \
    echo "Pulling llama3.2:1b..."     && ollama pull llama3.2:1b     && \
    echo "All models baked in." && ollama list

# ── Copy project source ───────────────────────────────────────────────────
COPY . .

# ── Bake pre-calibrated config for the model pool ────────────────────────
RUN mkdir -p /root/.hybridrouter && \
    cp /app/calibration/config_preset.json /root/.hybridrouter/config.json

# ── Create IO directories required by judging environment ─────────────────
RUN mkdir -p /input /output

# ── Make scripts executable ───────────────────────────────────────────────
RUN chmod +x /app/entrypoint.sh

# ── Entrypoint: starts Ollama, then runs the CLI ──────────────────────────
# Judge IO: if /input/tasks.json exists, runs in batch mode.
# Interactive: runs cli/main.py with any passed args.
ENTRYPOINT ["/app/entrypoint.sh"]
