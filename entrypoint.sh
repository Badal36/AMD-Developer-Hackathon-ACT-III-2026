#!/bin/bash
set -e

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama server to respond on 11434
echo "Waiting for Ollama to wake up..."
for i in {1..25}; do
  if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "Ollama is ready."
    break
  fi
  sleep 1
done

# ── Judge IO Mode ─────────────────────────────────────────────────────────
# If /input/tasks.json exists (judging environment), run batch evaluation.
# Reads tasks from /input/tasks.json, writes results to /output/results.json
if [ -f "/input/tasks.json" ]; then
  echo "Judge mode: /input/tasks.json detected."
  echo "Running batch evaluation -> /output/results.json"
  exec python cli/batch_runner.py \
    --input /input/tasks.json \
    --output /output/results.json
fi

# ── Interactive / CLI Mode ────────────────────────────────────────────────
# Pass through any CLI arguments (e.g. --demo, --stats, --recalibrate)
exec python cli/main.py "$@"
