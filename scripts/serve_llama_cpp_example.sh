#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/absolute/path/to/model.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:-35}"
THREADS="${THREADS:-8}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found in PATH"
  exit 1
fi

exec llama-server \
  -m "$MODEL_PATH" \
  -c "$CTX_SIZE" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS" \
  --n-gpu-layers "$GPU_LAYERS" \
  --flash-attn on
