#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/absolute/path/to/Qwen3-4B-Q4_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

exec env \
  CTX_SIZE="${CTX_SIZE:-8192}" \
  GPU_LAYERS="${GPU_LAYERS:-35}" \
  THREADS="${THREADS:-8}" \
  HOST="$HOST" PORT="$PORT" MODEL_PATH="$MODEL_PATH" \
  bash "$(dirname "$0")/serve_llama_cpp_example.sh"
