#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/absolute/path/to/gemma-3-4b-it-qat-q4_0.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

exec env \
  CTX_SIZE="${CTX_SIZE:-8192}" \
  GPU_LAYERS="${GPU_LAYERS:-35}" \
  THREADS="${THREADS:-8}" \
  HOST="$HOST" PORT="$PORT" MODEL_PATH="$MODEL_PATH" \
  bash "$(dirname "$0")/serve_llama_cpp_example.sh"
