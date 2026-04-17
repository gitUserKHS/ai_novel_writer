#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m conarrative.cli --config configs/demo.yaml init
python -m conarrative.cli --config configs/demo.yaml create-story --input-file examples/story.yaml
python -m conarrative.cli --config configs/demo.yaml outline --story-id moon-theater --scene-count 4
python -m conarrative.cli --config configs/demo.yaml run-scene --story-id moon-theater --input-file examples/scene1.yaml --print-text
python -m conarrative.cli --config configs/demo.yaml export --story-id moon-theater
python -m conarrative.cli --config configs/demo.yaml evaluate --story-id moon-theater

python -m conarrative.cli --config configs/demo.yaml serve --host 127.0.0.1 --port 8000
