#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conarrative.corpus import merge_training_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple CoNarrative training manifests into a mixed train/eval corpus.")
    parser.add_argument("--manifest", action="append", required=True, help="Training manifest path. Pass multiple times.")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged corpus.")
    parser.add_argument("--validation-story-ratio", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_training_manifests(
        manifest_paths=[Path(path) for path in args.manifest],
        output_dir=args.output_dir,
        validation_story_ratio=args.validation_story_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
