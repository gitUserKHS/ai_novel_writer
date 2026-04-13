#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conarrative.story_pack import story_count, write_balanced_story_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the balanced CoNarrative story pack.")
    parser.add_argument("--output-dir", default="examples/story_pack_balanced_54")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_balanced_story_pack(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "story_count": len(result["stories"]),
                "expected_story_count": story_count(),
                "manifest_path": result["manifest_path"],
                "genre_family_counts": result["manifest"]["genre_family_counts"],
                "tone_family_counts": result["manifest"]["tone_family_counts"],
                "conflict_engine_counts": result["manifest"]["conflict_engine_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
