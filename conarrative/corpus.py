from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .utils import ensure_dir, stable_hash


POOL_NAMES = [
    "writer_sft",
    "writer_dpo",
    "distill_stepwise",
    "critic_hard_negative",
    "critic_consistency_sft",
    "world_model_transitions",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def row_story_id(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) or {}
    for key in ["story_id", "source_story_id"]:
        value = metadata.get(key) or row.get(key)
        if value:
            return str(value)
    return stable_hash({"row": row})


def holdout_story_ids(story_ids: list[str], validation_story_ratio: float, seed: int) -> list[str]:
    unique_story_ids = sorted({story_id for story_id in story_ids if story_id})
    if validation_story_ratio <= 0 or len(unique_story_ids) < 2:
        return []
    holdout_count = min(len(unique_story_ids) - 1, max(1, int(round(len(unique_story_ids) * validation_story_ratio))))
    ranked_story_ids = sorted(unique_story_ids, key=lambda story_id: stable_hash({"seed": seed, "story_id": story_id}))
    return ranked_story_ids[:holdout_count]


def merge_training_manifests(
    manifest_paths: list[str | Path],
    output_dir: str | Path,
    validation_story_ratio: float = 0.34,
    seed: int = 42,
) -> dict[str, Any]:
    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in manifest_paths]
    output_root = ensure_dir(output_dir)

    pool_rows: dict[str, list[dict[str, Any]]] = {pool: [] for pool in POOL_NAMES}
    source_story_ids: list[str] = []

    for manifest in manifests:
        files = manifest.get("files", {}) or {}
        for pool in POOL_NAMES:
            file_path = files.get(pool)
            if not file_path:
                continue
            rows = read_jsonl(file_path)
            pool_rows[pool].extend(rows)
            source_story_ids.extend(row_story_id(row) for row in rows)

    holdout_ids = set(holdout_story_ids(source_story_ids, validation_story_ratio, seed))
    counts: dict[str, dict[str, int]] = {}
    paths: dict[str, str] = {}

    for pool in POOL_NAMES:
        all_rows = pool_rows[pool]
        train_rows = [row for row in all_rows if row_story_id(row) not in holdout_ids]
        eval_rows = [row for row in all_rows if row_story_id(row) in holdout_ids]

        all_path = output_root / f"{pool}_all.jsonl"
        train_path = output_root / f"{pool}_train.jsonl"
        eval_path = output_root / f"{pool}_eval.jsonl"

        counts[pool] = {
            "all": write_jsonl(all_path, all_rows),
            "train": write_jsonl(train_path, train_rows),
            "eval": write_jsonl(eval_path, eval_rows),
        }
        paths[f"{pool}_all"] = str(all_path)
        paths[f"{pool}_train"] = str(train_path)
        paths[f"{pool}_eval"] = str(eval_path)

    manifest = {
        "generated_from_manifests": [str(Path(path)) for path in manifest_paths],
        "story_ids": sorted({story_id for story_id in source_story_ids if story_id}),
        "holdout_story_ids": sorted(holdout_ids),
        "validation_story_ratio": validation_story_ratio,
        "seed": seed,
        "counts": counts,
        "files": paths,
    }
    manifest_path = output_root / "mixed_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "paths": paths,
        "counts": counts,
    }
