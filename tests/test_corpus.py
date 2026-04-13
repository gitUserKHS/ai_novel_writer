from __future__ import annotations

import json
from pathlib import Path

from conarrative.corpus import holdout_story_ids, merge_training_manifests, write_jsonl


def test_holdout_story_ids_keeps_at_least_one_train_story() -> None:
    story_ids = ["a", "b", "c"]

    holdout = holdout_story_ids(story_ids, validation_story_ratio=0.34, seed=42)

    assert len(holdout) == 1
    assert holdout[0] in story_ids


def test_merge_training_manifests_splits_by_story(tmp_path: Path) -> None:
    manifests = []
    for story_id in ["story-a", "story-b", "story-c"]:
        story_dir = tmp_path / story_id
        story_dir.mkdir(parents=True)
        files = {}
        for pool in ["writer_sft", "writer_dpo", "distill_stepwise", "critic_hard_negative", "critic_consistency_sft", "world_model_transitions"]:
            path = story_dir / f"{pool}.jsonl"
            write_jsonl(path, [{"text": pool, "metadata": {"story_id": story_id}}])
            files[pool] = str(path)
        manifest_path = story_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"files": files}, ensure_ascii=False), encoding="utf-8")
        manifests.append(manifest_path)

    result = merge_training_manifests(manifests, tmp_path / "mixed", validation_story_ratio=0.34, seed=42)

    assert len(result["manifest"]["holdout_story_ids"]) == 1
    assert result["counts"]["writer_sft"]["all"] == 3
    assert result["counts"]["writer_sft"]["train"] == 2
    assert result["counts"]["writer_sft"]["eval"] == 1
    assert Path(result["manifest_path"]).exists()
