from __future__ import annotations

import json
from pathlib import Path

from conarrative.story_pack import build_balanced_story_pack, story_count, write_balanced_story_pack


def test_balanced_story_pack_has_expected_size_and_balance() -> None:
    stories = build_balanced_story_pack()
    assert len(stories) == story_count() == 54

    genre_counts: dict[str, int] = {}
    tone_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    for story in stories:
        parts = story["id"].split("-")
        genre_counts[parts[0]] = genre_counts.get(parts[0], 0) + 1
        tone_key = parts[1]
        tone_counts[tone_key] = tone_counts.get(tone_key, 0) + 1
        conflict_counts[parts[-1]] = conflict_counts.get(parts[-1], 0) + 1

    assert set(genre_counts.values()) == {9}
    assert set(tone_counts.values()) == {18}
    assert set(conflict_counts.values()) == {18}


def test_write_balanced_story_pack_writes_manifest_and_yaml(tmp_path: Path) -> None:
    result = write_balanced_story_pack(tmp_path)
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["story_count"] == 54
    assert len(list(tmp_path.glob("*.yaml"))) == 54
