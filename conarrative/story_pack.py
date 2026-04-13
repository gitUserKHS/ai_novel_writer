from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


GENRE_SPECS = [
    {
        "family": "mystery",
        "genre": "mystery drama",
        "settings": ["sealed theater", "municipal archive", "winter port"],
        "roles": ["archivist", "stage manager", "night reporter"],
        "objects": ["ledger", "ticket stub", "missing reel"],
        "themes": ["memory", "evidence", "choice"],
        "constraints": ["no magic", "no time travel"],
        "title_prefix": "Hidden",
    },
    {
        "family": "science_fiction",
        "genre": "science fiction suspense",
        "settings": ["climate station", "tidal city", "orbital clinic"],
        "roles": ["systems diver", "maintenance analyst", "signal cartographer"],
        "objects": ["black box", "salt map", "deferred transmission"],
        "themes": ["systems", "survival", "responsibility"],
        "constraints": ["no omnipotent AI", "no instant travel"],
        "title_prefix": "Signal",
    },
    {
        "family": "fantasy",
        "genre": "low fantasy intrigue",
        "settings": ["river shrine", "ash market", "mountain court"],
        "roles": ["temple scribe", "courier", "oath broker"],
        "objects": ["votive bell", "ink seal", "winter charm"],
        "themes": ["oath", "debt", "kinship"],
        "constraints": ["magic has a cost", "no resurrection"],
        "title_prefix": "Oath",
    },
    {
        "family": "thriller",
        "genre": "urban thriller",
        "settings": ["subway interchange", "warehouse district", "election office"],
        "roles": ["fixer", "forensic accountant", "field producer"],
        "objects": ["burner phone", "payment sheet", "elevator key"],
        "themes": ["pressure", "trust", "exposure"],
        "constraints": ["no superpowers", "no invincible protagonist"],
        "title_prefix": "Pressure",
    },
    {
        "family": "historical",
        "genre": "historical drama",
        "settings": ["printing house", "frontier clinic", "colonial customs hall"],
        "roles": ["translator", "apprentice doctor", "ledger clerk"],
        "objects": ["proof sheet", "sealed crate", "ship registry"],
        "themes": ["record", "class", "duty"],
        "constraints": ["period-accurate technology only", "no prophecy shortcuts"],
        "title_prefix": "Ledger",
    },
    {
        "family": "contemporary",
        "genre": "contemporary emotional drama",
        "settings": ["group home", "regional campus", "night bakery"],
        "roles": ["social worker", "lab tutor", "assistant baker"],
        "objects": ["voice memo", "library card", "recipe notebook"],
        "themes": ["repair", "belonging", "silence"],
        "constraints": ["no supernatural twist", "keep the stakes human-scale"],
        "title_prefix": "Quiet",
    },
]


TONE_SPECS = [
    {
        "family": "lyrical_tense",
        "tone": "lyrical and tense",
        "modifiers": ["haunted", "rain-streaked", "breath-held"],
        "themes": ["grief", "echo"],
    },
    {
        "family": "precise_cool",
        "tone": "precise and cool-headed",
        "modifiers": ["clinical", "measured", "steel-lit"],
        "themes": ["control", "distance"],
    },
    {
        "family": "warm_melancholic",
        "tone": "warm but melancholic",
        "modifiers": ["tender", "late-autumn", "careworn"],
        "themes": ["care", "forgiveness"],
    },
]


CONFLICT_SPECS = [
    {
        "engine": "search",
        "goal": "locate the missing truth before it is erased",
        "premise": "A {role} finds the first trace of a vanished {object} in the {setting}, and the discovery points back to a decision nobody wants reopened.",
        "themes": ["search", "erasure"],
    },
    {
        "engine": "coverup",
        "goal": "break a cover-up without destroying the wrong person",
        "premise": "When a routine handoff in the {setting} fails, a {role} realizes the official story around a {object} was engineered to redirect blame.",
        "themes": ["cover-up", "blame"],
    },
    {
        "engine": "rivalry",
        "goal": "outmaneuver a rival while preserving a fragile alliance",
        "premise": "A {role} enters a quiet rivalry over a contested {object} in the {setting}, only to learn both sides are being steered toward the same trap.",
        "themes": ["rivalry", "alliance"],
    },
]


NAME_POOL = [
    ["Seorin", "Minjae", "Haeun"],
    ["Yujin", "Taesu", "Jihye"],
    ["Narin", "Dowan", "Mira"],
    ["Eunsol", "Junseo", "Haemin"],
    ["Sion", "Ara", "Hyobin"],
    ["Dajung", "Iseul", "Kyungmin"],
    ["Robin", "Sua", "Joon"],
    ["Yeonwoo", "Minki", "Sera"],
    ["Hajin", "Boram", "Taemin"],
]


def story_count() -> int:
    return len(GENRE_SPECS) * len(TONE_SPECS) * len(CONFLICT_SPECS)


def build_balanced_story_pack() -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []
    for genre_index, genre in enumerate(GENRE_SPECS):
        for tone_index, tone in enumerate(TONE_SPECS):
            for conflict_index, conflict in enumerate(CONFLICT_SPECS):
                combo_index = genre_index * len(TONE_SPECS) * len(CONFLICT_SPECS) + tone_index * len(CONFLICT_SPECS) + conflict_index
                setting = genre["settings"][(tone_index + conflict_index) % len(genre["settings"])]
                role = genre["roles"][(tone_index * 2 + conflict_index) % len(genre["roles"])]
                story_object = genre["objects"][(genre_index + conflict_index) % len(genre["objects"])]
                modifier = tone["modifiers"][(genre_index + conflict_index) % len(tone["modifiers"])]
                names = NAME_POOL[combo_index % len(NAME_POOL)]
                story_id = f"{genre['family']}-{tone['family']}-{conflict['engine']}"
                title = f"{genre['title_prefix']} {modifier.title()} {story_object.title()}"
                premise = conflict["premise"].format(role=role, object=story_object, setting=setting)
                themes = list(dict.fromkeys(genre["themes"] + tone["themes"] + conflict["themes"]))[:4]
                constraints = list(dict.fromkeys(genre["constraints"] + [f"keep the conflict centered on {conflict['engine']}"]))[:3]
                stories.append(
                    {
                        "id": story_id,
                        "title": title,
                        "genre": genre["genre"],
                        "tone": tone["tone"],
                        "premise": premise,
                        "themes": themes,
                        "characters": names,
                        "constraints": constraints,
                        "target_scene_count": 5,
                        "target_word_count": 7500,
                        "language": "ko",
                    }
                )
    return stories


def build_story_pack_manifest(stories: list[dict[str, Any]]) -> dict[str, Any]:
    genre_counts = Counter()
    tone_counts = Counter()
    conflict_counts = Counter()
    for story in stories:
        parts = str(story["id"]).split("-")
        genre_counts[parts[0]] += 1
        tone_counts[parts[1]] += 1
        conflict_counts[parts[-1]] += 1
    return {
        "story_count": len(stories),
        "expected_story_count": story_count(),
        "genre_family_counts": dict(sorted(genre_counts.items())),
        "tone_family_counts": dict(sorted(tone_counts.items())),
        "conflict_engine_counts": dict(sorted(conflict_counts.items())),
        "scene_count_distribution": dict(sorted(Counter(story["target_scene_count"] for story in stories).items())),
        "word_count_distribution": dict(sorted(Counter(story["target_word_count"] for story in stories).items())),
        "story_ids": [story["id"] for story in stories],
    }


def write_balanced_story_pack(output_dir: str | Path) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stories = build_balanced_story_pack()
    for story in stories:
        path = output_root / f"{story['id']}.yaml"
        path.write_text(yaml.safe_dump(story, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = build_story_pack_manifest(stories)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stories": stories, "manifest": manifest, "manifest_path": str(manifest_path)}
