from __future__ import annotations

import json
from pathlib import Path

from conarrative.config import AppConfig
from conarrative.db import Storage
from conarrative.llm import build_provider
from conarrative.models import OutlineGenerateRequest, SceneRequest, StoryCreate
from conarrative.orchestrator import Orchestrator
from conarrative.training import export_training_corpus


def make_config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.workspace.root = str(tmp_path / "workspace")
    config.workspace.database_path = str(tmp_path / "workspace" / "conarrative.db")
    config.workspace.exports_dir = str(tmp_path / "workspace" / "exports")
    config.workspace.runtime_settings_path = str(tmp_path / "workspace" / "runtime_settings.json")
    config.ensure_directories()
    return config


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_export_training_corpus_writes_separated_training_files(tmp_path: Path):
    config = make_config(tmp_path)
    storage = Storage(config.workspace.database_path)
    story = storage.create_story(
        StoryCreate(
            title="Training Target",
            premise="A violinist keeps hearing one extra note after every performance.",
            themes=["echo", "obsession"],
            characters=["Mina", "Jae"],
        )
    )

    provider = build_provider(config.backend)
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)
    outline = orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=2))
    result = orchestrator.run_scene(
        story.id,
        SceneRequest(
            title=outline[0].title,
            pov=outline[0].pov,
            goal=outline[0].goal,
            location=outline[0].location,
            time_label=outline[0].time_label,
            summary_request=outline[0].summary_request,
            beats=outline[0].beats,
            must_include=outline[0].must_include,
            must_avoid=outline[0].must_avoid,
            desired_length_words=500,
            outline_card_id=outline[0].id,
        ),
    )
    provider.close()

    assert result.accepted_scene.scene_index == 1

    records = storage.list_dataset_records(story_id=story.id)
    manifest = export_training_corpus(records, tmp_path / "training")

    assert manifest["counts"]["accepted_sft"] >= 1
    assert manifest["counts"]["prompt_only_teacher"] >= 1

    accepted_rows = load_jsonl(tmp_path / "training" / "accepted_sft.jsonl")
    prompt_rows = load_jsonl(tmp_path / "training" / "prompt_only_teacher.jsonl")

    assert accepted_rows
    assert prompt_rows
    assert accepted_rows[0]["messages"][0]["content"].startswith("당신은 한국어 장편소설용 장면을 쓰는 작가다.")
    assert accepted_rows[0]["messages"][-1]["role"] == "assistant"
    assert "장면 요청" in accepted_rows[0]["messages"][1]["content"]
    assert prompt_rows[0]["messages"][-1]["role"] == "user"
