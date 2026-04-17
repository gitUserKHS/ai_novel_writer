from __future__ import annotations

import json
from pathlib import Path

from conarrative.config import AppConfig
from conarrative.db import Storage
from conarrative.llm import build_provider
from conarrative.models import JobStatus, OutlineGenerateRequest, ProviderType, RuntimeSettings, SceneRequest, StoryCreate
from conarrative.orchestrator import Orchestrator
from conarrative.runtime_settings import RuntimeSettingsStore


def make_config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.workspace.root = str(tmp_path / "workspace")
    config.workspace.database_path = str(tmp_path / "workspace" / "conarrative.db")
    config.workspace.exports_dir = str(tmp_path / "workspace" / "exports")
    config.workspace.runtime_settings_path = str(tmp_path / "workspace" / "runtime_settings.json")
    config.ensure_directories()
    return config


def test_runtime_settings_store_salvages_valid_fields(tmp_path: Path):
    path = tmp_path / "runtime_settings.json"
    path.write_text(json.dumps({"provider": "bogus", "model": "custom-model", "candidate_count": 2}), encoding="utf-8")

    settings = RuntimeSettingsStore(str(path), RuntimeSettings()).load()

    assert settings.provider == ProviderType.MOCK
    assert settings.model == "custom-model"
    assert settings.candidate_count == 2


def test_recover_incomplete_jobs_marks_jobs_failed(tmp_path: Path):
    config = make_config(tmp_path)
    storage = Storage(config.workspace.database_path)
    story = storage.create_story(StoryCreate(title="Recovery Target", premise="A premise."))
    storage.create_job("job-1", story.id, "scene_generation")
    storage.append_job_log("job-1", "Job started", progress=0.5, status=JobStatus.RUNNING)

    recovered = storage.recover_incomplete_jobs()
    job = storage.get_job("job-1")

    assert recovered == 1
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.message == "Interrupted"
    assert "shutdown or restart" in job.error_text


def test_orchestrator_scene_and_evaluation(tmp_path: Path):
    config = make_config(tmp_path)
    storage = Storage(config.workspace.database_path)
    story = storage.create_story(
        StoryCreate(
            title="Ash River",
            genre="literary suspense",
            premise="Two siblings keep meeting the same river at different ages.",
            tone="quiet, haunting, intimate",
            themes=["memory", "river", "debt"],
            characters=["다해", "준"],
            forbidden_facts=["time travel"],
            target_length_scenes=6,
        )
    )
    provider = build_provider(config.backend)
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)

    outline = orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=3))
    assert len(outline) == 3

    result = orchestrator.run_scene(
        story.id,
        SceneRequest(
            title="강변의 흠집",
            pov="다해",
            goal="강변 계단에 남은 흔적의 의미를 알아낸다",
            location="강변 산책로",
            time_label="Day 1 / evening",
            summary_request="다해가 계단 표면의 흠집을 통해 오래된 부채의 실마리를 건드린다.",
            beats=[
                "다해가 강변 난간을 짚고 계단으로 내려간다",
                "젖은 콘크리트에 새겨진 흠집을 발견한다",
                "장면 끝에 준과 관련된 질문이 커진다",
            ],
            must_include=["젖은 흠집"],
            must_avoid=["time travel"],
            emotion_targets=["서늘함"],
            desired_length_words=650,
            outline_card_id=outline[0].id,
        ),
    )
    assert result.accepted_scene.scene_index == 1
    assert "젖은 흠집" in result.accepted_scene.accepted_text

    report = orchestrator.evaluate_story(story.id)
    assert report.scene_count == 1
    assert report.dataset_counts["accepted"] >= 1

    bundle = orchestrator.export_story_markdown(story.id)
    assert bundle["filename"].endswith(".md")
    assert "Ash River" in bundle["content"]
