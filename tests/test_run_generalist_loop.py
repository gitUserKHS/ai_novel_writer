from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conarrative.config import load_config
from conarrative.db import Storage
from conarrative.llm import BaseLLMProvider
from conarrative.models import (
    ConsistencyIssue,
    ConsistencyReport,
    CreativityReport,
    DraftCandidate,
    ExtractionOutput,
    OutlineGenerateRequest,
    PlanOutput,
    RuntimeSettings,
    SceneRequest,
    Severity,
    StoryCreate,
)
from conarrative.orchestrator import Orchestrator
from scripts.run_generalist_loop import resume_smoke_generate, select_story_files, story_resume_state
from scripts.run_pipeline import build_runtime


def write_test_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "test.yaml"
    payload = {
        "app_name": "CoNarrative Test",
        "workspace": {
            "root": "workspace",
            "database_path": "workspace/conarrative.db",
            "exports_dir": "workspace/exports",
            "runtime_settings_path": "workspace/runtime_settings.json",
        },
        "backend": {
            "provider": "mock",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "not-needed",
            "model": "mock-story-engine",
            "timeout_seconds": 60,
            "temperature": 0.9,
            "critic_temperature": 0.2,
            "max_tokens": 2048,
            "extra_headers": {},
            "role_models": {},
            "cache_responses": False,
            "cache_dir": "workspace/cache",
        },
        "orchestration": {
            "recent_scene_memory": 3,
            "adaptive_outline": True,
            "auto_revision": True,
            "minimum_release_consistency": 0.7,
            "consistency_weight": 0.58,
            "creativity_weight": 0.27,
            "world_model_weight": 0.15,
            "max_summary_memory": 6,
        },
    }
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return config_path


def make_story(story_id: str, title: str, premise: str, target_scene_count: int = 1) -> StoryCreate:
    return StoryCreate(
        id=story_id,
        title=title,
        genre="mystery fantasy",
        tone="lyrical and tense",
        premise=premise,
        themes=["memory", "choice", "loss"],
        characters=["Seoyun", "Minho", "Yujin"],
        constraints=["no time travel", "no resurrection"],
        target_scene_count=target_scene_count,
        language="ko",
    )


def test_select_story_files_applies_offset_and_limit() -> None:
    story_files = [Path(f"story_{idx:02d}.yaml") for idx in range(6)]

    selected = select_story_files(story_files, story_offset=2, story_limit=3)

    assert selected == story_files[2:5]


def test_story_resume_state_detects_completed_manifest_and_smoke_resume(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    config, storage, orchestrator = build_runtime(config_path)
    story = storage.create_story(make_story("resume-story", "Resume Story", "A hidden ledger must be recovered."))

    orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=1))
    orchestrator.auto_write_novel(story.id, scene_limit=1)
    bundle = orchestrator.write_training_bundle(story.id)

    story_file = tmp_path / "resume_story.yaml"
    story_file.write_text(
        yaml.safe_dump(make_story("resume-story", "Resume Story", "A hidden ledger must be recovered.").model_dump(mode="json"), allow_unicode=True),
        encoding="utf-8",
    )

    state = story_resume_state(storage, config.workspace.exports_dir, story_file)

    assert state["story_id"] == "resume-story"
    assert state["completed"] is True
    assert state["manifest_path"] == Path(bundle["paths"]["manifest"])

    resumed = resume_smoke_generate(orchestrator, story.id, None)

    assert resumed["mode"] == "smoke_resume"
    assert Path(resumed["training_bundle_manifest"]).exists()


class AlwaysBadReleaseGateProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(RuntimeSettings())
        self.write_calls = 0

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def generate_outline(self, memory_bundle, scene_count):
        raise NotImplementedError

    def plan_scene(self, memory_bundle, request):
        return PlanOutput(scene_title=request.title_hint or "Strict Gate", target_word_count=160)

    def write_candidates(self, memory_bundle, request, plan, count=3):
        self.write_calls += 1
        return [DraftCandidate(text=f"bad candidate {self.write_calls}")]

    def critique_consistency(self, memory_bundle, request, plan, text):
        return ConsistencyReport(
            score=0.41,
            world_plausibility_score=0.45,
            issues=[
                ConsistencyIssue(
                    issue_type="required_fact",
                    severity=Severity.HIGH,
                    message="required fact missing",
                )
            ],
        )

    def critique_creativity(self, memory_bundle, request, plan, text):
        return CreativityReport(
            novelty_score=0.5,
            hook_score=0.5,
            emotional_depth_score=0.5,
            style_fit_score=0.5,
            surprise_score=0.5,
        )

    def revise_scene(self, memory_bundle, request, plan, text, issues):
        return NotImplementedError

    def extract_scene(self, memory_bundle, request, plan, text):
        return ExtractionOutput(summary=text[:80])


def test_run_scene_strict_release_gate_raises_after_rescue_exhaustion(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    storage = Storage(config.workspace.database_path)
    provider = AlwaysBadReleaseGateProvider()
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)

    config.orchestration.candidate_count = 1
    config.orchestration.auto_revision = False
    config.orchestration.minimum_release_consistency = 0.7
    config.orchestration.release_gate_world_min_plausibility = 0.68
    config.orchestration.release_gate_max_medium_issues = 0
    config.orchestration.release_gate_rescue_rounds = 1
    config.orchestration.release_gate_rescue_candidate_count = 1
    config.orchestration.strict_release_gate = True

    story = storage.create_story(make_story("strict-gate", "Strict Gate", "A strict gate must reject bad scenes."))

    with pytest.raises(RuntimeError, match="Release gate failed after rescue rounds"):
        orchestrator.run_scene(
            story.id,
            SceneRequest(
                title_hint="Strict Scene",
                pov="Seoyun",
                location="Archive",
                time_label="D1 22:00",
                goal="Recover the ledger",
                required_facts=["ledger"],
            ),
        )

    assert provider.write_calls >= 2
