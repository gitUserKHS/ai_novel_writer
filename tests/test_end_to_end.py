from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from conarrative.app import create_app
from conarrative.config import load_config
from conarrative.db import Storage
from conarrative.llm import BaseLLMProvider, build_provider
from conarrative.models import DraftCandidate, OutlineCard, OutlineGenerateRequest, PlanOutput, RuntimeSettings, SceneRequest, StoryCreate
from conarrative.models import ConsistencyIssue, ConsistencyReport, CreativityReport, ExtractionOutput, Severity
from conarrative.orchestrator import Orchestrator
from conarrative.runtime_settings import RuntimeSettingsStore


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


def build_runtime(config_path: Path) -> tuple[Storage, Orchestrator]:
    config = load_config(config_path)
    storage = Storage(config.workspace.database_path)
    runtime_store = RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend)
    provider = build_provider(runtime_store.load())
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)
    return storage, orchestrator


def make_story(story_id: str, title: str, premise: str, target_scene_count: int = 4) -> StoryCreate:
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


def test_end_to_end_orchestrator(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    storage, orchestrator = build_runtime(config_path)
    story = storage.create_story(make_story("moon-theater", "Moon Theater", "A closed theater hides a sibling's trace.", target_scene_count=4))

    cards = orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=4))
    assert len(cards) == 4

    result = orchestrator.auto_write_novel(story.id)
    assert result["scene_count"] == 4

    scenes = storage.list_scenes(story.id)
    assert len(scenes) == 4
    assert all(scene["accepted_text"] for scene in scenes)

    export_path = Path(result["manuscript"]["artifact"]["path"])
    evaluation_path = Path(result["evaluation"]["artifact"]["path"])
    assert export_path.exists()
    assert evaluation_path.exists()

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["story_id"] == story.id
    assert evaluation["scene_count"] == 4
    assert storage.dataset_counts(story.id)["accepted"] == 4


def test_api_flow(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    app = create_app(config)
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "degraded"}

    create = client.post(
        "/api/stories",
        json=make_story("api-story", "Moon Theater", "A closed theater hides a sibling's trace.", target_scene_count=3).model_dump(),
    )
    assert create.status_code == 200
    story = create.json()

    outline = client.post(f"/api/stories/{story['id']}/outline/generate", json={"scene_count": 3})
    assert outline.status_code == 200
    assert len(outline.json()["items"]) == 3

    job = client.post(f"/api/stories/{story['id']}/jobs/auto-novel?scene_limit=3")
    assert job.status_code == 200
    job_id = job.json()["id"]

    payload = None
    for _ in range(100):
        current = client.get(f"/api/jobs/{job_id}")
        assert current.status_code == 200
        payload = current.json()
        if payload["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert payload is not None
    assert payload["status"] == "succeeded", payload

    detail = client.get(f"/api/stories/{story['id']}")
    assert detail.status_code == 200
    scenes = client.get(f"/api/stories/{story['id']}/scenes")
    assert scenes.status_code == 200
    assert len(scenes.json()["items"]) == 3

    export_res = client.post(f"/api/stories/{story['id']}/export")
    assert export_res.status_code == 200
    eval_res = client.post(f"/api/stories/{story['id']}/evaluate")
    assert eval_res.status_code == 200

    artifacts = client.get(f"/api/stories/{story['id']}/artifacts")
    assert artifacts.status_code == 200
    items = artifacts.json()["items"]
    assert items
    download = client.get("/api/artifacts/download", params={"path": items[0]["path"]})
    assert download.status_code == 200

    root = client.get("/")
    assert root.status_code == 200
    assert "CoNarrative AutoNovel Studio" in root.text


def test_training_bundle_export(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    storage, orchestrator = build_runtime(config_path)
    story = storage.create_story(make_story("ember-house", "Ember House", "A return home reveals a living house.", target_scene_count=2))

    orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=2))
    orchestrator.auto_write_novel(story.id, scene_limit=2)
    bundle = orchestrator.write_training_bundle(story.id)
    manifest_path = Path(bundle["paths"]["manifest"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["writer_sft"] == 2
    assert manifest["counts"]["critic_consistency_sft"] >= 2
    assert Path(bundle["paths"]["world_model_transitions"]).exists()


class UnderfilledCandidateProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(RuntimeSettings())
        self.calls = 0

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def generate_outline(self, memory_bundle, scene_count):
        raise NotImplementedError

    def plan_scene(self, memory_bundle, request):
        raise NotImplementedError

    def write_candidates(self, memory_bundle, request, plan, count=3):
        self.calls += 1
        return [DraftCandidate(text=f"candidate-{self.calls}")]

    def critique_consistency(self, memory_bundle, request, plan, text):
        raise NotImplementedError

    def critique_creativity(self, memory_bundle, request, plan, text):
        raise NotImplementedError

    def revise_scene(self, memory_bundle, request, plan, text, issues):
        raise NotImplementedError

    def extract_scene(self, memory_bundle, request, plan, text):
        raise NotImplementedError


def test_generate_candidates_tops_up_underfilled_provider(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    provider = UnderfilledCandidateProvider()
    orchestrator = Orchestrator(storage=Storage(config.workspace.database_path), provider=provider, config=config)

    config.orchestration.candidate_count = 3
    candidates = orchestrator._generate_candidates(
        memory_bundle={},
        request=SceneRequest(),
        plan=PlanOutput(scene_title="test"),
    )

    assert len(candidates) == 3
    assert provider.calls >= 3


def test_generate_outline_namespaces_duplicate_provider_ids(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    _, orchestrator = build_runtime(config_path)

    cards = orchestrator._normalize_outline_ids(
        "id-test",
        [
            OutlineCard(
                id="scene_1",
                scene_index=1,
                title="First Scene",
                pov="Seoyun",
                location="Makeup Room",
                time_label="D1 22:00",
                goal="Find the clue",
                beat="Inspect the trace",
            ),
            OutlineCard(
                id="scene_1",
                scene_index=2,
                title="Second Scene",
                pov="Minho",
                location="Lobby",
                time_label="D1 23:00",
                goal="Track the lie",
                beat="Distrust grows",
            ),
        ],
    )

    assert cards[0].id == "id-test-scene-1"
    assert cards[1].id.startswith("id-test-oc002")


class ReleaseGateRescueProvider(BaseLLMProvider):
    def __init__(self) -> None:
        super().__init__(RuntimeSettings())
        self.write_calls = 0

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def generate_outline(self, memory_bundle, scene_count):
        raise NotImplementedError

    def plan_scene(self, memory_bundle, request):
        return PlanOutput(scene_title=request.title_hint or "Gate Test", target_word_count=180)

    def write_candidates(self, memory_bundle, request, plan, count=3):
        self.write_calls += 1
        if self.write_calls == 1:
            return [DraftCandidate(text="bad candidate")]
        return [DraftCandidate(text="good candidate")]

    def critique_consistency(self, memory_bundle, request, plan, text):
        if "good" in text:
            return ConsistencyReport(score=0.9, world_plausibility_score=0.86, issues=[])
        return ConsistencyReport(
            score=0.45,
            world_plausibility_score=0.44,
            issues=[
                ConsistencyIssue(
                    issue_type="required_fact",
                    severity=Severity.HIGH,
                    message="missing required fact",
                )
            ],
        )

    def critique_creativity(self, memory_bundle, request, plan, text):
        return CreativityReport(
            novelty_score=0.6,
            hook_score=0.6,
            emotional_depth_score=0.6,
            style_fit_score=0.6,
            surprise_score=0.6,
        )

    def revise_scene(self, memory_bundle, request, plan, text, issues):
        return NotImplementedError

    def extract_scene(self, memory_bundle, request, plan, text):
        return ExtractionOutput(summary=text[:120])


def test_run_scene_release_gate_rescues_better_candidate(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    storage = Storage(config.workspace.database_path)
    provider = ReleaseGateRescueProvider()
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)

    config.orchestration.candidate_count = 1
    config.orchestration.auto_revision = False
    config.orchestration.minimum_release_consistency = 0.7
    config.orchestration.release_gate_world_min_plausibility = 0.68
    config.orchestration.release_gate_max_medium_issues = 0
    config.orchestration.release_gate_rescue_rounds = 1
    config.orchestration.release_gate_rescue_candidate_count = 1

    story = storage.create_story(make_story("gate-story", "Gate Story", "A clue must be recovered.", target_scene_count=1))
    result = orchestrator.run_scene(
        story.id,
        SceneRequest(
            title_hint="Gate Scene",
            pov="Seoyun",
            location="Archive",
            time_label="D1 22:00",
            goal="Recover the clue",
            required_facts=["clue"],
        ),
    )

    assert result.accepted_scene.accepted_text == "good candidate"
    assert provider.write_calls >= 2
