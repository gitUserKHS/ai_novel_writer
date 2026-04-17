from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from conarrative.app import create_app
from conarrative.config import AppConfig


def make_config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.workspace.root = str(tmp_path / "workspace")
    config.workspace.database_path = str(tmp_path / "workspace" / "conarrative.db")
    config.workspace.exports_dir = str(tmp_path / "workspace" / "exports")
    config.workspace.runtime_settings_path = str(tmp_path / "workspace" / "runtime_settings.json")
    config.ensure_directories()
    return config


def wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        last = res.json()
        if last["status"] in {"succeeded", "failed"}:
            return last
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for job {job_id}; last={last}")


def test_story_patch_ignores_explicit_null_fields(tmp_path: Path):
    app = create_app(make_config(tmp_path))
    client = TestClient(app)

    story = client.post(
        "/api/stories",
        json={
            "title": "Patch Target",
            "premise": "A premise.",
            "themes": ["grief"],
            "characters": ["Mina"],
            "forbidden_facts": ["teleportation"],
        },
    ).json()

    response = client.patch(
        f"/api/stories/{story['id']}",
        json={"themes": None, "characters": None, "notes": "updated"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["themes"] == ["grief"]
    assert payload["characters"] == ["Mina"]
    assert payload["notes"] == "updated"


def test_runtime_settings_reject_invalid_provider(tmp_path: Path):
    app = create_app(make_config(tmp_path))
    client = TestClient(app)

    response = client.put("/api/runtime-settings", json={"provider": "bogus"})

    assert response.status_code == 422


def test_story_detail_survives_invalid_runtime_settings_file(tmp_path: Path):
    config = make_config(tmp_path)
    runtime_path = Path(config.workspace.runtime_settings_path)
    runtime_path.write_text('{"provider":"bogus","model":"custom-model","candidate_count":2}', encoding="utf-8")

    app = create_app(config)
    client = TestClient(app)

    story = client.post("/api/stories", json={"title": "Moon Theater", "premise": "A quiet theater answers back."}).json()

    detail = client.get(f"/api/stories/{story['id']}")
    settings = client.get("/api/runtime-settings")

    assert detail.status_code == 200
    assert settings.status_code == 200
    assert settings.json()["provider"] == "mock"
    assert settings.json()["model"] == "custom-model"
    assert settings.json()["candidate_count"] == 2


def test_story_flow_end_to_end(tmp_path: Path):
    app = create_app(make_config(tmp_path))
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "degraded"}

    story_payload = {
        "title": "Moon Theater",
        "genre": "mystery drama",
        "premise": "A closed theater keeps answering questions nobody says aloud.",
        "tone": "haunted and intimate",
        "themes": ["grief", "performance"],
        "characters": ["서윤", "민호"],
        "forbidden_facts": ["instant teleportation"],
        "notes": "Keep the camera close to gesture.",
        "target_length_scenes": 8,
    }
    story_res = client.post("/api/stories", json=story_payload)
    assert story_res.status_code == 200
    story = story_res.json()
    story_id = story["id"]

    detail = client.get(f"/api/stories/{story_id}")
    assert detail.status_code == 200
    assert detail.json()["story"]["title"] == "Moon Theater"

    outline = client.post(f"/api/stories/{story_id}/outline/generate", json={"scene_count": 4})
    assert outline.status_code == 200
    assert len(outline.json()["items"]) == 4

    scene_payload = {
        "title": "열쇠의 반짝임",
        "pov": "서윤",
        "goal": "숨겨진 열쇠의 흔적을 잡는다",
        "location": "버려진 극장 로비",
        "time_label": "Day 1 / night",
        "summary_request": "극장 로비에서 서윤이 과거 공연의 흔적과 맞닥뜨리며 다음 장면으로 이어질 질문을 만든다.",
        "beats": [
            "서윤이 로비의 냄새와 흔적을 더듬으며 들어온다",
            "낡은 포스터 뒤에서 이상한 금속성 반짝임을 발견한다",
            "마지막에는 누군가가 자신을 보고 있었다는 징후를 남긴다",
        ],
        "must_include": ["은색 열쇠고리"],
        "must_avoid": ["instant teleportation"],
        "emotion_targets": ["긴장", "애도"],
        "desired_length_words": 700,
    }
    job = client.post(f"/api/stories/{story_id}/scenes/generate", json=scene_payload)
    assert job.status_code == 200
    job_done = wait_for_job(client, job.json()["id"])
    assert job_done["status"] == "succeeded"
    assert job_done["result"]["accepted_scene"]["scene_index"] == 1

    scenes = client.get(f"/api/stories/{story_id}/scenes")
    assert scenes.status_code == 200
    scene_items = scenes.json()["items"]
    assert len(scene_items) == 1
    assert "은색 열쇠고리" in scene_items[0]["accepted_text"]

    state = client.get(f"/api/stories/{story_id}/state")
    assert state.status_code == 200
    assert state.json()["last_scene_index"] == 1

    datasets = client.get(f"/api/stories/{story_id}/datasets")
    assert datasets.status_code == 200
    assert datasets.json()["accepted"] >= 1
    assert datasets.json()["prompt_only"] >= 1

    export_res = client.post(f"/api/stories/{story_id}/export")
    assert export_res.status_code == 200
    manuscript_artifact = export_res.json()["artifact"]
    assert manuscript_artifact["artifact_type"] == "manuscript_markdown"

    eval_res = client.post(f"/api/stories/{story_id}/evaluate")
    assert eval_res.status_code == 200
    assert eval_res.json()["report"]["scene_count"] == 1

    artifacts = client.get(f"/api/stories/{story_id}/artifacts")
    assert artifacts.status_code == 200
    assert len(artifacts.json()["items"]) >= 2

    download = client.get(f"/api/stories/{story_id}/artifacts/{manuscript_artifact['id']}/download")
    assert download.status_code == 200
    assert "Moon Theater" in download.text
