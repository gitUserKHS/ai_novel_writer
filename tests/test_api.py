from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from conarrative.app import create_app
from conarrative.config import AppConfig
from conarrative.models import AutoConnectOut, LocalModelCatalogOut, LocalModelOption, RuntimeSettings


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


def test_delete_story_removes_story_data_and_exports(tmp_path: Path):
    config = make_config(tmp_path)
    app = create_app(config)
    client = TestClient(app)

    quickstart = client.post(
        "/api/quickstart",
        json={
            "prompt": "A restorer keeps finding fresh paint on a portrait that should be dry.",
            "scene_count": 3,
            "desired_length_words": 650,
        },
    )
    assert quickstart.status_code == 200
    story_id = quickstart.json()["story"]["id"]

    export_response = client.post(f"/api/stories/{story_id}/export")
    assert export_response.status_code == 200

    story_dir = Path(config.workspace.exports_dir) / story_id
    assert story_dir.exists()

    with client.app.state.storage.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM stories WHERE id = ?", (story_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM outline_cards WHERE story_id = ?", (story_id,)).fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM scenes WHERE story_id = ?", (story_id,)).fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM artifacts WHERE story_id = ?", (story_id,)).fetchone()[0] >= 1

    delete_response = client.delete(f"/api/stories/{story_id}")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert delete_response.json()["story_id"] == story_id
    assert not story_dir.exists()

    detail = client.get(f"/api/stories/{story_id}")
    assert detail.status_code == 404

    listed = client.get("/api/stories")
    assert listed.status_code == 200
    assert all(item["id"] != story_id for item in listed.json()["items"])

    with client.app.state.storage.connect() as conn:
        for table, column in [
            ("stories", "id"),
            ("story_bibles", "story_id"),
            ("outline_cards", "story_id"),
            ("scenes", "story_id"),
            ("scene_candidates", "story_id"),
            ("state_snapshots", "story_id"),
            ("kg_edges", "story_id"),
            ("dataset_records", "story_id"),
            ("artifacts", "story_id"),
        ]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (story_id,)).fetchone()[0]
            assert count == 0, f"{table} still has rows for {story_id}"


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


def test_quickstart_creates_story_outline_and_first_scene(tmp_path: Path):
    config = make_config(tmp_path)
    runtime_path = Path(config.workspace.runtime_settings_path)
    runtime_path.write_text(
        '{"provider":"openai_compatible","base_url":"http://127.0.0.1:1/v1","model":"offline-model"}',
        encoding="utf-8",
    )

    app = create_app(config)
    client = TestClient(app)

    response = client.post(
        "/api/quickstart",
        json={
            "prompt": "A closed theater keeps answering questions nobody says aloud.",
            "scene_count": 4,
            "desired_length_words": 700,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai_compatible"
    assert payload["story"]["title"]
    assert len(payload["outline"]) == 4
    assert len(payload["recent_scenes"]) == 1
    assert payload["state"]["last_scene_index"] == 1
    assert "Trying your local model first" in payload["detail"]


def test_quickstart_fallback_handles_empty_themes(tmp_path: Path):
    config = make_config(tmp_path)
    runtime_path = Path(config.workspace.runtime_settings_path)
    runtime_path.write_text(
        '{"provider":"openai_compatible","base_url":"http://127.0.0.1:1/v1","model":"offline-model"}',
        encoding="utf-8",
    )

    app = create_app(config)
    client = TestClient(app)

    response = client.post(
        "/api/quickstart",
        json={
            "prompt": "A locksmith opens a door that remembers every visitor.",
            "scene_count": 4,
            "desired_length_words": 700,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["story"]["themes"] == []
    assert len(payload["outline"]) == 4
    assert len(payload["recent_scenes"]) == 1
    assert payload["state"]["last_scene_index"] == 1


def test_quickstart_can_create_multiple_stories_with_reused_outline_ids(tmp_path: Path):
    config = make_config(tmp_path)
    runtime_path = Path(config.workspace.runtime_settings_path)
    runtime_path.write_text(
        '{"provider":"openai_compatible","base_url":"http://127.0.0.1:1/v1","model":"offline-model"}',
        encoding="utf-8",
    )

    app = create_app(config)
    client = TestClient(app)

    first = client.post(
        "/api/quickstart",
        json={
            "prompt": "A singer hears tomorrow's applause before she has written the song.",
            "scene_count": 3,
            "desired_length_words": 650,
        },
    )
    second = client.post(
        "/api/quickstart",
        json={
            "prompt": "A night guard finds a notebook that logs doors opening before anyone touches them.",
            "scene_count": 3,
            "desired_length_words": 650,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()
    first_story_id = first_payload["story"]["id"]
    second_story_id = second_payload["story"]["id"]

    assert first_story_id != second_story_id
    assert all(item["id"].startswith(f"{first_story_id}-outline-") for item in first_payload["outline"])
    assert all(item["id"].startswith(f"{second_story_id}-outline-") for item in second_payload["outline"])
    assert {item["id"] for item in first_payload["outline"]}.isdisjoint({item["id"] for item in second_payload["outline"]})


def test_auto_connect_endpoint_saves_detected_backend(tmp_path: Path, monkeypatch):
    app = create_app(make_config(tmp_path))
    client = TestClient(app)
    detected = RuntimeSettings(
        provider="openai_compatible",
        base_url="http://127.0.0.1:11434/v1",
        model="llama3.1:latest",
        api_key="not-needed",
    )

    def fake_detect(_: RuntimeSettings) -> AutoConnectOut:
        return AutoConnectOut(
            found=True,
            source="Ollama",
            detail="Ollama detected at http://127.0.0.1:11434/v1 with model llama3.1:latest.",
            settings=detected,
            available_models=["llama3.1:latest", "nomic-embed-text:latest"],
        )

    monkeypatch.setattr("conarrative.app.detect_runtime_settings", fake_detect)

    response = client.post("/api/runtime-settings/auto-connect")

    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is True
    assert payload["source"] == "Ollama"
    assert payload["settings"]["provider"] == "openai_compatible"
    assert payload["settings"]["model"] == "llama3.1:latest"
    current = client.get("/api/runtime-settings").json()
    assert current["provider"] == "openai_compatible"
    assert current["base_url"] == "http://127.0.0.1:11434/v1"


def test_runtime_model_catalog_and_selection(tmp_path: Path, monkeypatch):
    option_one = LocalModelOption(source="Ollama", base_url="http://127.0.0.1:11434/v1", model="llama3.1:latest")
    option_two = LocalModelOption(source="Ollama", base_url="http://127.0.0.1:11434/v1", model="mistral:latest")
    fake_catalog = LocalModelCatalogOut(
        options=[option_one, option_two],
        current=option_one,
        detail="Detected 2 local model option(s).",
    )

    monkeypatch.setattr("conarrative.app.auto_connect_settings", lambda current: (current, fake_catalog, False))
    monkeypatch.setattr("conarrative.app.build_catalog_from_settings", lambda current, catalog=None: catalog or fake_catalog)

    app = create_app(make_config(tmp_path))
    client = TestClient(app)

    catalog_response = client.get("/api/runtime-settings/models")
    assert catalog_response.status_code == 200
    catalog_payload = catalog_response.json()
    assert len(catalog_payload["options"]) == 2
    assert catalog_payload["current"]["model"] == "llama3.1:latest"

    select_response = client.put(
        "/api/runtime-settings/select-model",
        json={
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "mistral:latest",
        },
    )
    assert select_response.status_code == 200
    assert select_response.json()["model"] == "mistral:latest"

    saved_settings = client.get("/api/runtime-settings").json()
    assert saved_settings["provider"] == "openai_compatible"
    assert saved_settings["model"] == "mistral:latest"


def test_continue_story_uses_next_outline_card(tmp_path: Path):
    app = create_app(make_config(tmp_path))
    client = TestClient(app)

    quickstart = client.post(
        "/api/quickstart",
        json={
            "prompt": "A woman receives letters from tomorrow that warn her about one room in her apartment.",
            "scene_count": 3,
            "desired_length_words": 650,
        },
    )
    assert quickstart.status_code == 200
    story_id = quickstart.json()["story"]["id"]

    continued = client.post(
        f"/api/stories/{story_id}/continue",
        json={"desired_length_words": 650},
    )

    assert continued.status_code == 200
    payload = continued.json()
    assert len(payload["recent_scenes"]) == 2
    assert payload["state"]["last_scene_index"] == 2

    outline = client.get(f"/api/stories/{story_id}/outline")
    assert outline.status_code == 200
    statuses = [item["status"] for item in outline.json()["items"]]
    assert statuses[:2] == ["used", "used"]


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
