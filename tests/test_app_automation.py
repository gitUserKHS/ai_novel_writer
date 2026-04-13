from __future__ import annotations

import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from conarrative.app import build_generalist_command, build_one_click_command, build_training_command, create_app
from conarrative.config import load_config
from conarrative.models import GeneralistLoopRequest, OneClickLoopRequest, TrainingRunRequest


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


def wait_for_job(client: TestClient, job_id: str) -> dict:
    payload = None
    for _ in range(100):
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        payload = res.json()
        if payload["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert payload is not None
    return payload


def test_build_one_click_command_resolves_paths() -> None:
    cmd = build_one_click_command(
        OneClickLoopRequest(
            preset="qwen-native",
            mode="smoke",
            train_action="skip",
            train_preset="none",
            story_file="examples/story.yaml",
            scene_file="examples/scene_smoke.yaml",
            scene_limit=2,
            run_tests=True,
        )
    )

    assert "one_click_loop.ps1" in cmd[4]
    assert "-StoryFile" in cmd
    assert "-SceneFile" in cmd
    assert "-SceneLimit" in cmd
    assert "-RunTests" in cmd


def test_build_generalist_command_supports_resume_and_sharding() -> None:
    cmd = build_generalist_command(
        GeneralistLoopRequest(
            preset="qwen-loop-local-critic-world",
            mode="full",
            story_dir="examples/story_pack",
            story_offset=6,
            story_limit=3,
            resume=True,
        )
    )

    assert "one_click_generalist.ps1" in cmd[4]
    assert "-StoryOffset" in cmd
    assert "-StoryLimit" in cmd
    assert "-Resume" in cmd


def test_build_training_command_supports_dry_run() -> None:
    cmd = build_training_command(
        TrainingRunRequest(
            config="configs/training_qwen3_4b_sft_smoke.yaml",
            dry_run=True,
            print_config=True,
        )
    )

    assert "train_qlora.py" in cmd[1]
    assert "--dry-run" in cmd
    assert "--print-config" in cmd


def test_system_job_endpoints_submit_and_complete(tmp_path: Path, monkeypatch) -> None:
    config_path = write_test_config(tmp_path)
    app = create_app(load_config(config_path))
    client = TestClient(app)

    def fake_run_process_job(command, emit, cwd=None):
        emit("fake command start", 0.5)
        return {"command": command, "returncode": 0, "output_tail": ["ok"], "parsed_json": {"ok": True}}

    monkeypatch.setattr("conarrative.app.run_process_job", fake_run_process_job)

    one_click = client.post(
        "/api/system/jobs/one-click",
        json={
            "preset": "qwen-native",
            "mode": "smoke",
            "train_action": "skip",
            "train_preset": "none",
            "story_file": "examples/story.yaml",
            "scene_file": "examples/scene_smoke.yaml",
        },
    )
    assert one_click.status_code == 200
    one_click_job = wait_for_job(client, one_click.json()["id"])
    assert one_click_job["status"] == "succeeded"

    generalist = client.post(
        "/api/system/jobs/generalist",
        json={
            "preset": "mock",
            "mode": "smoke",
            "train_action": "skip",
            "train_preset": "none",
            "story_dir": "examples/story_pack",
            "resume": True,
        },
    )
    assert generalist.status_code == 200
    generalist_job = wait_for_job(client, generalist.json()["id"])
    assert generalist_job["status"] == "succeeded"

    training = client.post(
        "/api/system/jobs/train",
        json={
            "config": "configs/training_qwen3_4b_sft_smoke.yaml",
            "dry_run": True,
        },
    )
    assert training.status_code == 200
    training_job = wait_for_job(client, training.json()["id"])
    assert training_job["status"] == "succeeded"


def test_story_import_and_ui_preset_endpoints(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    app = create_app(load_config(config_path))
    client = TestClient(app)

    import_response = client.post(
        "/api/stories/import",
        json={
            "yaml_text": yaml.safe_dump(
                {
                    "title": "Imported Story",
                    "genre": "mystery",
                    "tone": "tense",
                    "premise": "A locked theater holds the missing clue.",
                    "themes": ["memory", "loss"],
                    "characters": ["Seo-yun", "Min-ho"],
                    "constraints": ["No time travel"],
                    "target_scene_count": 4,
                    "target_word_count": 5000,
                },
                allow_unicode=True,
            )
        },
    )
    assert import_response.status_code == 200
    imported = import_response.json()
    assert imported["title"] == "Imported Story"
    assert imported["id"].startswith("imported-story")

    preset_response = client.post(
        "/api/ui-presets",
        json={
            "kind": "runtime",
            "name": "local-qwen",
            "payload": {
                "provider": "ollama",
                "model": "qwen3:4b",
                "base_url": "http://127.0.0.1:11434",
            },
        },
    )
    assert preset_response.status_code == 200
    assert preset_response.json()["name"] == "local-qwen"

    listed = client.get("/api/ui-presets")
    assert listed.status_code == 200
    assert "runtime" in listed.json()["items"]
    assert listed.json()["items"]["runtime"][0]["payload"]["model"] == "qwen3:4b"

    deleted = client.delete("/api/ui-presets/runtime/local-qwen")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True

    listed_after_delete = client.get("/api/ui-presets")
    assert listed_after_delete.status_code == 200
    assert listed_after_delete.json()["items"].get("runtime", []) == []
