from __future__ import annotations

import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from conarrative.app import (
    build_generalist_command,
    build_hf_onboard_command,
    build_hf_pull_command,
    build_hf_publish_command,
    build_one_click_command,
    suggest_hf_release,
    search_hf_hub,
    build_training_command,
    create_app,
)
from conarrative.config import load_config
from conarrative.models import (
    GeneralistLoopRequest,
    HFOnboardRequest,
    HFPullRequest,
    HFPublishRequest,
    OneClickLoopRequest,
    TrainingRunRequest,
)


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


def test_build_hf_commands_support_repo_sync() -> None:
    publish_cmd = build_hf_publish_command(
        HFPublishRequest(
            source_dir="outputs/training_qwen3_4b_sft",
            namespace="your-org",
            project="conarrative",
            role="writer",
            base_model="Qwen/Qwen3-4B",
            stage="sft",
            repo_type="model",
            private=True,
            exclude_checkpoints=True,
            auto_tag=True,
            release_prefix="v",
            ignore_patterns=["*.pt"],
        )
    )
    assert "publish_to_hf.py" in publish_cmd[1]
    assert publish_cmd[2] == "publish"
    assert "--private" in publish_cmd
    assert "--exclude-checkpoints" in publish_cmd
    assert "--namespace" in publish_cmd
    assert "--auto-tag" in publish_cmd
    assert "--ignore-pattern" in publish_cmd

    pull_cmd = build_hf_pull_command(
        HFPullRequest(
            repo_id="your-org/conarrative-critic-qwen3-4b-lora",
            repo_type="model",
            local_dir="outputs/hf_download/critic",
            allow_patterns=["adapter_*"],
            ignore_patterns=["checkpoint-*"],
        )
    )
    assert "publish_to_hf.py" in pull_cmd[1]
    assert pull_cmd[2] == "pull"
    assert "--allow-pattern" in pull_cmd
    assert "--ignore-pattern" in pull_cmd

    onboard_cmd = build_hf_onboard_command(
        HFOnboardRequest(
            writer_repo_id="your-org/conarrative-writer-qwen3-4b-sft-lora",
            critic_repo_id="your-org/conarrative-critic-qwen3-4b-consistency-lora",
            world_repo_id="your-org/conarrative-world-model-qwen3-4b-sft-lora",
            preset_name="team-runtime",
            save_ui_preset=True,
        ),
        ui_presets_path=Path("workspace/ui_presets.json"),
    )
    assert "onboard_hf_collaborator.py" in onboard_cmd[1]
    assert "--writer-repo-id" in onboard_cmd
    assert "--critic-repo-id" in onboard_cmd
    assert "--world-repo-id" in onboard_cmd
    assert "--save-ui-preset" in onboard_cmd
    assert "--ui-presets-path" in onboard_cmd


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

    hf_publish = client.post(
        "/api/system/jobs/hf-publish",
        json={
            "source_dir": "outputs/training_qwen3_4b_sft",
            "repo_id": "your-org/conarrative-writer-qwen3-4b-lora",
            "repo_type": "model",
        },
    )
    assert hf_publish.status_code == 200
    hf_publish_job = wait_for_job(client, hf_publish.json()["id"])
    assert hf_publish_job["status"] == "succeeded"

    hf_pull = client.post(
        "/api/system/jobs/hf-pull",
        json={
            "repo_id": "your-org/conarrative-critic-qwen3-4b-lora",
            "repo_type": "model",
            "local_dir": "outputs/hf_download/critic",
        },
    )
    assert hf_pull.status_code == 200
    hf_pull_job = wait_for_job(client, hf_pull.json()["id"])
    assert hf_pull_job["status"] == "succeeded"

    hf_onboard = client.post(
        "/api/system/jobs/hf-onboard",
        json={
            "writer_repo_id": "your-org/conarrative-writer-qwen3-4b-sft-lora",
            "critic_repo_id": "your-org/conarrative-critic-qwen3-4b-consistency-lora",
            "world_repo_id": "your-org/conarrative-world-model-qwen3-4b-sft-lora",
            "preset_name": "team-runtime",
            "save_ui_preset": True,
        },
    )
    assert hf_onboard.status_code == 200
    hf_onboard_job = wait_for_job(client, hf_onboard.json()["id"])
    assert hf_onboard_job["status"] == "succeeded"


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


def test_suggest_hf_release_builds_standard_repo_id() -> None:
    payload = suggest_hf_release(
        namespace="your-org",
        repo_type="model",
        project="conarrative",
        role="writer",
        base_model="Qwen/Qwen3-4B",
        stage="sft",
        release_prefix="v",
    )
    assert payload["repo_id"] == "your-org/conarrative-writer-qwen3-4b-sft-lora"
    assert payload["suggested_tag"].startswith("v")


def test_hf_repo_browser_endpoint(tmp_path: Path, monkeypatch) -> None:
    config_path = write_test_config(tmp_path)
    app = create_app(load_config(config_path))
    client = TestClient(app)

    def fake_search_hf_hub(repo_type: str, search: str = "", author: str = "", limit: int = 12) -> dict:
        return {
            "items": [
                {
                    "repo_id": "your-org/conarrative-writer-qwen3-4b-lora",
                    "repo_type": repo_type,
                    "author": author,
                    "downloads": 42,
                    "likes": 7,
                    "private": False,
                    "last_modified": "2026-04-14T00:00:00Z",
                    "sha": "abc123",
                    "tags": ["conarrative", "peft"],
                }
            ],
            "repo_type": repo_type,
            "search": search,
            "author": author,
            "limit": limit,
        }

    monkeypatch.setattr("conarrative.app.search_hf_hub", fake_search_hf_hub)
    response = client.get("/api/hf/repos?repo_type=model&search=conarrative&author=your-org&limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_type"] == "model"
    assert payload["items"][0]["repo_id"] == "your-org/conarrative-writer-qwen3-4b-lora"


def test_hf_release_suggest_endpoint(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    app = create_app(load_config(config_path))
    client = TestClient(app)

    response = client.get(
        "/api/hf/suggest-release?namespace=your-org&repo_type=model&project=conarrative&role=writer&base_model=Qwen/Qwen3-4B&stage=sft&release_prefix=v"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_id"] == "your-org/conarrative-writer-qwen3-4b-sft-lora"
    assert payload["suggested_tag"].startswith("v")
