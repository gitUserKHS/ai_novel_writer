#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from conarrative.config import load_config
from conarrative.db import Storage
from conarrative.llm import build_provider
from conarrative.models import OutlineGenerateRequest, SceneRequest, StoryCreate
from conarrative.orchestrator import Orchestrator
from conarrative.runtime_settings import RuntimeSettingsStore


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local CoNarrative generation/evaluation/training loop.")
    parser.add_argument("--app-config", required=True, help="Generation app config YAML")
    parser.add_argument("--story-file", required=True, help="Story YAML input")
    parser.add_argument("--scene-file", help="Optional scene YAML input for smoke mode")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--story-id", help="Override story id")
    parser.add_argument("--scene-limit", type=int, help="Optional limit for full generation mode")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--train-config", action="append", help="Optional training YAML preset. Pass multiple times to chain stages.")
    parser.add_argument("--train-action", choices=["skip", "dry-run", "run"], default="skip")
    parser.add_argument("--train-output-dir", help="Optional override for training output dir")
    return parser.parse_args()


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def summarize_command_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    def clean(text: str) -> str:
        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)

    return {
        "returncode": result.returncode,
        "stdout": clean(result.stdout),
        "stderr": clean(result.stderr),
    }


def read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_runtime(app_config_path: str | Path) -> tuple[Any, Storage, Orchestrator]:
    config = load_config(app_config_path)
    storage = Storage(config.workspace.database_path)
    runtime_store = RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend)
    provider = build_provider(runtime_store.load())
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)
    return config, storage, orchestrator


def create_story(storage: Storage, story_file: str | Path, story_id: str | None) -> Any:
    payload = read_yaml(story_file)
    if story_id:
        payload["id"] = story_id
    story = storage.create_story(StoryCreate(**payload))
    return story


def local_ollama_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    port = parsed.port
    return host in {"127.0.0.1", "localhost"} and port == 11434


def uses_local_ollama(settings: Any) -> bool:
    provider = str(getattr(settings, "provider", "") or "").lower()
    base_url = str(getattr(settings, "base_url", "") or "")
    if not base_url or not local_ollama_base_url(base_url):
        return False
    return provider in {"ollama", "openai_compatible"}


def configured_ollama_models(settings: Any) -> list[str]:
    models = {str(getattr(settings, "model", "") or "").strip()}
    role_models = getattr(settings, "role_models", {}) or {}
    if isinstance(role_models, dict):
        models.update(str(value or "").strip() for value in role_models.values())
    return sorted(model for model in models if model and "/" not in model and "\\" not in model)


def parse_ollama_ps(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    names: list[str] = []
    for line in lines[1:]:
        name = line.split()[0].strip()
        if name and name != "NAME":
            names.append(name)
    return names


def unload_local_ollama_models(settings: Any) -> dict[str, Any]:
    models = configured_ollama_models(settings)
    stop_results: dict[str, Any] = {}
    for model in models:
        stop_results[model] = summarize_command_result(run_command(["ollama", "stop", model]))
    ps_result = run_command(["ollama", "ps"])
    running_models = parse_ollama_ps(ps_result.stdout)
    remaining = [model for model in models if model in running_models]
    return {
        "models": models,
        "stop_results": stop_results,
        "ps": summarize_command_result(ps_result),
        "remaining": remaining,
        "ok": len(remaining) == 0,
    }


def smoke_generate(orchestrator: Orchestrator, story_id: str, scene_file: str | None) -> dict[str, Any]:
    if scene_file:
        request = SceneRequest(**read_yaml(scene_file))
    else:
        outline = orchestrator.storage.list_outline(story_id)
        if not outline:
            story = orchestrator.storage.get_story(story_id)
            if story is None:
                raise RuntimeError(f"Story not found: {story_id}")
            outline = orchestrator.generate_outline(story_id, OutlineGenerateRequest(scene_count=min(1, story.target_scene_count)))
        request = orchestrator.scene_request_from_card(outline[0])

    result = orchestrator.run_scene(story_id, request)
    manuscript = orchestrator.write_export_files(story_id)
    evaluation = orchestrator.write_evaluation_file(story_id)
    training_bundle = orchestrator.write_training_bundle(story_id)
    return {
        "mode": "smoke",
        "accepted_scene_id": result.accepted_scene.scene_id,
        "accepted_text_preview": result.accepted_scene.accepted_text[:400],
        "manuscript_path": manuscript["path"],
        "evaluation_path": evaluation["path"],
        "training_bundle_manifest": training_bundle["paths"]["manifest"],
        "training_bundle_paths": training_bundle["paths"],
    }


def full_generate(orchestrator: Orchestrator, story_id: str, scene_limit: int | None) -> dict[str, Any]:
    result = orchestrator.auto_write_novel(story_id, scene_limit=scene_limit)
    return {
        "mode": "full",
        "story_id": result["story_id"],
        "scene_count": result["scene_count"],
        "manuscript_path": result["manuscript"]["path"],
        "evaluation_path": result["evaluation"]["path"],
        "training_bundle_manifest": result["training_bundle"]["paths"]["manifest"],
        "training_bundle_paths": result["training_bundle"]["paths"],
    }


def pool_key_for_config(train_cfg: dict[str, Any]) -> str:
    explicit_pool = str(train_cfg.get("pool_key", "") or "").strip()
    if explicit_pool:
        return explicit_pool
    train_mode = str(train_cfg.get("mode", "sft"))
    if train_mode == "sft":
        return "writer_sft"
    if train_mode == "dpo":
        return "writer_dpo"
    if train_mode == "distill":
        return "distill_stepwise"
    raise ValueError(f"Unsupported train mode: {train_mode}")


def train_file_for_mode(training_bundle_paths: dict[str, str], train_cfg: dict[str, Any] | str) -> str:
    if isinstance(train_cfg, str):
        return train_file_for_mode(training_bundle_paths, {"mode": train_cfg})
    return training_bundle_paths[pool_key_for_config(train_cfg)]


def stage_output_dir(train_config_path: str | Path, output_dir: str | None, stage_index: int, stage_count: int) -> str:
    if output_dir:
        base = Path(output_dir)
        if stage_count == 1:
            return str(base)
        return str(base / f"{stage_index:02d}_{Path(train_config_path).stem}")
    return str(Path("outputs") / Path(train_config_path).stem)


def run_training(
    train_config_path: str | Path,
    training_bundle_paths: dict[str, str],
    action: str,
    output_dir: str | None,
    stage_index: int,
    stage_count: int,
    model_name_or_path: str | None = None,
) -> dict[str, Any]:
    train_cfg = read_yaml(train_config_path)
    train_mode = str(train_cfg.get("mode", "sft"))
    train_file = train_file_for_mode(training_bundle_paths, train_cfg)
    resolved_output_dir = stage_output_dir(train_config_path, output_dir, stage_index, stage_count)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_qlora.py"),
        "--config",
        str(train_config_path),
        "--train-file",
        train_file,
        "--output-dir",
        resolved_output_dir,
    ]
    if model_name_or_path:
        cmd += ["--model-name-or-path", model_name_or_path]
    if action == "dry-run":
        cmd.append("--dry-run")
    result = run_command(cmd)
    return {
        "stage_index": stage_index,
        "stage_count": stage_count,
        "config": str(train_config_path),
        "mode": train_mode,
        "pool_key": pool_key_for_config(train_cfg),
        "chain_from_previous": bool(train_cfg.get("chain_from_previous", True)),
        "action": action,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "train_file": train_file,
        "output_dir": resolved_output_dir,
        "model_name_or_path": model_name_or_path or str(train_cfg.get("model_name_or_path", "")),
    }


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {
        "ok": True,
        "steps": {},
    }

    if args.run_tests:
        test_result = run_command([sys.executable, "-m", "pytest", "-q"])
        summary["steps"]["tests"] = summarize_command_result(test_result)
        if test_result.returncode != 0:
            summary["ok"] = False
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            raise SystemExit(test_result.returncode)

    config, storage, orchestrator = build_runtime(args.app_config)
    health_ok, health_message = orchestrator.provider.health()
    summary["steps"]["health"] = {
        "ok": health_ok,
        "message": health_message,
    }
    if not health_ok:
        summary["ok"] = False
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    story = create_story(storage, args.story_file, args.story_id)
    summary["steps"]["story"] = {
        "story_id": story.id,
        "title": story.title,
        "config": str(Path(args.app_config)),
    }

    if args.mode == "smoke":
        generation = smoke_generate(orchestrator, story.id, args.scene_file)
    else:
        generation = full_generate(orchestrator, story.id, args.scene_limit)
    summary["steps"]["generation"] = generation

    train_configs = args.train_config or []
    if train_configs and args.train_action != "skip":
        if args.train_action == "run" and uses_local_ollama(config.backend):
            unload_result = unload_local_ollama_models(config.backend)
            summary["steps"]["ollama_unload"] = unload_result
            if not unload_result["ok"]:
                summary["ok"] = False
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                raise SystemExit(1)
        training_results: list[dict[str, Any]] = []
        previous_output_dir: str | None = None
        for stage_index, train_config in enumerate(train_configs, start=1):
            stage_cfg = read_yaml(train_config)
            chain_from_previous = bool(stage_cfg.get("chain_from_previous", True))
            train_result = run_training(
                train_config,
                generation["training_bundle_paths"],
                args.train_action,
                args.train_output_dir,
                stage_index,
                len(train_configs),
                previous_output_dir if stage_index > 1 and chain_from_previous else None,
            )
            training_results.append(train_result)
            if train_result["returncode"] != 0:
                summary["steps"]["training"] = training_results
                summary["ok"] = False
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                raise SystemExit(train_result["returncode"])
            previous_output_dir = train_result["output_dir"]
        summary["steps"]["training"] = training_results[0] if len(training_results) == 1 else training_results

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
