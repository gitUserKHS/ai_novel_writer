
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import uvicorn
import yaml

from .app import create_app
from .config import load_config
from .db import Storage
from .llm import build_provider
from .models import OutlineGenerateRequest, SceneRequest, StoryCreate
from .orchestrator import Orchestrator
from .runtime_settings import RuntimeSettingsStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoNarrative AutoNovel CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")
    subparsers.add_parser("health")
    subparsers.add_parser("list-stories")

    create_story = subparsers.add_parser("create-story")
    create_story.add_argument("--input-file", required=True)

    outline = subparsers.add_parser("outline")
    outline.add_argument("--story-id", required=True)
    outline.add_argument("--scene-count", type=int, default=None)

    run_scene = subparsers.add_parser("run-scene")
    run_scene.add_argument("--story-id", required=True)
    run_scene.add_argument("--input-file")
    run_scene.add_argument("--print-text", action="store_true")

    auto_novel = subparsers.add_parser("auto-novel")
    auto_novel.add_argument("--story-id", required=True)
    auto_novel.add_argument("--scene-limit", type=int)

    export = subparsers.add_parser("export")
    export.add_argument("--story-id", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--story-id", required=True)

    export_ds = subparsers.add_parser("export-datasets")
    export_ds.add_argument("--story-id", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser.parse_args()


def load_runtime(config_path: str) -> tuple[Any, Storage, RuntimeSettingsStore, Orchestrator]:
    config = load_config(config_path)
    storage = Storage(config.workspace.database_path)
    runtime_store = RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend)
    provider = build_provider(runtime_store.load())
    orchestrator = Orchestrator(storage=storage, provider=provider, config=config)
    return config, storage, runtime_store, orchestrator


def read_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main() -> None:
    args = parse_args()
    config, storage, runtime_store, orchestrator = load_runtime(args.config)

    if args.command == "init":
        print(json.dumps({"ok": True, "workspace": config.workspace.model_dump(), "runtime_settings": runtime_store.load().model_dump()}, ensure_ascii=False, indent=2))
        return

    if args.command == "health":
        provider = build_provider(runtime_store.load())
        ok, detail = provider.health()
        print(json.dumps({"ok": ok, "detail": detail, "settings": runtime_store.load().model_dump()}, ensure_ascii=False, indent=2))
        return

    if args.command == "list-stories":
        print(json.dumps([story.model_dump(mode="json") for story in storage.list_stories()], ensure_ascii=False, indent=2))
        return

    if args.command == "create-story":
        payload = StoryCreate(**read_yaml(args.input_file))
        story = storage.create_story(payload)
        print(json.dumps(story.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "outline":
        scene_count = args.scene_count
        if scene_count is None:
            story = storage.get_story(args.story_id)
            if story is None:
                raise SystemExit(f"Story not found: {args.story_id}")
            scene_count = story.target_scene_count
        cards = orchestrator.generate_outline(args.story_id, OutlineGenerateRequest(scene_count=scene_count))
        print(json.dumps([card.model_dump(mode="json") for card in cards], ensure_ascii=False, indent=2))
        return

    if args.command == "run-scene":
        if args.input_file:
            request = SceneRequest(**read_yaml(args.input_file))
        else:
            outline_cards = [card for card in storage.list_outline(args.story_id) if card.status != "done"]
            if not outline_cards:
                story = storage.get_story(args.story_id)
                if story is None:
                    raise SystemExit(f"Story not found: {args.story_id}")
                orchestrator.generate_outline(args.story_id, OutlineGenerateRequest(scene_count=story.target_scene_count))
                outline_cards = [card for card in storage.list_outline(args.story_id) if card.status != "done"]
            request = orchestrator.scene_request_from_card(outline_cards[0])

        result = orchestrator.run_scene(args.story_id, request)
        if args.print_text:
            print(result.accepted_scene.accepted_text)
        else:
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "auto-novel":
        result = orchestrator.auto_write_novel(args.story_id, scene_limit=args.scene_limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "export":
        result = orchestrator.write_export_files(args.story_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        result = orchestrator.write_evaluation_file(args.story_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "export-datasets":
        result = orchestrator.write_training_bundle(args.story_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "serve":
        app = create_app(config)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return


if __name__ == "__main__":
    main()
