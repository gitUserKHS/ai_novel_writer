from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Dict, Iterator

import uvicorn
import yaml

from .app import create_app
from .config import AppConfig, load_config
from .db import Storage
from .llm import build_provider
from .models import BibleContent, OutlineGenerateRequest, SceneRequest, StoryCreate, StoryUpdate
from .orchestrator import Orchestrator
from .runtime_settings import RuntimeSettingsStore


def load_structured_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


@contextmanager
def orchestrator_context(config: AppConfig) -> Iterator[Orchestrator]:
    storage = Storage(config.workspace.database_path)
    settings = RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend).load()
    provider = build_provider(settings)
    try:
        yield Orchestrator(storage=storage, provider=provider, config=config)
    finally:
        provider.close()


def cmd_init(config: AppConfig, args: argparse.Namespace) -> None:
    Storage(config.workspace.database_path)
    RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend).save(config.backend)
    print(f"Initialized workspace at {config.workspace.root}")
    print(f"Database: {config.workspace.database_path}")
    print(f"Runtime settings: {config.workspace.runtime_settings_path}")


def cmd_create_story(config: AppConfig, args: argparse.Namespace) -> None:
    storage = Storage(config.workspace.database_path)
    if args.input_file:
        payload = StoryCreate.model_validate(load_structured_file(args.input_file))
    else:
        payload = StoryCreate(
            title=args.title,
            genre=args.genre,
            premise=args.premise,
            tone=args.tone,
            themes=args.themes or [],
            characters=args.characters or [],
            forbidden_facts=args.forbidden or [],
            notes=args.notes or "",
            target_length_scenes=args.target_length_scenes,
        )
    story = storage.create_story(payload)
    print(json.dumps(story.model_dump(), ensure_ascii=False, indent=2))


def cmd_update_story(config: AppConfig, args: argparse.Namespace) -> None:
    storage = Storage(config.workspace.database_path)
    payload = StoryUpdate.model_validate(load_structured_file(args.input_file))
    updated = storage.update_story(args.story_id, payload)
    if updated is None:
        raise SystemExit(f"Story not found: {args.story_id}")
    print(json.dumps(updated.model_dump(), ensure_ascii=False, indent=2))


def cmd_show_state(config: AppConfig, args: argparse.Namespace) -> None:
    storage = Storage(config.workspace.database_path)
    print(json.dumps(storage.get_latest_state(args.story_id).model_dump(), ensure_ascii=False, indent=2))


def cmd_bible(config: AppConfig, args: argparse.Namespace) -> None:
    storage = Storage(config.workspace.database_path)
    if args.input_file:
        payload = BibleContent.model_validate(load_structured_file(args.input_file))
        saved = storage.save_bible(args.story_id, payload)
        print(json.dumps(saved.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(storage.get_bible(args.story_id).model_dump(), ensure_ascii=False, indent=2))


def cmd_outline(config: AppConfig, args: argparse.Namespace) -> None:
    with orchestrator_context(config) as orchestrator:
        cards = orchestrator.generate_outline(args.story_id, OutlineGenerateRequest(scene_count=args.scene_count))
    print(json.dumps([card.model_dump() for card in cards], ensure_ascii=False, indent=2))


def cmd_run_scene(config: AppConfig, args: argparse.Namespace) -> None:
    payload = SceneRequest.model_validate(load_structured_file(args.input_file))

    def log(message: str, progress: float) -> None:
        print(f"[{progress:>4.0%}] {message}")

    with orchestrator_context(config) as orchestrator:
        result = orchestrator.run_scene(args.story_id, payload, log=log)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    if args.print_text:
        print("\n=== ACCEPTED SCENE ===\n")
        print(result.accepted_scene.accepted_text)


def cmd_export(config: AppConfig, args: argparse.Namespace) -> None:
    with orchestrator_context(config) as orchestrator:
        bundle = orchestrator.export_story_markdown(args.story_id)
    out_dir = Path(config.workspace.exports_dir) / args.story_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / bundle["filename"]
    out_path.write_text(bundle["content"], encoding="utf-8")
    print(str(out_path))


def cmd_evaluate(config: AppConfig, args: argparse.Namespace) -> None:
    with orchestrator_context(config) as orchestrator:
        report = orchestrator.evaluate_story(args.story_id)
    out_dir = Path(config.workspace.exports_dir) / args.story_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.story_id}_evaluation.json"
    out_path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    print(f"Saved to {out_path}")


def cmd_serve(config: AppConfig, args: argparse.Namespace) -> None:
    app = create_app(config)
    uvicorn.run(app, host=args.host or config.server.host, port=args.port or config.server.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoNarrative Studio CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Initialize workspace")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("create-story", help="Create a story")
    p.add_argument("--input-file", type=str)
    p.add_argument("--title", type=str, default="Untitled Story")
    p.add_argument("--genre", type=str, default="literary fiction")
    p.add_argument("--premise", type=str, default="A quiet fracture becomes impossible to ignore.")
    p.add_argument("--tone", type=str, default="lyrical and emotionally grounded")
    p.add_argument("--themes", nargs="*", default=[])
    p.add_argument("--characters", nargs="*", default=[])
    p.add_argument("--forbidden", nargs="*", default=[])
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--target-length-scenes", dest="target_length_scenes", type=int, default=12)
    p.set_defaults(func=cmd_create_story)

    p = sub.add_parser("update-story", help="Update a story from YAML/JSON")
    p.add_argument("--story-id", required=True)
    p.add_argument("--input-file", required=True)
    p.set_defaults(func=cmd_update_story)

    p = sub.add_parser("bible", help="Show or replace bible")
    p.add_argument("--story-id", required=True)
    p.add_argument("--input-file")
    p.set_defaults(func=cmd_bible)

    p = sub.add_parser("outline", help="Generate outline cards")
    p.add_argument("--story-id", required=True)
    p.add_argument("--scene-count", type=int, default=6)
    p.set_defaults(func=cmd_outline)

    p = sub.add_parser("run-scene", help="Generate a scene synchronously")
    p.add_argument("--story-id", required=True)
    p.add_argument("--input-file", required=True)
    p.add_argument("--print-text", action="store_true")
    p.set_defaults(func=cmd_run_scene)

    p = sub.add_parser("show-state", help="Print latest state snapshot")
    p.add_argument("--story-id", required=True)
    p.set_defaults(func=cmd_show_state)

    p = sub.add_parser("export", help="Export story as markdown")
    p.add_argument("--story-id", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("evaluate", help="Evaluate story and save JSON report")
    p.add_argument("--story-id", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="Run web UI and API")
    p.add_argument("--host", type=str)
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    args.func(config, args)


if __name__ == "__main__":
    main()
