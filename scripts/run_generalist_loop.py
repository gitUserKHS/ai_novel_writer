#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conarrative.corpus import merge_training_manifests
from conarrative.models import StoryCreate
from conarrative.utils import slugify
from scripts.run_pipeline import (
    build_runtime,
    create_story,
    full_generate,
    read_yaml,
    run_command,
    smoke_generate,
    summarize_command_result,
    unload_local_ollama_models,
    uses_local_ollama,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-story CoNarrative generation/training loop for generalist tuning.")
    parser.add_argument("--app-config", required=True, help="Generation app config YAML")
    parser.add_argument("--story-dir", required=True, help="Directory containing story YAML files")
    parser.add_argument("--scene-file", help="Optional explicit scene YAML input for smoke mode")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--scene-limit", type=int, help="Optional scene limit for full generation")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--train-config", action="append", help="Optional training YAML preset. Pass multiple times to chain stages.")
    parser.add_argument("--train-action", choices=["skip", "dry-run", "run"], default="skip")
    parser.add_argument("--train-output-dir", help="Optional override for training output dir")
    parser.add_argument("--corpus-output-dir", default="outputs/generalist_corpus")
    parser.add_argument("--validation-story-ratio", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--story-offset", type=int, default=0)
    parser.add_argument("--story-limit", type=int, help="Optional maximum number of story files to process after offset")
    parser.add_argument("--resume", action="store_true", help="Reuse existing stories in the workspace and skip completed manifests")
    return parser.parse_args()


def discover_story_files(story_dir: str | Path) -> list[Path]:
    paths = sorted(Path(story_dir).glob("*.yaml"))
    if not paths:
        raise SystemExit(f"No story YAML files found in: {story_dir}")
    return paths


def select_story_files(story_files: list[Path], story_offset: int, story_limit: int | None) -> list[Path]:
    if story_offset < 0:
        raise ValueError("story_offset must be >= 0")
    selected = story_files[story_offset:]
    if story_limit is not None and story_limit > 0:
        selected = selected[:story_limit]
    return selected


def canonical_story_id_from_payload(payload: dict[str, Any]) -> str:
    validated = StoryCreate(**payload)
    return validated.id or slugify(validated.title)


def expected_training_manifest_path(exports_dir: str | Path, story_id: str) -> Path:
    return Path(exports_dir) / story_id / "training" / f"{story_id}_training_manifest.json"


def story_resume_state(storage: Any, exports_dir: str | Path, story_file: str | Path) -> dict[str, Any]:
    payload = read_yaml(story_file)
    story_id = canonical_story_id_from_payload(payload)
    story = storage.get_story(story_id)
    manifest_path = expected_training_manifest_path(exports_dir, story_id)
    scene_count = len(storage.list_scenes(story_id)) if story is not None else 0
    return {
        "story_id": story_id,
        "payload": payload,
        "story": story,
        "manifest_path": manifest_path,
        "scene_count": scene_count,
        "completed": story is not None and manifest_path.exists(),
    }


def resume_smoke_generate(orchestrator: Any, story_id: str, scene_file: str | None) -> dict[str, Any]:
    scenes = orchestrator.storage.list_scenes(story_id)
    if not scenes:
        return smoke_generate(orchestrator, story_id, scene_file)

    manuscript = orchestrator.write_export_files(story_id)
    evaluation = orchestrator.write_evaluation_file(story_id)
    training_bundle = orchestrator.write_training_bundle(story_id)
    latest_scene = scenes[-1]
    return {
        "mode": "smoke_resume",
        "accepted_scene_id": latest_scene["id"],
        "accepted_text_preview": latest_scene["accepted_text"][:400],
        "manuscript_path": manuscript["path"],
        "evaluation_path": evaluation["path"],
        "training_bundle_manifest": training_bundle["paths"]["manifest"],
        "training_bundle_paths": training_bundle["paths"],
    }


def corpus_pool_key(train_cfg: dict[str, Any] | str) -> str:
    if isinstance(train_cfg, dict):
        explicit_pool = str(train_cfg.get("pool_key", "") or "").strip()
        if explicit_pool:
            return explicit_pool
        train_mode = str(train_cfg.get("mode", "sft"))
    else:
        train_mode = train_cfg
    if train_mode == "sft":
        return "writer_sft"
    if train_mode == "dpo":
        return "writer_dpo"
    if train_mode == "distill":
        return "distill_stepwise"
    raise ValueError(f"Unsupported train mode: {train_mode}")


def stage_output_dir(train_config_path: str | Path, output_dir: str | None, stage_index: int, stage_count: int) -> str:
    if output_dir:
        base = Path(output_dir)
        if stage_count == 1:
            return str(base)
        return str(base / f"{stage_index:02d}_{Path(train_config_path).stem}")
    return str(Path("outputs") / Path(train_config_path).stem)


def run_training_stage(
    train_config_path: str | Path,
    corpus_paths: dict[str, str],
    corpus_counts: dict[str, dict[str, int]],
    action: str,
    output_dir: str | None,
    stage_index: int,
    stage_count: int,
    model_name_or_path: str | None = None,
) -> dict[str, Any]:
    train_cfg = read_yaml(train_config_path)
    train_mode = str(train_cfg.get("mode", "sft"))
    pool_key = corpus_pool_key(train_cfg)
    train_file = corpus_paths[f"{pool_key}_train"]
    eval_file = corpus_paths[f"{pool_key}_eval"] if corpus_counts.get(pool_key, {}).get("eval", 0) > 0 else None
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
    if eval_file:
        cmd += ["--eval-file", eval_file]
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
        "pool_key": pool_key,
        "chain_from_previous": bool(train_cfg.get("chain_from_previous", True)),
        "action": action,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "train_file": train_file,
        "eval_file": eval_file,
        "output_dir": resolved_output_dir,
        "model_name_or_path": model_name_or_path or str(train_cfg.get("model_name_or_path", "")),
    }


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {"ok": True, "steps": {}}

    if args.run_tests:
        test_result = run_command([sys.executable, "-m", "pytest", "-q"])
        summary["steps"]["tests"] = summarize_command_result(test_result)
        if test_result.returncode != 0:
            summary["ok"] = False
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            raise SystemExit(test_result.returncode)

    config, storage, orchestrator = build_runtime(args.app_config)
    health_ok, health_message = orchestrator.provider.health()
    summary["steps"]["health"] = {"ok": health_ok, "message": health_message}
    if not health_ok:
        summary["ok"] = False
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    discovered_story_files = discover_story_files(args.story_dir)
    selected_story_files = select_story_files(discovered_story_files, args.story_offset, args.story_limit)
    summary["steps"]["story_selection"] = {
        "story_dir": str(args.story_dir),
        "discovered_count": len(discovered_story_files),
        "selected_count": len(selected_story_files),
        "story_offset": args.story_offset,
        "story_limit": args.story_limit,
        "resume": bool(args.resume),
    }
    if not selected_story_files:
        summary["ok"] = False
        summary["steps"]["story_selection"]["error"] = "No story files selected after applying offset and limit."
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    story_results: list[dict[str, Any]] = []
    manifest_paths: list[str] = []
    for story_file in selected_story_files:
        resume_state = story_resume_state(storage, config.workspace.exports_dir, story_file)
        if args.resume and resume_state["completed"]:
            generation = {
                "mode": "resume_skip",
                "training_bundle_manifest": str(resume_state["manifest_path"]),
            }
            story_results.append(
                {
                    "story_file": str(story_file),
                    "story_id": resume_state["story_id"],
                    "title": resume_state["story"].title,
                    "status": "skipped_completed",
                    "scene_count": resume_state["scene_count"],
                    "generation": generation,
                }
            )
            manifest_paths.append(str(resume_state["manifest_path"]))
            continue

        if args.resume and resume_state["story"] is not None:
            story = resume_state["story"]
            status = "resumed_existing"
        else:
            story = create_story(storage, story_file, None)
            status = "created_new"

        if args.mode == "smoke":
            if status == "resumed_existing":
                generation = resume_smoke_generate(orchestrator, story.id, args.scene_file)
            else:
                generation = smoke_generate(orchestrator, story.id, args.scene_file)
        else:
            generation = full_generate(orchestrator, story.id, args.scene_limit)
        story_results.append(
            {
                "story_file": str(story_file),
                "story_id": story.id,
                "title": story.title,
                "status": status,
                "generation": generation,
            }
        )
        manifest_paths.append(generation["training_bundle_manifest"])
    summary["steps"]["stories"] = story_results

    train_configs = args.train_config or []
    if train_configs and args.train_action != "skip":
        if args.train_action == "run" and uses_local_ollama(config.backend):
            unload_result = unload_local_ollama_models(config.backend)
            summary["steps"]["ollama_unload"] = unload_result
            if not unload_result["ok"]:
                summary["ok"] = False
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                raise SystemExit(1)

        corpus_result = merge_training_manifests(
            manifest_paths=manifest_paths,
            output_dir=args.corpus_output_dir,
            validation_story_ratio=args.validation_story_ratio,
            seed=args.seed,
        )
        summary["steps"]["corpus"] = corpus_result

        training_results: list[dict[str, Any]] = []
        previous_output_dir: str | None = None
        for stage_index, train_config in enumerate(train_configs, start=1):
            stage_cfg = read_yaml(train_config)
            chain_from_previous = bool(stage_cfg.get("chain_from_previous", True))
            train_result = run_training_stage(
                train_config,
                corpus_result["paths"],
                corpus_result["counts"],
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
