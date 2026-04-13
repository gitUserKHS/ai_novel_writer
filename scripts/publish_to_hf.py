#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish or pull CoNarrative artifacts to or from Hugging Face Hub.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Upload a local folder to Hugging Face Hub.")
    publish.add_argument("--source-dir", required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    publish.add_argument("--path-in-repo", default="")
    publish.add_argument("--revision", default=None)
    publish.add_argument("--commit-message", default=None)
    publish.add_argument("--private", action="store_true")
    publish.add_argument("--exclude-checkpoints", action="store_true")
    publish.add_argument("--ignore-pattern", action="append", default=[])

    pull = subparsers.add_parser("pull", help="Download a repo snapshot from Hugging Face Hub.")
    pull.add_argument("--repo-id", required=True)
    pull.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    pull.add_argument("--local-dir", required=True)
    pull.add_argument("--revision", default=None)
    pull.add_argument("--allow-pattern", action="append", default=[])
    pull.add_argument("--ignore-pattern", action="append", default=[])

    return parser.parse_args()


def require_hf_hub() -> tuple[Any, Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, ModelCard, ModelCardData, snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            f"Missing Hugging Face Hub dependency: {type(exc).__name__}: {exc}. Install with: pip install huggingface_hub"
        ) from exc
    return HfApi, ModelCard, ModelCardData, snapshot_download


def repo_type_arg(repo_type: str) -> str | None:
    return None if repo_type == "model" else repo_type


def default_ignore_patterns(exclude_checkpoints: bool) -> list[str]:
    patterns: list[str] = []
    if exclude_checkpoints:
        patterns.extend(
            [
                "checkpoint-*",
                "checkpoint-*/*",
                "**/optimizer.pt",
                "**/scheduler.pt",
                "**/rng_state.pth",
                "**/trainer_state.json",
                "**/training_args.bin",
            ]
        )
    return patterns


def infer_base_model(source_dir: Path) -> str | None:
    adapter_config = source_dir / "adapter_config.json"
    if not adapter_config.exists():
        return None
    try:
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ["base_model_name_or_path", "base_model_name"]:
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value
    return None


def infer_tags(source_dir: Path, repo_type: str) -> list[str]:
    tags = ["conarrative"]
    if repo_type == "dataset":
        tags.extend(["dataset", "creative-writing", "story-generation"])
        return tags
    if (source_dir / "adapter_config.json").exists():
        tags.extend(["peft", "lora", "text-generation"])
    else:
        tags.append("text-generation")
    return tags


def build_card(repo_id: str, repo_type: str, source_dir: Path) -> str:
    _, ModelCard, ModelCardData, _ = require_hf_hub()
    source_name = source_dir.name
    base_model = infer_base_model(source_dir)
    tags = infer_tags(source_dir, repo_type)
    if repo_type == "dataset":
        data = ModelCardData(
            language=["ko"],
            tags=tags,
            library_name="datasets",
        )
        body = f"""# {repo_id}

CoNarrative dataset snapshot for collaborative fiction experiments.

## Contents

- source directory: `{source_name}`
- intended use: SFT / DPO / distillation / critic training for scene-level narrative generation

## Notes

- Generated from the CoNarrative pipeline.
- Review licensing and privacy before publishing any prose or preference data publicly.
"""
    else:
        data = ModelCardData(
            language=["ko"],
            license="other",
            library_name="peft",
            tags=tags,
            base_model=base_model,
        )
        body = f"""# {repo_id}

CoNarrative adapter for scene-level fiction generation or evaluation.

## Artifact

- source directory: `{source_name}`
- adapter type: `LoRA` or PEFT-compatible fine-tune
- base model: `{base_model or "unknown"}`

## Intended use

- writer, critic, or world-model role specialization inside the CoNarrative pipeline
- local or hub-hosted collaborative experimentation

## Notes

- Load this adapter on top of the base model listed above.
- Prefer sharing adapters instead of merged full weights for collaboration.
- Review the base model license before redistribution or commercial use.
"""
    return ModelCard.from_template(card_data=data, model_description=body).content


def upload_generated_card(api: Any, repo_id: str, repo_type: str, source_dir: Path, revision: str | None) -> str:
    readme_path = source_dir / "README.md"
    if readme_path.exists():
        return "existing README.md kept"
    content = build_card(repo_id, repo_type, source_dir)
    commit_info = api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type_arg(repo_type),
        revision=revision,
        commit_message=f"Add autogenerated model card for {repo_id}",
    )
    return str(getattr(commit_info, "commit_url", "") or "generated README.md uploaded")


def publish(args: argparse.Namespace) -> None:
    HfApi, _, _, _ = require_hf_hub()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_type = repo_type_arg(args.repo_type)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=repo_type,
        private=bool(args.private),
        exist_ok=True,
    )
    ignore_patterns = default_ignore_patterns(bool(args.exclude_checkpoints)) + list(args.ignore_pattern or [])
    commit_message = args.commit_message or f"Upload {source_dir.name} from CoNarrative"
    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=repo_type,
        folder_path=str(source_dir),
        path_in_repo=args.path_in_repo or None,
        revision=args.revision,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns or None,
    )
    card_result = upload_generated_card(api, args.repo_id, args.repo_type, source_dir, args.revision)
    output = {
        "ok": True,
        "action": "publish",
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "source_dir": str(source_dir),
        "path_in_repo": args.path_in_repo or "",
        "private": bool(args.private),
        "revision": args.revision,
        "exclude_checkpoints": bool(args.exclude_checkpoints),
        "ignored_patterns": ignore_patterns,
        "commit_message": commit_message,
        "commit_url": str(commit_info.commit_url) if getattr(commit_info, "commit_url", None) else "",
        "oid": str(getattr(commit_info, "oid", "") or ""),
        "card_result": card_result,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def pull(args: argparse.Namespace) -> None:
    _, _, _, snapshot_download = require_hf_hub()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    repo_type = repo_type_arg(args.repo_type)
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        revision=args.revision,
        allow_patterns=list(args.allow_pattern or []) or None,
        ignore_patterns=list(args.ignore_pattern or []) or None,
    )
    output = {
        "ok": True,
        "action": "pull",
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "local_dir": str(local_dir),
        "downloaded_path": str(downloaded_path),
        "revision": args.revision,
        "allow_patterns": list(args.allow_pattern or []),
        "ignore_patterns": list(args.ignore_pattern or []),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "publish":
        publish(args)
        return
    if args.command == "pull":
        pull(args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
