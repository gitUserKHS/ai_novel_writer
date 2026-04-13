#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conarrative.hf_release import infer_base_model_slug, next_release_tag, suggest_repo_id
from conarrative.training_metadata import load_training_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish or pull CoNarrative artifacts to or from Hugging Face Hub.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="Upload a local folder to Hugging Face Hub.")
    publish.add_argument("--source-dir", required=True)
    publish.add_argument("--repo-id", required=False)
    publish.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    publish.add_argument("--path-in-repo", default="")
    publish.add_argument("--revision", default=None)
    publish.add_argument("--commit-message", default=None)
    publish.add_argument("--private", action="store_true")
    publish.add_argument("--exclude-checkpoints", action="store_true")
    publish.add_argument("--ignore-pattern", action="append", default=[])
    publish.add_argument("--namespace", default="")
    publish.add_argument("--project", default="conarrative")
    publish.add_argument("--role", default="")
    publish.add_argument("--base-model", default="")
    publish.add_argument("--stage", default="")
    publish.add_argument("--auto-tag", action="store_true")
    publish.add_argument("--release-tag", default=None)
    publish.add_argument("--release-prefix", default="v")
    publish.add_argument("--bump", choices=["patch", "minor", "major"], default="patch")
    publish.add_argument("--tag-message", default=None)

    pull = subparsers.add_parser("pull", help="Download a repo snapshot from Hugging Face Hub.")
    pull.add_argument("--repo-id", required=True)
    pull.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    pull.add_argument("--local-dir", required=True)
    pull.add_argument("--revision", default=None)
    pull.add_argument("--allow-pattern", action="append", default=[])
    pull.add_argument("--ignore-pattern", action="append", default=[])

    suggest = subparsers.add_parser("suggest", help="Suggest a standardized Hugging Face repo id and release tag.")
    suggest.add_argument("--namespace", required=True)
    suggest.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    suggest.add_argument("--project", default="conarrative")
    suggest.add_argument("--role", default="")
    suggest.add_argument("--base-model", default="")
    suggest.add_argument("--stage", default="")
    suggest.add_argument("--release-prefix", default="v")

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


def format_metadata_lines(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    trainer_state = metadata.get("trainer_state", {}) or {}
    train_metrics = metadata.get("train_metrics", {}) or {}
    eval_metrics = metadata.get("eval_metrics", {}) or {}
    extra = metadata.get("extra", {}) or {}
    lines = [
        "## Training Summary",
        "",
        f"- mode: `{metadata.get('mode', '') or 'unknown'}`",
        f"- source model: `{metadata.get('model_name_or_path', '') or 'unknown'}`",
        f"- dataset format: `{metadata.get('dataset_format', '') or 'auto'}`",
        f"- train file: `{metadata.get('train_file', '') or 'unknown'}`",
        f"- eval file: `{metadata.get('eval_file', '') or 'not provided'}`",
        f"- train examples: `{metadata.get('train_examples', 0)}`",
        f"- eval examples: `{metadata.get('eval_examples', 0)}`",
    ]
    if extra:
        lines.extend(
            [
                f"- learning rate: `{extra.get('learning_rate', 'unknown')}`",
                f"- max seq length: `{extra.get('max_seq_length', 'unknown')}`",
                f"- epochs: `{extra.get('num_train_epochs', 'unknown')}`",
            ]
        )
    if trainer_state:
        lines.extend(
            [
                f"- best metric: `{trainer_state.get('best_metric', 'n/a')}`",
                f"- best checkpoint: `{trainer_state.get('best_model_checkpoint', 'n/a')}`",
            ]
        )
    metric_rows: list[str] = []
    for key, value in train_metrics.items():
        metric_rows.append(f"| train `{key}` | `{value}` |")
    for key, value in eval_metrics.items():
        metric_rows.append(f"| eval `{key}` | `{value}` |")
    if metric_rows:
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                "| metric | value |",
                "| --- | --- |",
                *metric_rows,
            ]
        )
    lines.extend(
        [
            "",
            "## Artifact Lineage",
            "",
            f"- generated by: `{metadata.get('generated_by', 'scripts/train_qlora.py')}`",
            f"- generated at: `{metadata.get('generated_at', '') or 'unknown'}`",
            f"- source artifact dir: `{metadata.get('output_dir', '') or 'unknown'}`",
        ]
    )
    return "\n".join(lines)


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
    metadata = load_training_metadata(source_dir)
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
        if metadata:
            body = f"{body.rstrip()}\n\n{format_metadata_lines(metadata)}\n"
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
        if metadata:
            body = f"{body.rstrip()}\n\n{format_metadata_lines(metadata)}\n"
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


def resolve_repo_id(args: argparse.Namespace, source_dir: Path) -> str:
    repo_id = str(args.repo_id or "").strip()
    if repo_id:
        return repo_id
    namespace = str(args.namespace or "").strip()
    if not namespace:
        raise SystemExit("Either --repo-id or --namespace is required.")
    base_model = str(args.base_model or "").strip() or (infer_base_model(source_dir) or "")
    return suggest_repo_id(
        namespace,
        repo_type=args.repo_type,
        project=args.project,
        role=args.role,
        base_model=base_model,
        stage=args.stage,
    )


def maybe_create_release_tag(api: Any, args: argparse.Namespace, repo_id: str) -> dict[str, str]:
    explicit_tag = str(args.release_tag or "").strip()
    should_tag = bool(args.auto_tag or explicit_tag)
    if not should_tag:
        return {"release_tag": "", "tag_result": "skipped"}
    refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type_arg(args.repo_type))
    existing_tags = [str(getattr(tag, "name", "") or "") for tag in getattr(refs, "tags", [])]
    release_tag = explicit_tag or next_release_tag(existing_tags, prefix=args.release_prefix, bump=args.bump)
    tag_message = args.tag_message or f"Release {release_tag} for {repo_id}"
    api.create_tag(
        repo_id=repo_id,
        repo_type=repo_type_arg(args.repo_type),
        revision=args.revision or "main",
        tag=release_tag,
        tag_message=tag_message,
    )
    return {"release_tag": release_tag, "tag_result": "created"}


def publish(args: argparse.Namespace) -> None:
    HfApi, _, _, _ = require_hf_hub()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    repo_id = resolve_repo_id(args, source_dir)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_type = repo_type_arg(args.repo_type)
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=bool(args.private),
        exist_ok=True,
    )
    ignore_patterns = default_ignore_patterns(bool(args.exclude_checkpoints)) + list(args.ignore_pattern or [])
    commit_message = args.commit_message or f"Upload {source_dir.name} from CoNarrative"
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(source_dir),
        path_in_repo=args.path_in_repo or None,
        revision=args.revision,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns or None,
    )
    card_result = upload_generated_card(api, repo_id, args.repo_type, source_dir, args.revision)
    release = maybe_create_release_tag(api, args, repo_id)
    output = {
        "ok": True,
        "action": "publish",
        "repo_id": repo_id,
        "repo_type": args.repo_type,
        "source_dir": str(source_dir),
        "path_in_repo": args.path_in_repo or "",
        "private": bool(args.private),
        "revision": args.revision,
        "project": args.project,
        "role": args.role,
        "base_model": str(args.base_model or "") or (infer_base_model(source_dir) or ""),
        "base_model_slug": infer_base_model_slug(str(args.base_model or "") or (infer_base_model(source_dir) or "")),
        "stage": args.stage,
        "exclude_checkpoints": bool(args.exclude_checkpoints),
        "ignored_patterns": ignore_patterns,
        "commit_message": commit_message,
        "commit_url": str(commit_info.commit_url) if getattr(commit_info, "commit_url", None) else "",
        "oid": str(getattr(commit_info, "oid", "") or ""),
        "card_result": card_result,
        **release,
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


def suggest(args: argparse.Namespace) -> None:
    repo_id = suggest_repo_id(
        args.namespace,
        repo_type=args.repo_type,
        project=args.project,
        role=args.role,
        base_model=args.base_model,
        stage=args.stage,
    )
    output = {
        "ok": True,
        "action": "suggest",
        "repo_id": repo_id,
        "repo_type": args.repo_type,
        "project": args.project,
        "role": args.role,
        "base_model": args.base_model,
        "base_model_slug": infer_base_model_slug(args.base_model),
        "stage": args.stage,
        "release_tag": next_release_tag([], prefix=args.release_prefix, bump="patch"),
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
    if args.command == "suggest":
        suggest(args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
