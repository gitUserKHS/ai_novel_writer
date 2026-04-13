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

from conarrative.ui_presets import UIPresetStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull shared HF adapters and assemble a collaborator-ready runtime preset.")
    parser.add_argument("--writer-repo-id", default="")
    parser.add_argument("--critic-repo-id", default="")
    parser.add_argument("--world-repo-id", default="")
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-root", default="outputs/hf_download")
    parser.add_argument("--package-dir", default="outputs/hf_onboarding")
    parser.add_argument("--preset-name", default="hf-collab-runtime")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--critic-temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--cache-responses", action="store_true")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--save-ui-preset", action="store_true")
    parser.add_argument("--ui-presets-path", default="")
    return parser.parse_args()


def require_hf_hub() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            f"Missing Hugging Face Hub dependency: {type(exc).__name__}: {exc}. Install with: pip install huggingface_hub"
        ) from exc
    return snapshot_download


def repo_type_arg(repo_type: str) -> str | None:
    return None if repo_type == "model" else repo_type


def repo_leaf(repo_id: str) -> str:
    return str(repo_id).strip().split("/")[-1].strip() or "artifact"


def download_repo(
    *,
    snapshot_download: Any,
    repo_id: str,
    repo_type: str,
    role: str,
    download_root: Path,
    revision: str | None,
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> dict[str, str]:
    local_dir = (download_root / role / repo_leaf(repo_id)).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type_arg(repo_type),
        local_dir=str(local_dir),
        revision=revision,
        allow_patterns=allow_patterns or None,
        ignore_patterns=ignore_patterns or None,
    )
    return {
        "role": role,
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "downloaded_path": str(downloaded_path),
    }


def build_runtime_payload(args: argparse.Namespace, role_paths: dict[str, str]) -> dict[str, Any]:
    shared_model = str(args.model).strip()
    role_models = {
        "planner": shared_model,
        "writer": role_paths.get("writer", shared_model),
        "consistency_critic": role_paths.get("consistency_critic", shared_model),
        "creativity_critic": shared_model,
        "world_model": role_paths.get("world_model", shared_model),
        "revision": shared_model,
        "extractor": shared_model,
    }
    return {
        "provider": args.provider,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": shared_model,
        "temperature": args.temperature,
        "critic_temperature": args.critic_temperature,
        "max_tokens": args.max_tokens,
        "cache_responses": bool(args.cache_responses),
        "role_models": role_models,
    }


def write_package(package_root: Path, preset_name: str, runtime_payload: dict[str, Any], downloads: list[dict[str, str]]) -> dict[str, str]:
    package_root.mkdir(parents=True, exist_ok=True)
    runtime_preset_path = package_root / "runtime_preset.json"
    manifest_path = package_root / "hf_onboarding_manifest.json"
    ui_record_path = package_root / "ui_runtime_preset_record.json"
    runtime_preset_path.write_text(json.dumps(runtime_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "ok": True,
                "preset_name": preset_name,
                "downloads": downloads,
                "runtime_preset_path": str(runtime_preset_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ui_record_path.write_text(
        json.dumps(
            {
                "kind": "runtime",
                "name": preset_name,
                "payload": runtime_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "runtime_preset_path": str(runtime_preset_path),
        "manifest_path": str(manifest_path),
        "ui_record_path": str(ui_record_path),
    }


def main() -> None:
    args = parse_args()
    repo_map = {
        "writer": str(args.writer_repo_id or "").strip(),
        "consistency_critic": str(args.critic_repo_id or "").strip(),
        "world_model": str(args.world_repo_id or "").strip(),
    }
    if not any(repo_map.values()):
        raise SystemExit("At least one of --writer-repo-id, --critic-repo-id, or --world-repo-id is required.")

    snapshot_download = require_hf_hub()
    download_root = Path(args.download_root).resolve()
    package_root = (Path(args.package_dir).resolve() / str(args.preset_name).strip())
    allow_patterns = [pattern for pattern in args.allow_pattern if pattern]
    ignore_patterns = [pattern for pattern in args.ignore_pattern if pattern]

    downloads: list[dict[str, str]] = []
    role_paths: dict[str, str] = {}
    for role, repo_id in repo_map.items():
        if not repo_id:
            continue
        download = download_repo(
            snapshot_download=snapshot_download,
            repo_id=repo_id,
            repo_type=args.repo_type,
            role=role,
            download_root=download_root,
            revision=args.revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        downloads.append(download)
        role_paths[role] = download["local_dir"]

    runtime_payload = build_runtime_payload(args, role_paths)
    package_paths = write_package(package_root, str(args.preset_name).strip(), runtime_payload, downloads)

    saved_ui_preset = False
    if args.save_ui_preset:
        ui_presets_path = str(args.ui_presets_path or "").strip()
        if not ui_presets_path:
            raise SystemExit("--ui-presets-path is required when --save-ui-preset is set.")
        UIPresetStore(Path(ui_presets_path)).save("runtime", str(args.preset_name).strip(), runtime_payload)
        saved_ui_preset = True

    print(
        json.dumps(
            {
                "ok": True,
                "action": "hf_onboard",
                "preset_name": str(args.preset_name).strip(),
                "downloads": downloads,
                "runtime_payload": runtime_payload,
                "saved_ui_preset": saved_ui_preset,
                "download_root": str(download_root),
                "package_dir": str(package_root),
                **package_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
