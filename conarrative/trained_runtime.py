from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import AppConfig
from .models import ProviderType, RuntimeSettings, TrainedAdapterOut
from .train_runtime import ensure_training_environment, training_python_path, training_root


def list_trained_adapters(config: AppConfig, story_id: str = "") -> list[TrainedAdapterOut]:
    runs_root = training_root(config) / "runs"
    if not runs_root.exists():
        return []

    adapters: list[TrainedAdapterOut] = []
    metadata_paths = runs_root.glob(f"{story_id}/*/run_metadata.json") if story_id else runs_root.glob("*/*/run_metadata.json")
    for metadata_path in metadata_paths:
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        run_dir = metadata_path.parent
        adapter_dir = Path(raw.get("final_adapter_dir") or run_dir / "final_adapter")
        if not adapter_dir.exists():
            continue
        adapters.append(
            TrainedAdapterOut(
                story_id=run_dir.parent.name,
                run_id=run_dir.name,
                base_model=str(raw.get("base_model") or ""),
                adapter_dir=str(adapter_dir),
                metadata_path=str(metadata_path),
                created_at=_mtime_iso(metadata_path),
                training_profile=str(raw.get("training_profile") or ""),
            )
        )
    adapters.sort(key=lambda item: item.created_at, reverse=True)
    return adapters


def select_trained_adapter(config: AppConfig, *, story_id: str = "", adapter_dir: str = "") -> TrainedAdapterOut:
    adapters = list_trained_adapters(config, story_id=story_id)
    if adapter_dir:
        resolved = str(Path(adapter_dir).resolve())
        for adapter in adapters or list_trained_adapters(config):
            if str(Path(adapter.adapter_dir).resolve()) == resolved:
                return adapter
        raise RuntimeError(f"Trained adapter was not found: {adapter_dir}")
    if not adapters:
        raise RuntimeError("No trained adapter was found. Run one-click training first.")
    return adapters[0]


def start_trained_adapter_server(
    *,
    config: AppConfig,
    adapter: TrainedAdapterOut,
    current_settings: RuntimeSettings,
    registry: dict[str, subprocess.Popen],
    host: str = "127.0.0.1",
    port: int = 5001,
) -> tuple[RuntimeSettings, str]:
    ensure_training_environment(config)
    env_python = training_python_path(config)
    if not env_python.exists():
        raise RuntimeError("Training Python environment is missing. Prepare the training environment first.")

    key = f"{host}:{port}:{adapter.adapter_dir}"
    existing = registry.get(key)
    if existing is not None and existing.poll() is None:
        return _settings_for_server(current_settings, host, port, adapter), ""

    model_id = _model_id(adapter)
    log_dir = training_root(config) / "servers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_id.replace('/', '_').replace(':', '_')}.log"
    command = [
        str(env_python),
        "scripts/serve_trained_adapter.py",
        "--base-model",
        adapter.base_model,
        "--adapter-dir",
        adapter.adapter_dir,
        "--model-id",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
    ]
    env = os.environ.copy()
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    registry[key] = process
    try:
        _wait_for_openai_server(host=host, port=port, timeout_seconds=180)
    except Exception:
        if process.poll() is not None:
            raise RuntimeError(f"Trained model server exited early. Check log: {log_path}")
        raise
    return _settings_for_server(current_settings, host, port, adapter), str(log_path)


def stop_trained_servers(registry: dict[str, subprocess.Popen]) -> None:
    for process in list(registry.values()):
        if process.poll() is None:
            process.terminate()
    registry.clear()


def _settings_for_server(current: RuntimeSettings, host: str, port: int, adapter: TrainedAdapterOut) -> RuntimeSettings:
    return RuntimeSettings.model_validate(
        {
            **current.model_dump(),
            "provider": ProviderType.OPENAI_COMPATIBLE,
            "base_url": f"http://{host}:{port}/v1",
            "model": _model_id(adapter),
            "api_key": "not-needed",
        }
    )


def _model_id(adapter: TrainedAdapterOut) -> str:
    suffix = adapter.story_id or "story"
    run = adapter.run_id[:19] if adapter.run_id else "latest"
    return f"conarrative-trained-{suffix}-{run}"


def _wait_for_openai_server(*, host: str, port: int, timeout_seconds: int) -> None:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code < 500:
                return
            last_error = response.text[:300]
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for trained model server at {url}. Last error: {last_error}")


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
