from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import httpx

from .config import AppConfig
from .models import ProviderType, RuntimeSettings, TrainedAdapterOut
from .train_runtime import active_adapter_dir, active_adapter_metadata_path, ensure_training_environment, training_python_path, training_root


def list_trained_adapters(config: AppConfig, story_id: str = "") -> list[TrainedAdapterOut]:
    runs_root = training_root(config) / "runs"

    adapters: list[TrainedAdapterOut] = []
    adapters.extend(_list_active_adapters(config, story_id=story_id))
    if not runs_root.exists():
        return adapters

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
                is_active=False,
            )
        )
    adapters.sort(key=lambda item: (item.is_active, item.created_at), reverse=True)
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
    force_restart: bool = False,
) -> tuple[RuntimeSettings, str]:
    ensure_training_environment(config)
    env_python = training_python_path(config)
    if not env_python.exists():
        raise RuntimeError("Training Python environment is missing. Prepare the training environment first.")

    model_id = _model_id(adapter)
    key = f"{host}:{port}:{adapter.adapter_dir}"
    existing = registry.get(key)
    if existing is not None and existing.poll() is None and not force_restart:
        try:
            _wait_for_openai_server(host=host, port=port, timeout_seconds=180, expected_model_id=model_id, process=existing)
            return _settings_for_server(current_settings, host, port, adapter), ""
        except Exception:
            _terminate_process(existing)
            registry.pop(key, None)
    if force_restart:
        _stop_servers_on_port(registry, host=host, port=port)

    log_dir = training_root(config) / "servers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_id.replace('/', '_').replace(':', '_')}.log"
    log_handle, resolved_log_path = _open_process_log(log_path)
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
    try:
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()
    registry[key] = process
    try:
        _wait_for_openai_server(host=host, port=port, timeout_seconds=180, expected_model_id=model_id, process=process)
        time.sleep(1.0)
        _wait_for_openai_server(host=host, port=port, timeout_seconds=10, expected_model_id=model_id, process=process)
    except Exception as exc:
        registry.pop(key, None)
        if process.poll() is None:
            _terminate_process(process)
        log_tail = _tail_text(Path(resolved_log_path)) if resolved_log_path else ""
        detail = f" Trained model server log tail: {log_tail}" if log_tail else ""
        log_hint = f" Check log: {resolved_log_path}." if resolved_log_path else ""
        raise RuntimeError(f"Trained model server did not stay ready.{log_hint}{detail}") from exc
    return _settings_for_server(current_settings, host, port, adapter), resolved_log_path


def stop_trained_servers(registry: dict[str, subprocess.Popen]) -> None:
    for process in list(registry.values()):
        if process.poll() is None:
            process.terminate()
    registry.clear()


def _stop_servers_on_port(registry: dict[str, subprocess.Popen], *, host: str, port: int) -> None:
    prefix = f"{host}:{port}:"
    for key, process in list(registry.items()):
        if not key.startswith(prefix):
            continue
        _terminate_process(process)
        registry.pop(key, None)


def _list_active_adapters(config: AppConfig, story_id: str = "") -> list[TrainedAdapterOut]:
    roots: list[tuple[str, Path, Path]] = []
    if story_id:
        roots.append((story_id, active_adapter_dir(config, story_id), active_adapter_metadata_path(config, story_id)))
    else:
        active_root = training_root(config) / "active_adapters"
        if active_root.exists():
            for path in active_root.glob("*/run_metadata.json"):
                roots.append((path.parent.name, path.parent / "final_adapter", path))

    adapters: list[TrainedAdapterOut] = []
    for resolved_story_id, adapter_dir, metadata_path in roots:
        if not adapter_dir.exists() or not metadata_path.exists():
            continue
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception:
            raw = {}
        adapters.append(
            TrainedAdapterOut(
                story_id=resolved_story_id,
                run_id="active",
                base_model=str(raw.get("base_model") or ""),
                adapter_dir=str(adapter_dir),
                metadata_path=str(metadata_path),
                created_at=_mtime_iso(metadata_path),
                training_profile=str(raw.get("training_profile") or ""),
                is_active=True,
            )
        )
    return adapters


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
    suffix = _ascii_slug(adapter.story_id or "story")
    if adapter.is_active:
        return f"conarrative-active-{suffix}"
    run = _ascii_slug(adapter.run_id[:19] if adapter.run_id else "latest")
    return f"conarrative-trained-{suffix}-{run}"


def _wait_for_openai_server(
    *,
    host: str,
    port: int,
    timeout_seconds: int,
    expected_model_id: str = "",
    process: subprocess.Popen | None = None,
) -> None:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Trained model server exited with code {process.returncode}. Last error: {last_error}")
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code < 500:
                if expected_model_id:
                    payload = response.json()
                    model_ids = {
                        str(item.get("id") or "")
                        for item in payload.get("data", [])
                        if isinstance(item, dict)
                    }
                    if expected_model_id in model_ids:
                        return
                    last_error = f"server responded but did not expose model {expected_model_id}; models={sorted(model_ids)}"
                    time.sleep(1.0)
                    continue
                else:
                    return
            last_error = response.text[:300]
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for trained model server at {url}. Last error: {last_error}")


def _ascii_slug(value: str, limit: int = 48) -> str:
    raw = (value or "").strip() or "story"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._").lower()
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    if not cleaned:
        return digest
    cleaned = cleaned[:limit].strip("-._")
    return f"{cleaned}-{digest}" if cleaned else digest


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except Exception:
        process.kill()


def _tail_text(path: Path, max_chars: int = 1400) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _open_process_log(preferred_path: Path) -> tuple[TextIO, str]:
    try:
        return preferred_path.open("a", encoding="utf-8"), str(preferred_path)
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "conarrative_trained_servers"
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / preferred_path.name
            return fallback_path.open("a", encoding="utf-8"), str(fallback_path)
        except OSError:
            return open(os.devnull, "w", encoding="utf-8"), ""


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
