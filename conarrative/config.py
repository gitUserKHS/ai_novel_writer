
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field

from .models import RuntimeSettings
from .utils import ensure_dir


class WorkspaceConfig(BaseModel):
    root: str = "workspace"
    database_path: str = "workspace/conarrative.db"
    exports_dir: str = "workspace/exports"
    runtime_settings_path: str = "workspace/runtime_settings.json"


class BackendConfig(RuntimeSettings):
    pass


class OrchestrationConfig(BaseModel):
    recent_scene_memory: int = 3
    candidate_count: int = 3
    scene_min_words: int = 320
    scene_max_words: int = 680
    adaptive_outline: bool = True
    auto_revision: bool = True
    minimum_release_consistency: float = 0.72
    consistency_weight: float = 0.58
    creativity_weight: float = 0.27
    world_model_weight: float = 0.15
    max_summary_memory: int = 6
    release_gate_world_min_plausibility: float = 0.68
    release_gate_max_medium_issues: int = 1
    release_gate_rescue_rounds: int = 1
    release_gate_rescue_candidate_count: int = 2
    strict_release_gate: bool = False


class AppConfig(BaseModel):
    app_name: str = "CoNarrative AutoNovel Studio"
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)


def _resolve_path(base: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw: Dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    app_config = AppConfig(**raw)
    base_dir = config_path.parent

    workspace = app_config.workspace.model_copy(
        update={
            "root": _resolve_path(base_dir, app_config.workspace.root),
            "database_path": _resolve_path(base_dir, app_config.workspace.database_path),
            "exports_dir": _resolve_path(base_dir, app_config.workspace.exports_dir),
            "runtime_settings_path": _resolve_path(base_dir, app_config.workspace.runtime_settings_path),
        }
    )
    backend = app_config.backend.model_copy(
        update={
            "cache_dir": _resolve_path(base_dir, app_config.backend.cache_dir),
        }
    )
    resolved = app_config.model_copy(update={"workspace": workspace, "backend": backend})
    ensure_dir(Path(resolved.workspace.root))
    ensure_dir(Path(resolved.workspace.database_path).parent)
    ensure_dir(Path(resolved.workspace.exports_dir))
    ensure_dir(Path(resolved.backend.cache_dir))
    return resolved
