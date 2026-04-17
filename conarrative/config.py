from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field

from .models import RuntimeSettings


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False


class WorkspaceConfig(BaseModel):
    root: str = "./workspace"
    database_path: str = "./workspace/conarrative.db"
    exports_dir: str = "./workspace/exports"
    runtime_settings_path: str = "./workspace/runtime_settings.json"


class OrchestrationConfig(BaseModel):
    recent_scene_memory: int = 3
    candidate_count: int = 3
    auto_revision: bool = True
    max_revision_passes: int = 1
    keep_candidate_text: bool = True
    creativity_weight: float = 0.45
    consistency_weight: float = 0.55


class AppConfig(BaseModel):
    app_name: str = "CoNarrative Studio"
    backend: RuntimeSettings = Field(default_factory=RuntimeSettings)
    server: ServerConfig = Field(default_factory=ServerConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)

    def resolve_paths(self, config_path: Path) -> "AppConfig":
        root = config_path.parent.resolve()
        self.workspace.root = str((root / self.workspace.root).resolve()) if not Path(self.workspace.root).is_absolute() else self.workspace.root
        self.workspace.database_path = str((root / self.workspace.database_path).resolve()) if not Path(self.workspace.database_path).is_absolute() else self.workspace.database_path
        self.workspace.exports_dir = str((root / self.workspace.exports_dir).resolve()) if not Path(self.workspace.exports_dir).is_absolute() else self.workspace.exports_dir
        self.workspace.runtime_settings_path = str((root / self.workspace.runtime_settings_path).resolve()) if not Path(self.workspace.runtime_settings_path).is_absolute() else self.workspace.runtime_settings_path
        return self

    def ensure_directories(self) -> None:
        Path(self.workspace.root).mkdir(parents=True, exist_ok=True)
        Path(self.workspace.exports_dir).mkdir(parents=True, exist_ok=True)
        Path(self.workspace.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.workspace.runtime_settings_path).parent.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        config = AppConfig()
        config.ensure_directories()
        return config
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    config = AppConfig.model_validate(raw)
    config.resolve_paths(p)
    config.ensure_directories()
    return config
