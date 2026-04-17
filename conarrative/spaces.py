from __future__ import annotations

import os
from pathlib import Path

from .config import AppConfig, OrchestrationConfig, ServerConfig, WorkspaceConfig
from .models import ProviderType, RuntimeSettings


def _provider_from_env(value: str | None) -> ProviderType:
    if not value:
        return ProviderType.MOCK
    try:
        return ProviderType(value)
    except ValueError:
        return ProviderType.MOCK


def build_space_config() -> AppConfig:
    workspace_root = Path(os.getenv("CONARRATIVE_WORKSPACE", "/data/conarrative")).expanduser()
    workspace = WorkspaceConfig(
        root=str(workspace_root),
        database_path=str(workspace_root / "conarrative.db"),
        exports_dir=str(workspace_root / "exports"),
        runtime_settings_path=str(workspace_root / "runtime_settings.json"),
    )
    backend = RuntimeSettings(
        provider=_provider_from_env(os.getenv("CONARRATIVE_PROVIDER")),
        base_url=os.getenv("CONARRATIVE_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=os.getenv("CONARRATIVE_MODEL", "local-model"),
        api_key=os.getenv("CONARRATIVE_API_KEY", "not-needed"),
    )
    server = ServerConfig(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )
    config = AppConfig(
        backend=backend,
        server=server,
        workspace=workspace,
        orchestration=OrchestrationConfig(),
    )
    config.ensure_directories()
    return config
