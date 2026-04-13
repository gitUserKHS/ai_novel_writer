
from __future__ import annotations

import json
from pathlib import Path

from .models import RuntimeSettings
from .utils import ensure_dir


def _deep_merge(default_payload: dict, override_payload: dict) -> dict:
    merged = dict(default_payload)
    for key, value in override_payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class RuntimeSettingsStore:
    def __init__(self, path: str | Path, default: RuntimeSettings) -> None:
        self.path = Path(path)
        self.default = RuntimeSettings(**default.model_dump())
        ensure_dir(self.path.parent)
        if not self.path.exists():
            self.save(self.default)

    def load(self) -> RuntimeSettings:
        if not self.path.exists():
            return self.save(self.default)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        merged = _deep_merge(self.default.model_dump(), payload)
        return RuntimeSettings(**merged)

    def save(self, settings: RuntimeSettings) -> RuntimeSettings:
        ensure_dir(self.path.parent)
        self.path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        return settings
