from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import RuntimeSettings


class RuntimeSettingsStore:
    def __init__(self, path: str, defaults: RuntimeSettings) -> None:
        self.path = Path(path)
        self.defaults = defaults
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _defaults_copy(self) -> RuntimeSettings:
        return RuntimeSettings.model_validate(self.defaults.model_dump())

    def load(self) -> RuntimeSettings:
        if not self.path.exists():
            return self._defaults_copy()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return self._defaults_copy()
        if not isinstance(raw, dict):
            return self._defaults_copy()

        merged = self.defaults.model_dump()
        for field_name, field_value in raw.items():
            if field_name not in RuntimeSettings.model_fields:
                continue
            try:
                candidate = RuntimeSettings.model_validate({**merged, field_name: field_value})
            except ValidationError:
                continue
            merged[field_name] = candidate.model_dump()[field_name]
        return RuntimeSettings.model_validate(merged)

    def save(self, settings: RuntimeSettings) -> RuntimeSettings:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(settings.model_dump(), f, ensure_ascii=False, indent=2)
        return settings
