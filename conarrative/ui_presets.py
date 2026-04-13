from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import UIPresetRecord, utcnow_iso
from .utils import ensure_dir


class UIPresetStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        if not self.path.exists():
            self.path.write_text(json.dumps({}, indent=2), encoding="utf-8")

    def _load_payload(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _save_payload(self, payload: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        ensure_dir(self.path.parent)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_all(self) -> Dict[str, List[Dict[str, Any]]]:
        payload = self._load_payload()
        output: Dict[str, List[Dict[str, Any]]] = {}
        for kind, items in payload.items():
            output[kind] = [
                UIPresetRecord(kind=kind, name=name, **record).model_dump(mode="json")
                for name, record in sorted(items.items())
            ]
        return output

    def save(self, kind: str, name: str, preset_payload: Dict[str, Any]) -> UIPresetRecord:
        normalized_kind = str(kind).strip()
        normalized_name = str(name).strip()
        if not normalized_kind:
            raise ValueError("Preset kind is required")
        if not normalized_name:
            raise ValueError("Preset name is required")
        payload = self._load_payload()
        payload.setdefault(normalized_kind, {})
        record = UIPresetRecord(
            kind=normalized_kind,
            name=normalized_name,
            payload=dict(preset_payload or {}),
            saved_at=utcnow_iso(),
        )
        payload[normalized_kind][normalized_name] = {
            "payload": record.payload,
            "saved_at": record.saved_at,
        }
        self._save_payload(payload)
        return record

    def delete(self, kind: str, name: str) -> bool:
        payload = self._load_payload()
        items = payload.get(kind, {})
        if name not in items:
            return False
        del items[name]
        if not items:
            payload.pop(kind, None)
        self._save_payload(payload)
        return True
