from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


TRAINING_METADATA_FILENAME = "conarrative_training_metadata.json"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def normalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    return {str(key): json_ready(value) for key, value in metrics.items()}


def build_training_metadata(
    *,
    mode: str,
    model_name_or_path: str,
    train_file: str,
    eval_file: str | None,
    output_dir: str,
    dataset_format: str,
    train_examples: int,
    eval_examples: int,
    train_metrics: dict[str, Any] | None,
    eval_metrics: dict[str, Any] | None,
    trainer_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "generated_at": utcnow_iso(),
        "generated_by": "scripts/train_qlora.py",
        "mode": mode,
        "model_name_or_path": model_name_or_path,
        "train_file": train_file,
        "eval_file": eval_file or "",
        "output_dir": output_dir,
        "dataset_format": dataset_format,
        "train_examples": int(train_examples),
        "eval_examples": int(eval_examples),
        "train_metrics": normalize_metrics(train_metrics),
        "eval_metrics": normalize_metrics(eval_metrics),
        "trainer_state": json_ready(trainer_state or {}),
    }
    if extra:
        payload["extra"] = json_ready(extra)
    return payload


def metadata_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / TRAINING_METADATA_FILENAME


def write_training_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> Path:
    path = metadata_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_training_metadata(source_dir: str | Path) -> dict[str, Any] | None:
    path = metadata_path(source_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
