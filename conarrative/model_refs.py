from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=256)
def is_adapter_reference(model_ref: str | Path) -> bool:
    candidate = Path(str(model_ref))
    if candidate.exists():
        return (candidate / "adapter_config.json").exists()

    try:
        from peft import PeftConfig
    except Exception:
        return False

    try:
        PeftConfig.from_pretrained(str(model_ref))
        return True
    except Exception:
        return False
