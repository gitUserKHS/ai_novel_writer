from __future__ import annotations

import json

from conarrative.models import RuntimeSettings
from conarrative.runtime_settings import RuntimeSettingsStore


def test_runtime_settings_store_deep_merges_nested_dicts(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "runtime_settings.json"
    default = RuntimeSettings(
        role_models={"writer": "qwen3:4b", "world_model": "outputs/training_qwen3_4b_world_model"},
        extra_headers={"X-Test": "default", "X-Trace": "keep"},
    )
    RuntimeSettingsStore(path, default)
    path.write_text(
        json.dumps(
            {
                "role_models": {"writer": "gemma4:e2b"},
                "extra_headers": {"X-Test": "override"},
            }
        ),
        encoding="utf-8",
    )

    loaded = RuntimeSettingsStore(path, default).load()

    assert loaded.role_models == {
        "writer": "gemma4:e2b",
        "world_model": "outputs/training_qwen3_4b_world_model",
    }
    assert loaded.extra_headers == {
        "X-Test": "override",
        "X-Trace": "keep",
    }
