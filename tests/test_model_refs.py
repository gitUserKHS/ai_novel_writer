from __future__ import annotations

from pathlib import Path

from conarrative.model_refs import is_adapter_reference


def test_is_adapter_reference_for_local_directory(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    non_adapter_dir = tmp_path / "plain_model"
    non_adapter_dir.mkdir()

    assert is_adapter_reference(str(adapter_dir)) is True
    assert is_adapter_reference(str(non_adapter_dir)) is False
