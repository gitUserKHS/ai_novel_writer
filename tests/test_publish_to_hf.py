from __future__ import annotations

import json

import pytest

pytest.importorskip("huggingface_hub")

from scripts.publish_to_hf import build_card, infer_base_model


def test_infer_base_model_from_adapter_config(tmp_path: Path) -> None:
    source_dir = tmp_path / "adapter"
    source_dir.mkdir()
    (source_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B"}),
        encoding="utf-8",
    )

    assert infer_base_model(source_dir) == "Qwen/Qwen3-4B"


def test_build_model_card_contains_repo_and_base_model(tmp_path: Path) -> None:
    source_dir = tmp_path / "adapter"
    source_dir.mkdir()
    (source_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B"}),
        encoding="utf-8",
    )

    card = build_card("your-org/conarrative-writer-qwen3-4b-lora", "model", source_dir)
    assert "your-org/conarrative-writer-qwen3-4b-lora" in card
    assert "Qwen/Qwen3-4B" in card
    assert "conarrative" in card.lower()
