from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from conarrative.hf_release import next_release_tag, suggest_repo_id
from conarrative.training_metadata import build_training_metadata, write_training_metadata
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


def test_build_model_card_includes_training_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "adapter"
    source_dir.mkdir()
    (source_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B"}),
        encoding="utf-8",
    )
    metadata = build_training_metadata(
        mode="sft",
        model_name_or_path="Qwen/Qwen3-4B",
        train_file="outputs/generalist_corpus/writer_sft_train.jsonl",
        eval_file="outputs/generalist_corpus/writer_sft_eval.jsonl",
        output_dir=str(source_dir),
        dataset_format="chat",
        train_examples=128,
        eval_examples=32,
        train_metrics={"train_loss": 1.23, "train_runtime": 12.5},
        eval_metrics={"eval_loss": 1.11},
        trainer_state={"best_metric": 1.11, "best_model_checkpoint": "checkpoint-10"},
    )
    write_training_metadata(source_dir, metadata)

    card = build_card("your-org/conarrative-writer-qwen3-4b-sft-lora", "model", source_dir)
    assert "Training Summary" in card
    assert "Artifact Lineage" in card
    assert "writer_sft_train.jsonl" in card
    assert "eval `eval_loss`" in card


def test_release_helpers_suggest_repo_and_next_tag() -> None:
    repo_id = suggest_repo_id(
        "your-org",
        repo_type="model",
        project="conarrative",
        role="critic",
        base_model="Qwen/Qwen3-4B",
        stage="consistency",
    )
    assert repo_id == "your-org/conarrative-critic-qwen3-4b-consistency-lora"
    assert next_release_tag(["v0.1.0", "v0.1.1"]) == "v0.1.2"
