from __future__ import annotations

from conarrative.models import RuntimeSettings
from scripts.run_pipeline import configured_ollama_models, parse_ollama_ps, stage_output_dir, train_file_for_mode, uses_local_ollama


def test_uses_local_ollama_for_local_openai_compatible_backend() -> None:
    settings = RuntimeSettings(
        provider="openai_compatible",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="gemma4:e4b",
    )

    assert uses_local_ollama(settings) is True


def test_configured_ollama_models_collects_default_and_role_models() -> None:
    settings = RuntimeSettings(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        api_key="ollama",
        model="qwen3:4b",
        role_models={
            "planner": "qwen3:4b",
            "writer": "gemma4:e2b",
            "extractor": "",
        },
    )

    assert configured_ollama_models(settings) == ["gemma4:e2b", "qwen3:4b"]


def test_configured_ollama_models_ignores_local_role_paths() -> None:
    settings = RuntimeSettings(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        api_key="ollama",
        model="qwen3:4b",
        role_models={
            "consistency_critic": "outputs/training_qwen3_4b_critic_consistency",
            "writer": "qwen3:4b",
        },
    )

    assert configured_ollama_models(settings) == ["qwen3:4b"]


def test_parse_ollama_ps_reads_loaded_model_names() -> None:
    output = "\n".join(
        [
            "NAME        ID              SIZE      PROCESSOR    CONTEXT    UNTIL",
            "qwen3:4b    359d7dd4bcda    3.5 GB    100% GPU     4096       3 minutes from now",
            "gemma4:e2b  abcdef123456    2.7 GB    100% GPU     4096       1 minute from now",
        ]
    )

    assert parse_ollama_ps(output) == ["qwen3:4b", "gemma4:e2b"]


def test_train_file_for_mode_supports_distill() -> None:
    paths = {
        "writer_sft": "writer_sft.jsonl",
        "writer_dpo": "writer_dpo.jsonl",
        "distill_stepwise": "distill_stepwise.jsonl",
        "critic_consistency_sft": "critic_consistency_sft.jsonl",
    }

    assert train_file_for_mode(paths, "distill") == "distill_stepwise.jsonl"
    assert train_file_for_mode(paths, {"mode": "sft", "pool_key": "critic_consistency_sft"}) == "critic_consistency_sft.jsonl"


def test_stage_output_dir_appends_stage_name_for_multi_stage_runs() -> None:
    output_dir = stage_output_dir("configs/training_qwen3_4b_dpo.yaml", "outputs/chain", stage_index=2, stage_count=3)

    assert output_dir.endswith("outputs\\chain\\02_training_qwen3_4b_dpo")
