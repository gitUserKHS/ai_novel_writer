#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import locale
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conarrative.model_refs import is_adapter_reference
from conarrative.utils import short_text, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CoNarrative writer/critic adapter with QLoRA.")
    parser.add_argument("--config", help="YAML config file. CLI args override config values.")
    parser.add_argument("--mode", choices=["sft", "dpo", "distill"], default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validation-split-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None, help="DPO beta")
    parser.add_argument("--dataset-format", choices=["auto", "chat", "world_model"], default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def merged_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = {
        "mode": "sft",
        "model_name_or_path": None,
        "train_file": None,
        "eval_file": None,
        "output_dir": None,
        "load_in_4bit": True,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 1.0,
        "max_seq_length": 4096,
        "gradient_checkpointing": True,
        "validation_split_ratio": 0.0,
        "seed": 42,
        "early_stopping_patience": 2,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "save_total_limit": 2,
        "logging_steps": 5,
        "max_train_samples": None,
        "max_eval_samples": None,
        "beta": 0.1,
        "dataset_format": "auto",
    }
    cfg.update(load_config(args.config))
    boolean_flags = {"load_in_4bit", "no_load_in_4bit", "gradient_checkpointing", "print_config", "dry_run"}
    meta_flags = {"print_config", "dry_run"}
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        if key in boolean_flags and value is False:
            continue
        if key in meta_flags:
            continue
        if key == "no_load_in_4bit":
            if value:
                cfg["load_in_4bit"] = False
            continue
        cfg[key] = value
    return cfg


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def require_training_stack() -> Dict[str, Any]:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"  # type: ignore[assignment]
    except Exception:
        pass
    try:
        import torch
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, set_seed
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            f"Missing or incompatible training dependencies: {type(exc).__name__}: {exc}. Install with: pip install -e .[training]"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoPeftModelForCausalLM": AutoPeftModelForCausalLM,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "set_seed": set_seed,
        "SFTConfig": SFTConfig,
        "DPOTrainer": DPOTrainer,
        "DPOConfig": DPOConfig,
        "SFTTrainer": SFTTrainer,
    }


def simple_chat_template(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"<|{role}|>\n{content}")
    return "\n\n".join(lines)


def build_world_model_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    output = []
    for row in rows:
        request = row.get("request", {}) or {}
        plan = row.get("plan", {}) or {}
        previous_state = row.get("previous_state", {}) or {}
        messages = [
            {
                "role": "system",
                "content": "You are a narrative world model. Predict the next structured story state implied by the scene and return JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "story_id": row.get("story_id"),
                        "scene_id": row.get("scene_id"),
                        "scene_index": row.get("scene_index"),
                        "request": {
                            "pov": request.get("pov", ""),
                            "location": request.get("location", ""),
                            "time_label": request.get("time_label", ""),
                            "goal": request.get("goal", ""),
                            "foreshadowing": request.get("foreshadowing", []),
                            "required_facts": request.get("required_facts", []),
                        },
                        "plan": {
                            "scene_title": plan.get("scene_title", ""),
                            "must_include": plan.get("must_include", []),
                        },
                        "previous_state": {
                            "current_time_label": previous_state.get("current_time_label", ""),
                            "current_location": previous_state.get("current_location", ""),
                            "active_threads": previous_state.get("active_threads", []),
                            "resolved_threads": previous_state.get("resolved_threads", []),
                            "inventory": previous_state.get("inventory", {}),
                            "emotional_state": previous_state.get("emotional_state", {}),
                        },
                        "accepted_text": short_text(str(row.get("accepted_text", "") or ""), 1600),
                        "schema": {
                            "next_state": {},
                            "extraction": {
                                "summary": "string",
                                "new_static_facts": ["string"],
                                "state_updates": {},
                                "new_threads": ["string"],
                                "resolved_threads": ["string"],
                                "knowledge_updates": {},
                                "inventory_updates": {},
                                "emotional_updates": {},
                                "kg_edges": [{"source": "string", "relation": "string", "target": "string"}],
                            },
                            "notes": ["string"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "next_state": row.get("next_state", {}),
                        "extraction": row.get("extraction", {}),
                        "notes": [],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = simple_chat_template(messages)
        output.append({"text": text})
    return output


def build_sft_rows(rows: list[dict[str, Any]], tokenizer: Any, dataset_format: str = "auto") -> list[dict[str, str]]:
    if dataset_format == "world_model":
        return build_world_model_rows(rows, tokenizer)
    output = []
    for row in rows:
        if "messages" in row:
            messages = row["messages"]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                text = simple_chat_template(messages)
            output.append({"text": text})
            continue
        prompt = row.get("prompt", "")
        completion = row.get("completion", row.get("accepted_text", ""))
        if prompt or completion:
            output.append({"text": f"{prompt}\n\n{completion}".strip()})
    return output


def build_distill_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    output = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", "")).strip()
        teacher_trace = row.get("teacher_trace", {}) or {}
        assistant_payload = {
            "teacher_trace": teacher_trace,
            "completion": completion,
        }
        messages = [
            {"role": "system", "content": "당신은 장면 생성 교사 모델이다. 계획 이유, 비평 요약, 수정 결과, 최종 장면을 구조화해 반환한다."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = simple_chat_template(messages)
        output.append({"text": text})
    return output


def load_tokenizer(model_name_or_path: str, auto_tokenizer: Any) -> Any:
    tokenizer = auto_tokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config(cfg: Dict[str, Any], torch: Any, bits_and_bytes_config: Any) -> Any | None:
    if not cfg["load_in_4bit"]:
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return bits_and_bytes_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: Dict[str, Any], lora_config_cls: Any) -> Any:
    return lora_config_cls(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def load_trainable_model(cfg: Dict[str, Any], stack: Dict[str, Any]) -> tuple[Any, Any, Any | None]:
    torch = stack["torch"]
    AutoPeftModelForCausalLM = stack["AutoPeftModelForCausalLM"]
    LoraConfig = stack["LoraConfig"]
    prepare_model_for_kbit_training = stack["prepare_model_for_kbit_training"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]

    quantization_config = build_quantization_config(cfg, torch, BitsAndBytesConfig)
    tokenizer = load_tokenizer(cfg["model_name_or_path"], AutoTokenizer)

    if is_adapter_reference(cfg["model_name_or_path"]):
        model = AutoPeftModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"],
            is_trainable=True,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
        )
        if cfg["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        return tokenizer, model, None

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    return tokenizer, model, build_lora_config(cfg, LoraConfig)


def row_story_id(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) or {}
    for key in ["story_id", "source_story_id"]:
        value = metadata.get(key) or row.get(key)
        if value:
            return str(value)
    return stable_hash({"prompt": row.get("prompt"), "messages": row.get("messages"), "text": row.get("text")})


def truncate_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def split_rows_for_eval(rows: list[dict[str, Any]], validation_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_ratio <= 0 or len(rows) < 2:
        return rows, []

    story_ids = sorted({row_story_id(row) for row in rows})
    if len(story_ids) >= 2:
        holdout_count = min(len(story_ids) - 1, max(1, int(round(len(story_ids) * validation_ratio))))
        ranked_story_ids = sorted(story_ids, key=lambda story_id: stable_hash({"seed": seed, "story_id": story_id}))
        holdout_story_ids = set(ranked_story_ids[:holdout_count])
        train_rows = [row for row in rows if row_story_id(row) not in holdout_story_ids]
        eval_rows = [row for row in rows if row_story_id(row) in holdout_story_ids]
        if train_rows and eval_rows:
            return train_rows, eval_rows

    eval_count = min(len(rows) - 1, max(1, int(round(len(rows) * validation_ratio))))
    ranked_rows = sorted(rows, key=lambda row: stable_hash({"seed": seed, "row": row}))
    eval_rows = ranked_rows[:eval_count]
    train_rows = ranked_rows[eval_count:]
    return train_rows, eval_rows


def load_datasets(cfg: Dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = read_jsonl(cfg["train_file"])
    eval_rows: list[dict[str, Any]] = []
    if cfg.get("eval_file"):
        eval_rows = read_jsonl(cfg["eval_file"])
    elif float(cfg.get("validation_split_ratio") or 0.0) > 0:
        train_rows, eval_rows = split_rows_for_eval(train_rows, float(cfg["validation_split_ratio"]), int(cfg["seed"]))
    train_rows = truncate_rows(train_rows, cfg.get("max_train_samples"))
    eval_rows = truncate_rows(eval_rows, cfg.get("max_eval_samples"))
    return train_rows, eval_rows


def common_training_kwargs(cfg: Dict[str, Any], torch: Any, has_eval: bool) -> dict[str, Any]:
    return {
        "output_dir": cfg["output_dir"],
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "num_train_epochs": cfg["num_train_epochs"],
        "logging_steps": cfg["logging_steps"],
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if has_eval else "no",
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": [],
        "max_length": cfg["max_seq_length"],
        "seed": cfg["seed"],
        "data_seed": cfg["seed"],
        "weight_decay": cfg["weight_decay"],
        "warmup_ratio": cfg["warmup_ratio"],
        "save_total_limit": cfg["save_total_limit"],
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False if has_eval else None,
        "do_eval": has_eval,
    }


def training_callbacks(cfg: Dict[str, Any], stack: Dict[str, Any], has_eval: bool) -> list[Any]:
    if not has_eval or int(cfg.get("early_stopping_patience") or 0) <= 0:
        return []
    return [stack["EarlyStoppingCallback"](early_stopping_patience=int(cfg["early_stopping_patience"]))]


def train_sft(cfg: Dict[str, Any], stack: Dict[str, Any]) -> None:
    torch = stack["torch"]
    Dataset = stack["Dataset"]
    SFTConfig = stack["SFTConfig"]
    SFTTrainer = stack["SFTTrainer"]
    stack["set_seed"](cfg["seed"])

    tokenizer, model, peft_config = load_trainable_model(cfg, stack)
    train_rows, eval_rows = load_datasets(cfg)
    if not train_rows:
        raise SystemExit(f"Training file is empty: {cfg['train_file']}")
    train_dataset = Dataset.from_list(build_sft_rows(train_rows, tokenizer, dataset_format=str(cfg.get("dataset_format", "auto"))))
    eval_dataset = Dataset.from_list(build_sft_rows(eval_rows, tokenizer, dataset_format=str(cfg.get("dataset_format", "auto")))) if eval_rows else None
    training_args = SFTConfig(
        dataset_text_field="text",
        **common_training_kwargs(cfg, torch, eval_dataset is not None),
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        callbacks=training_callbacks(cfg, stack, eval_dataset is not None),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


def train_distill(cfg: Dict[str, Any], stack: Dict[str, Any]) -> None:
    torch = stack["torch"]
    Dataset = stack["Dataset"]
    SFTConfig = stack["SFTConfig"]
    SFTTrainer = stack["SFTTrainer"]
    stack["set_seed"](cfg["seed"])

    tokenizer, model, peft_config = load_trainable_model(cfg, stack)
    train_rows, eval_rows = load_datasets(cfg)
    if not train_rows:
        raise SystemExit(f"Training file is empty: {cfg['train_file']}")
    train_dataset = Dataset.from_list(build_distill_rows(train_rows, tokenizer))
    eval_dataset = Dataset.from_list(build_distill_rows(eval_rows, tokenizer)) if eval_rows else None
    training_args = SFTConfig(
        dataset_text_field="text",
        **common_training_kwargs(cfg, torch, eval_dataset is not None),
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        callbacks=training_callbacks(cfg, stack, eval_dataset is not None),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


def train_dpo(cfg: Dict[str, Any], stack: Dict[str, Any]) -> None:
    torch = stack["torch"]
    Dataset = stack["Dataset"]
    DPOConfig = stack["DPOConfig"]
    DPOTrainer = stack["DPOTrainer"]
    stack["set_seed"](cfg["seed"])

    tokenizer, model, peft_config = load_trainable_model(cfg, stack)
    train_rows, eval_rows = load_datasets(cfg)
    if not train_rows:
        raise SystemExit(f"Training file is empty: {cfg['train_file']}")
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    training_args = DPOConfig(
        beta=cfg["beta"],
        **common_training_kwargs(cfg, torch, eval_dataset is not None),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=training_callbacks(cfg, stack, eval_dataset is not None),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


def main() -> None:
    args = parse_args()
    cfg = merged_config(args)
    missing = [key for key in ["model_name_or_path", "train_file", "output_dir"] if not cfg.get(key)]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    train_rows, eval_rows = load_datasets(cfg)
    if args.dry_run:
        preview = {
            "ok": Path(cfg["train_file"]).exists(),
            "mode": cfg["mode"],
            "model_name_or_path": cfg["model_name_or_path"],
            "train_file": str(Path(cfg["train_file"])),
            "train_file_exists": Path(cfg["train_file"]).exists(),
            "eval_file": str(Path(cfg["eval_file"])) if cfg.get("eval_file") else None,
            "eval_file_exists": Path(cfg["eval_file"]).exists() if cfg.get("eval_file") else False,
            "output_dir": str(Path(cfg["output_dir"])),
            "output_dir_exists": Path(cfg["output_dir"]).exists(),
            "load_in_4bit": cfg["load_in_4bit"],
            "max_seq_length": cfg["max_seq_length"],
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
            "validation_split_ratio": cfg["validation_split_ratio"],
            "early_stopping_patience": cfg["early_stopping_patience"],
            "seed": cfg["seed"],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    stack = require_training_stack()
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    if cfg["mode"] == "sft":
        train_sft(cfg, stack)
    elif cfg["mode"] == "dpo":
        train_dpo(cfg, stack)
    elif cfg["mode"] == "distill":
        train_distill(cfg, stack)
    else:
        raise SystemExit(f"Unsupported mode: {cfg['mode']}")


if __name__ == "__main__":
    main()
