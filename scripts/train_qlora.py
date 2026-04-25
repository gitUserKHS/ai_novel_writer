from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA/SFT trainer for CoNarrative Studio datasets")
    parser.add_argument("--train-file", required=True, nargs="+", help="One or more JSONL files with a messages field")
    parser.add_argument("--eval-file", default=None, help="Optional validation JSONL file")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--profile", choices=["auto", "low-vram", "quality"], default="auto")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "disabled"], default="sdpa")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU training for debugging only")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-adapter", default="", help="Existing LoRA adapter directory to continue training from")
    return parser.parse_args()


def cuda_total_gib(torch) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def configure_torch_speed(torch) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def is_low_vram_profile(args, torch) -> bool:
    if args.profile == "low-vram":
        return True
    if args.profile == "quality":
        return False
    total_gib = cuda_total_gib(torch)
    return bool(total_gib and total_gib <= 8.5)


def recommend_max_seq_length(args, torch) -> int:
    if is_low_vram_profile(args, torch) and args.max_seq_length > 2048:
        total_gib = cuda_total_gib(torch)
        print(
            f"Detected {total_gib:.1f} GiB VRAM. "
            f"Reducing max_seq_length from {args.max_seq_length} to 2048 for a safer 8 GB training profile."
        )
        return 2048
    return args.max_seq_length


def recommend_lora_config(args, torch) -> tuple[int, int]:
    if is_low_vram_profile(args, torch) and args.lora_r > 8:
        total_gib = cuda_total_gib(torch)
        alpha = min(args.lora_alpha, 16)
        print(
            f"Detected {total_gib:.1f} GiB VRAM. "
            f"Reducing LoRA rank from r={args.lora_r}, alpha={args.lora_alpha} to r=8, alpha={alpha}."
        )
        return 8, alpha
    return args.lora_r, args.lora_alpha


def load_model_with_fallbacks(
    *,
    args,
    torch,
    AutoModelForCausalLM,
    quantization_config,
    torch_dtype,
):
    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": torch_dtype if torch.cuda.is_available() else torch.float32,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available() and args.attn_implementation != "disabled":
        load_kwargs["attn_implementation"] = args.attn_implementation
    strategies = []
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        print(f"CUDA memory before model load: free={free_gib:.2f} GiB / total={total_gib:.2f} GiB")
        torch.cuda.empty_cache()
        # Prefer a single-GPU placement first. For 4-bit fine-tuning on 8 GB cards,
        # `device_map=\"auto\"` can spuriously offload some modules to CPU/disk and fail early.
        strategies.append({"device_map": {"": 0}})
        strategies.append({"device_map": "auto"})
    else:
        strategies.append({"device_map": None})

    errors = []
    def load_with_strategy(strategy):
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **load_kwargs,
            device_map=strategy["device_map"],
        )

    for index, strategy in enumerate(strategies, start=1):
        try:
            print(f"Loading model with strategy {index}/{len(strategies)}: device_map={strategy['device_map']}")
            return load_with_strategy(strategy)
        except TypeError as exc:
            text = str(exc)
            errors.append(text)
            if "attn_implementation" in text and "attn_implementation" in load_kwargs:
                print("This model loader does not accept attn_implementation. Retrying without it.")
                load_kwargs.pop("attn_implementation", None)
                return load_with_strategy(strategy)
            raise
        except ValueError as exc:
            text = str(exc)
            errors.append(text)
            if "dispatched on the CPU or the disk" in text:
                print("Model loader tried to offload quantized modules to CPU/disk. Retrying with a different placement strategy.")
                continue
            raise
        except RuntimeError as exc:
            text = str(exc)
            errors.append(text)
            if torch.cuda.is_available() and "out of memory" in text.lower():
                torch.cuda.empty_cache()
                print("CUDA OOM while loading the model. Retrying with the next placement strategy.")
                continue
            raise

    summary = "\n\n".join(errors[-2:]) if errors else "unknown error"
    raise RuntimeError(
        "Unable to load the base model on the current GPU. "
        "Close other GPU-heavy apps or unload the teacher model server, then retry.\n\n"
        f"{summary}"
    )


def load_jsonl_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"Missing non-empty messages list in {path}:{line_number}")
                rows.append({"messages": messages})
    if not rows:
        raise ValueError("No training rows were loaded from --train-file.")
    return rows


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional training stack
        raise SystemExit(
            "Training dependencies are missing. Install them with "
            "`pip install -r requirements-train.txt` before running this script."
        ) from exc

    train_rows = load_jsonl_rows(args.train_file)
    eval_rows = load_jsonl_rows([args.eval_file]) if args.eval_file else []

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available in the current Python environment. "
            "Install a CUDA-enabled PyTorch build or run with --allow-cpu for a very slow debug run."
        )

    use_4bit = not args.no_4bit
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    configure_torch_speed(torch)
    effective_max_seq_length = recommend_max_seq_length(args, torch)
    effective_lora_r, effective_lora_alpha = recommend_lora_config(args, torch)
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_fallbacks(
        args=args,
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    resume_adapter = Path(args.resume_adapter) if args.resume_adapter else None
    if resume_adapter and resume_adapter.exists():
        print(f"Continuing training from adapter: {resume_adapter}")
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    dataset = {"train": Dataset.from_list(train_rows)}
    if eval_rows:
        dataset["validation"] = Dataset.from_list(eval_rows)

    def _render_chat(messages, *, add_generation_prompt: bool) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            rendered = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
            if add_generation_prompt:
                rendered.append("assistant:")
            return "\n".join(rendered)

    def apply_template(batch):
        texts = []
        prompt_texts = []
        for messages in batch["messages"]:
            messages = list(messages or [])
            texts.append(_render_chat(messages, add_generation_prompt=False))
            if messages and messages[-1].get("role") == "assistant":
                prompt_texts.append(_render_chat(messages[:-1], add_generation_prompt=True))
            else:
                prompt_texts.append(_render_chat(messages, add_generation_prompt=False))
        return {"text": texts, "prompt_text": prompt_texts}

    def tokenize(batch):
        encoded = {"input_ids": [], "attention_mask": [], "labels": []}
        for text, prompt_text in zip(batch["text"], batch["prompt_text"]):
            full = tokenizer(
                text,
                truncation=True,
                max_length=effective_max_seq_length,
                padding=False,
            )
            prefix = tokenizer(
                prompt_text,
                truncation=True,
                max_length=effective_max_seq_length,
                padding=False,
            )
            input_ids = full["input_ids"]
            labels = input_ids[:]
            prefix_len = min(len(prefix["input_ids"]), len(labels))
            labels[:prefix_len] = [-100] * prefix_len
            if labels and all(value == -100 for value in labels):
                labels[-1] = input_ids[-1]
            encoded["input_ids"].append(input_ids)
            encoded["attention_mask"].append(full["attention_mask"])
            encoded["labels"].append(labels)
        return encoded

    num_proc = args.dataset_num_proc if args.dataset_num_proc > 1 else None
    train_dataset = dataset["train"].map(apply_template, batched=True, num_proc=num_proc)
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=num_proc, remove_columns=train_dataset.column_names)

    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = dataset["validation"].map(apply_template, batched=True, num_proc=num_proc)
        eval_dataset = eval_dataset.map(tokenize, batched=True, num_proc=num_proc, remove_columns=eval_dataset.column_names)

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "warmup_ratio": args.warmup_ratio,
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "remove_unused_columns": False,
        "eval_steps": args.save_steps if eval_dataset is not None else None,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
    }
    training_signature = inspect.signature(TrainingArguments.__init__)
    eval_strategy = "steps" if eval_dataset is not None else "no"
    if "eval_strategy" in training_signature.parameters:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
    if "save_safetensors" in training_signature.parameters:
        training_kwargs["save_safetensors"] = True
    if "gradient_checkpointing_kwargs" in training_signature.parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**training_kwargs)
    tokenizer.padding_side = "right"

    def collate_causal_lm(features):
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_causal_lm,
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
