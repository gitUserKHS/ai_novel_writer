from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA/SFT trainer for CoNarrative Studio datasets")
    parser.add_argument("--train-file", required=True, nargs="+", help="One or more JSONL files with a messages field")
    parser.add_argument("--eval-file", default=None, help="Optional validation JSONL file")
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
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
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU training for debugging only")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional training stack
        raise SystemExit(
            "Training dependencies are missing. Install them with "
            "`pip install -r requirements-train.txt` before running this script."
        ) from exc

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available in the current Python environment. "
            "Install a CUDA-enabled PyTorch build or run with --allow-cpu for a very slow debug run."
        )

    use_4bit = not args.no_4bit
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    def apply_template(batch):
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
        return {"text": texts}

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        tokens["labels"] = [input_ids[:] for input_ids in tokens["input_ids"]]
        return tokens

    train_dataset = dataset["train"].map(apply_template, batched=True)
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)

    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = dataset["validation"].map(apply_template, batched=True)
        eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        save_total_limit=2,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
