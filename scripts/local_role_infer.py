#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conarrative.utils import extract_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local structured role inference request in an isolated subprocess.")
    parser.add_argument("--request-file", required=True)
    return parser.parse_args()


def load_runtime(model_ref: str) -> tuple[Any, Any, Any]:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs["device_map"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["device_map"] = "cpu"

    adapter_config = Path(model_ref) / "adapter_config.json"
    if adapter_config.exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    model.eval()
    return torch, tokenizer, model


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
    model_ref = str(payload["model_ref"])
    messages = payload["messages"]
    max_tokens = int(payload["max_tokens"])
    temperature = float(payload["temperature"])
    assistant_prefill = str(payload.get("assistant_prefill", "") or "")

    torch, tokenizer, model = load_runtime(model_ref)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_messages = list(messages)
            apply_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            if assistant_prefill:
                prompt_messages = list(messages) + [{"role": "assistant", "content": assistant_prefill}]
                apply_kwargs = {
                    "tokenize": False,
                    "continue_final_message": True,
                    "enable_thinking": False,
                }
            prompt = tokenizer.apply_chat_template(prompt_messages, **apply_kwargs)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n\n".join(f"<|{item['role']}|>\n{item['content']}" for item in messages) + "\n\n<|assistant|>\n"
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.05,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1] :]
    content = (assistant_prefill + tokenizer.decode(generated, skip_special_tokens=True)).strip()
    try:
        parsed = extract_json_object(content)
    except Exception as exc:
        parsed = {
            "__raw_text__": content,
            "__parse_error__": str(exc),
        }
    sys.stdout.write(json.dumps(parsed, ensure_ascii=False))


if __name__ == "__main__":
    main()
