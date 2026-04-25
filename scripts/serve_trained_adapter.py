from __future__ import annotations

import argparse
import json
import threading
import traceback
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for a trained CoNarrative LoRA adapter")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--model-id", default="conarrative-trained")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--max-new-tokens", type=int, default=2600)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


class AdapterRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = None
        if use_cuda and not self.args.no_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.adapter_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "device_map": {"": 0} if use_cuda else None,
            "dtype": dtype if use_cuda else torch.float32,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
        }
        if use_cuda:
            load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(self.args.base_model, **load_kwargs)
        model = PeftModel.from_pretrained(model, self.args.adapter_dir)
        model.eval()
        self.model = model

    def render_prompt(self, messages: list[dict[str, str]]) -> Any:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = self.model.device
        if hasattr(rendered, "items"):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in rendered.items()
            }
        return {"input_ids": rendered.to(device)}

    def generate(self, payload: dict[str, Any]) -> str:
        text = self._generate_once(payload, use_adapter=True)
        if self._looks_degenerate(text):
            print("Adapter output looked degenerate; retrying once with the base model.", flush=True)
            fallback = self._generate_once(payload, use_adapter=False)
            if fallback and not self._looks_degenerate(fallback):
                return fallback
        return text

    def _generate_once(self, payload: dict[str, Any], *, use_adapter: bool) -> str:
        import torch

        messages = payload.get("messages") or []
        if not messages:
            raise ValueError("messages is required")
        max_new_tokens = int(payload.get("max_tokens") or self.args.max_new_tokens)
        temperature = float(payload.get("temperature", 0.7))
        inputs = self.render_prompt(messages)
        input_ids = inputs["input_ids"]
        adapter_context = nullcontext() if use_adapter or not hasattr(self.model, "disable_adapter") else self.model.disable_adapter()
        with self.lock, torch.inference_mode():
            with adapter_context:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0.0,
                    top_p=float(payload.get("top_p", 0.95)),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        generated = output[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def stream(self, payload: dict[str, Any]):
        text = self.generate(payload)
        for index in range(0, len(text), 36):
            yield text[index : index + 36]

    @staticmethod
    def _looks_degenerate(text: str) -> bool:
        visible = [char for char in (text or "").strip() if not char.isspace()]
        if not visible:
            return True
        if len(visible) < 12:
            return False
        question_ratio = sum(1 for char in visible if char == "?") / len(visible)
        replacement_ratio = sum(1 for char in visible if char == "\ufffd") / len(visible)
        hangul_count = sum(1 for char in visible if "\uac00" <= char <= "\ud7a3")
        if replacement_ratio > 0.05:
            return True
        if question_ratio > 0.35 and hangul_count < 4:
            return True
        if len(visible) >= 40 and question_ratio > 0.15 and hangul_count == 0:
            return True
        return False


def make_handler(runtime: AdapterRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": runtime.args.model_id,
                                "object": "model",
                                "owned_by": "conarrative",
                            }
                        ],
                    }
                )
                return
            self._json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json({"error": "not found"}, status=404)
                return
            try:
                payload = self._read_json()
                if payload.get("stream"):
                    self._stream_completion(payload)
                    return
                text = runtime.generate(payload)
                self._json(self._completion_payload(text))
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc) or exc.__class__.__name__}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _completion_payload(self, text: str) -> dict[str, Any]:
            return {
                "id": "chatcmpl-conarrative-trained",
                "object": "chat.completion",
                "model": runtime.args.model_id,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
            }

        def _stream_completion(self, payload: dict[str, Any]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                for delta in runtime.stream(payload):
                    event = {
                        "id": "chatcmpl-conarrative-trained",
                        "object": "chat.completion.chunk",
                        "model": runtime.args.model_id,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except Exception as exc:
                traceback.print_exc()
                error_event = {
                    "id": "chatcmpl-conarrative-trained",
                    "object": "chat.completion.chunk",
                    "model": runtime.args.model_id,
                    "choices": [{"index": 0, "delta": {"content": f"\n[stream error: {exc}]\n"}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            done = {
                "id": "chatcmpl-conarrative-trained",
                "object": "chat.completion.chunk",
                "model": runtime.args.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def main() -> None:
    args = parse_args()
    server = None
    print(
        f"Starting trained adapter server: model_id={args.model_id}, base_model={args.base_model}, adapter={args.adapter_dir}",
        flush=True,
    )
    try:
        runtime = AdapterRuntime(args)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
        server.daemon_threads = True
        print(f"Serving {args.model_id} at http://{args.host}:{args.port}/v1", flush=True)
        server.serve_forever()
    except BaseException:
        print("Trained adapter server crashed or stopped unexpectedly.", flush=True)
        traceback.print_exc()
        raise
    finally:
        if server is not None:
            server.server_close()
        print("Trained adapter server stopped.", flush=True)


if __name__ == "__main__":
    main()
