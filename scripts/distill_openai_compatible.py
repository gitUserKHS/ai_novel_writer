from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import httpx


DISTILLATION_SYSTEM_PROMPT = (
    "당신은 한국어 장면 소설을 쓰는 상위 교사 모델이다. "
    "입력된 기억 문맥과 장면 요청을 엄격히 지키면서, 자연스럽고 개연성 있는 한국어 장면 본문만 작성하라. "
    "영어 설명, 제목, 메타 문장 없이 결과만 출력하라."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill prompt-only CoNarrative examples using an OpenAI-compatible teacher")
    parser.add_argument("--input-file", required=True, help="prompt_only_teacher.jsonl path")
    parser.add_argument("--output-file", required=True, help="Output JSONL path with assistant responses added")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, for example http://127.0.0.1:1234/v1")
    parser.add_argument("--model", required=True, help="Teacher model name")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--resume", action="store_true", help="Append only missing rows if the output file already exists")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_keys(rows: Iterable[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        story_id = metadata.get("story_id", "")
        scene_index = metadata.get("scene_index", "")
        keys.add(f"{story_id}::{scene_index}")
    return keys


def make_key(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return f"{metadata.get('story_id', '')}::{metadata.get('scene_index', '')}"


def call_teacher(client: httpx.Client, args: argparse.Namespace, messages: list[Dict[str, str]]) -> str:
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": DISTILLATION_SYSTEM_PROMPT}, *messages],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    response = client.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover - depends on backend shape
        raise RuntimeError(f"Unexpected teacher response format: {json.dumps(data)[:400]}") from exc


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    mode = "w"
    if args.resume and output_path.exists():
        existing = load_jsonl(output_path)
        done = completed_keys(existing)
        mode = "a"

    with httpx.Client(timeout=args.timeout_seconds) as client, output_path.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(source_rows, start=1):
            key = make_key(row)
            if key in done:
                continue
            messages = row.get("messages") or []
            if not messages:
                raise SystemExit(f"Row {index} has no messages field.")
            assistant = call_teacher(client, args, messages)
            distilled = {
                "messages": [*messages, {"role": "assistant", "content": assistant}],
                "metadata": {
                    **(row.get("metadata") or {}),
                    "teacher_model": args.model,
                    "teacher_base_url": args.base_url,
                    "source": "distilled_prompt_only",
                },
            }
            handle.write(json.dumps(distilled, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(source_rows)}] distilled {key}")


if __name__ == "__main__":
    main()
