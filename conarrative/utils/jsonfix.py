from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json_payload(text: str) -> str:
    text = text.strip()
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [idx for idx in [start_obj, start_arr] if idx != -1]
    if not starts:
        raise ValueError("No JSON payload found in model output")
    start = min(starts)
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return text[start:]


def loads_json_loose(text: str) -> Any:
    payload = extract_json_payload(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        normalized = payload.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        return json.loads(normalized)
