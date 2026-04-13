
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_list(items: Iterable[Any]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def slugify(text: str) -> str:
    ascii_hint = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", text.strip().lower())
    ascii_hint = re.sub(r"-{2,}", "-", ascii_hint).strip("-")
    if ascii_hint:
        return ascii_hint[:80]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"story-{digest}"


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def lexical_jaccard(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[0-9A-Za-z가-힣]+", (a or "").lower()))
    tokens_b = set(re.findall(r"[0-9A-Za-z가-힣]+", (b or "").lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def transition_cue_present(text: str) -> bool:
    lowered = (text or "").lower()
    cues = [
        "걸어", "이동", "도착", "향했", "향했다", "다다랐", "돌아왔", "내려갔", "올라갔",
        "moved", "arrived", "walked", "headed", "returned", "crossed",
    ]
    return any(cue in lowered for cue in cues)


def short_text(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> Any:
    """Best-effort extraction of the last top-level JSON object or array from text."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response")
    candidates: List[tuple[int, int, Any]] = []
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = 0
        while start < len(text):
            start = text.find(start_char, start)
            if start == -1:
                break
            depth = 0
            in_string = False
            escape = False
            matched_end = -1
            for idx, ch in enumerate(text[start:], start=start):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        matched_end = idx
                        candidate = text[start: idx + 1]
                        try:
                            candidates.append((start, idx + 1, json.loads(candidate)))
                        except json.JSONDecodeError:
                            pass
                        break
            start = matched_end + 1 if matched_end != -1 else start + 1
    if candidates:
        top_level: List[tuple[int, int, Any]] = []
        for candidate in candidates:
            start, end, payload = candidate
            if any(other_start <= start and end <= other_end and (other_start, other_end) != (start, end) for other_start, other_end, _ in candidates):
                continue
            top_level.append(candidate)
        top_level.sort(key=lambda item: (item[0], item[1]))
        return top_level[-1][2]
    raise ValueError(f"Could not extract JSON from: {short_text(text, 300)}")
