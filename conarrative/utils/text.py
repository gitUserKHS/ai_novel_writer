from __future__ import annotations

import hashlib
import re
from typing import Iterable, List


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def normalize_list(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in values:
        value = (item or "").strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def clip_text(text: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def stable_seed(*parts: str) -> int:
    joined = "||".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)


def has_batchim(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    ch = text[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False


def korean_particle(text: str, pair: str) -> str:
    left, right = pair.split("/")
    return left if has_batchim(text) else right


def join_korean_and(items: Iterable[str]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        connector = "과" if has_batchim(values[0]) else "와"
        return f"{values[0]}{connector} {values[1]}"
    return ", ".join(values[:-1]) + f" 그리고 {values[-1]}"
