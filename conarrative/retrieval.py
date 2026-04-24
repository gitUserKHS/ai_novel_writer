from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")


def select_relevant_kg_edges(
    edges: Iterable[Dict[str, Any]],
    *,
    query_text: str = "",
    limit: int = 24,
) -> List[Dict[str, Any]]:
    """Return a small KG slice ranked by lexical overlap with a recency fallback."""

    edge_list = list(edges)
    if not edge_list:
        return []

    query_tokens = set(_tokens(query_text))
    scored = []
    for index, edge in enumerate(edge_list):
        text = " ".join(
            str(edge.get(key, ""))
            for key in ("source", "relation", "target", "edge_type", "scene_id")
        )
        score = len(query_tokens.intersection(_tokens(text))) if query_tokens else 0
        scored.append((score, index, edge))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = [edge for _, _, edge in scored[:limit]]
    return sorted(chosen, key=lambda edge: edge.get("id", 0))


def request_query_text(request: Any) -> str:
    if request is None:
        return ""
    if hasattr(request, "model_dump"):
        payload = request.model_dump()
    elif isinstance(request, dict):
        payload = request
    else:
        return str(request)

    parts: List[str] = []
    for key in ("title", "pov", "goal", "location", "time_label", "summary_request"):
        if payload.get(key):
            parts.append(str(payload[key]))
    for key in ("beats", "must_include", "must_avoid", "emotion_targets"):
        parts.extend(str(item) for item in payload.get(key, []) if item)
    return " ".join(parts)


def _tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]
