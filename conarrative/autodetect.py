from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import httpx

from .models import AutoConnectOut, ProviderType, RuntimeSettings


@dataclass(frozen=True)
class CandidateEndpoint:
    source: str
    base_url: str
    models_url: str
    tags_url: str | None = None


CANDIDATE_ENDPOINTS: tuple[CandidateEndpoint, ...] = (
    CandidateEndpoint(
        source="Ollama",
        base_url="http://127.0.0.1:11434/v1",
        models_url="http://127.0.0.1:11434/v1/models",
        tags_url="http://127.0.0.1:11434/api/tags",
    ),
    CandidateEndpoint(
        source="LM Studio",
        base_url="http://127.0.0.1:1234/v1",
        models_url="http://127.0.0.1:1234/v1/models",
    ),
    CandidateEndpoint(
        source="OpenAI-compatible local server",
        base_url="http://127.0.0.1:8080/v1",
        models_url="http://127.0.0.1:8080/v1/models",
    ),
    CandidateEndpoint(
        source="OpenAI-compatible local server",
        base_url="http://127.0.0.1:5001/v1",
        models_url="http://127.0.0.1:5001/v1/models",
    ),
)


def detect_runtime_settings(current: RuntimeSettings) -> AutoConnectOut:
    timeout = min(max(current.timeout_seconds, 1), 2)
    try:
        with httpx.Client(timeout=timeout) as client:
            for candidate in CANDIDATE_ENDPOINTS:
                models = _fetch_openai_models(client, candidate.models_url)
                if not models and candidate.tags_url:
                    models = _fetch_ollama_models(client, candidate.tags_url)
                if not models:
                    continue
                chosen_model = _pick_preferred_model(models)
                settings = RuntimeSettings.model_validate(
                    {
                        **current.model_dump(),
                        "provider": ProviderType.OPENAI_COMPATIBLE,
                        "base_url": candidate.base_url,
                        "model": chosen_model,
                        "api_key": current.api_key or "not-needed",
                    }
                )
                return AutoConnectOut(
                    found=True,
                    source=candidate.source,
                    detail=f"{candidate.source} detected at {candidate.base_url} with model {chosen_model}.",
                    settings=settings,
                    available_models=models,
                )
    except Exception as exc:
        return AutoConnectOut(
            found=False,
            detail=f"Auto-connect failed while scanning local ports: {exc}",
        )
    return AutoConnectOut(
        found=False,
        detail="No local model server was detected. Start Ollama or LM Studio, then click auto-connect again.",
    )


def _fetch_openai_models(client: httpx.Client, models_url: str) -> List[str]:
    try:
        response = client.get(models_url)
        response.raise_for_status()
    except Exception:
        return []
    payload = response.json()
    items = payload.get("data", []) if isinstance(payload, dict) else []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("id"):
            names.append(str(item["id"]).strip())
    return _normalize_model_names(names)


def _fetch_ollama_models(client: httpx.Client, tags_url: str) -> List[str]:
    try:
        response = client.get(tags_url)
        response.raise_for_status()
    except Exception:
        return []
    payload = response.json()
    items = payload.get("models", []) if isinstance(payload, dict) else []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]).strip())
    return _normalize_model_names(names)


def _normalize_model_names(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _pick_preferred_model(models: List[str]) -> str:
    if not models:
        return "local-model"
    preferred = [name for name in models if not _looks_like_utility_model(name)]
    return (preferred or models)[0]


def _looks_like_utility_model(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("embed", "embedding", "rerank", "bge", "nomic-embed"))
