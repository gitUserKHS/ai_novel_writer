from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import httpx

from .models import AutoConnectOut, LocalModelCatalogOut, LocalModelOption, ProviderType, RuntimeSettings


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
    catalog = discover_runtime_catalog(current)
    if catalog.options:
        current_option = catalog.current or catalog.options[0]
        settings = RuntimeSettings.model_validate(
            {
                **current.model_dump(),
                "provider": ProviderType.OPENAI_COMPATIBLE,
                "base_url": current_option.base_url,
                "model": current_option.model,
                "api_key": current.api_key or "not-needed",
            }
        )
        return AutoConnectOut(
            found=True,
            source=current_option.source,
            detail=catalog.detail or f"{current_option.source} detected at {current_option.base_url} with model {current_option.model}.",
            settings=settings,
            available_models=[option.model for option in catalog.options if option.base_url == current_option.base_url],
        )
    return AutoConnectOut(found=False, detail=catalog.detail)


def discover_runtime_catalog(current: RuntimeSettings) -> LocalModelCatalogOut:
    timeout = min(max(current.timeout_seconds, 1), 2)
    options: list[LocalModelOption] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            for candidate in CANDIDATE_ENDPOINTS:
                models = _fetch_openai_models(client, candidate.models_url)
                if not models and candidate.tags_url:
                    models = _fetch_ollama_models(client, candidate.tags_url)
                if not models:
                    continue
                for model in models:
                    options.append(
                        LocalModelOption(
                            source=candidate.source,
                            base_url=candidate.base_url,
                            model=model,
                        )
                    )
        normalized = _normalize_options(options)
        if not normalized:
            return LocalModelCatalogOut(
                options=[],
                current=None,
                detail="No local model server was detected. Start Ollama or LM Studio, then reopen the app or refresh.",
            )
        current_option = _pick_current_option(normalized, current)
        if current_option is None:
            current_option = _pick_preferred_option(normalized)
        return LocalModelCatalogOut(
            options=normalized,
            current=current_option,
            detail=f"Detected {len(normalized)} local model option(s).",
        )
    except Exception as exc:
        return LocalModelCatalogOut(
            options=[],
            current=None,
            detail=f"Local model scan failed: {exc}",
        )


def auto_connect_settings(current: RuntimeSettings) -> tuple[RuntimeSettings, LocalModelCatalogOut, bool]:
    catalog = discover_runtime_catalog(current)
    if not catalog.options:
        return current, catalog, False
    current_option = _pick_current_option(catalog.options, current)
    if current.provider == ProviderType.OPENAI_COMPATIBLE and current_option is not None:
        return current, catalog, False
    chosen = catalog.current or _pick_preferred_option(catalog.options)
    settings = RuntimeSettings.model_validate(
        {
            **current.model_dump(),
            "provider": ProviderType.OPENAI_COMPATIBLE,
            "base_url": chosen.base_url,
            "model": chosen.model,
            "api_key": current.api_key or "not-needed",
        }
    )
    current_catalog = LocalModelCatalogOut(
        options=catalog.options,
        current=chosen,
        detail=f"{chosen.source} auto-connected with model {chosen.model}.",
    )
    return settings, current_catalog, True


def build_catalog_from_settings(current: RuntimeSettings, catalog: LocalModelCatalogOut | None = None) -> LocalModelCatalogOut:
    resolved = catalog or discover_runtime_catalog(current)
    if not resolved.options:
        return resolved
    current_option = _pick_current_option(resolved.options, current)
    if current_option is None and current.provider == ProviderType.OPENAI_COMPATIBLE:
        current_option = LocalModelOption(
            source="Custom OpenAI-compatible",
            base_url=current.base_url,
            model=current.model,
        )
        return LocalModelCatalogOut(
            options=[current_option] + resolved.options,
            current=current_option,
            detail=resolved.detail,
        )
    return LocalModelCatalogOut(
        options=resolved.options,
        current=current_option or resolved.current or _pick_preferred_option(resolved.options),
        detail=resolved.detail,
    )


def _normalize_options(items: Iterable[LocalModelOption]) -> List[LocalModelOption]:
    seen: set[tuple[str, str, str]] = set()
    out: list[LocalModelOption] = []
    for item in items:
        key = (item.source, item.base_url, item.model)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _pick_current_option(options: List[LocalModelOption], current: RuntimeSettings) -> LocalModelOption | None:
    if current.provider != ProviderType.OPENAI_COMPATIBLE:
        return None
    for option in options:
        if option.base_url == current.base_url and option.model == current.model:
            return option
    return None


def _pick_preferred_option(options: List[LocalModelOption]) -> LocalModelOption:
    preferred = [option for option in options if not _looks_like_utility_model(option.model)]
    ranked = preferred or options
    return ranked[0]


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


def _looks_like_utility_model(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("embed", "embedding", "rerank", "bge", "nomic-embed"))
