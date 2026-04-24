from __future__ import annotations

import re

from .models import ContinueStoryRequest, OutlineCard, ProviderType, QuickstartRequest, RuntimeSettings, SceneRequest, StoryCreate
from .utils.text import clip_text, normalize_list


GENRE_HINTS = [
    ("murder", "mystery thriller"),
    ("ghost", "supernatural drama"),
    ("memory", "literary suspense"),
    ("robot", "science fiction drama"),
    ("future", "science fiction drama"),
    ("detective", "mystery thriller"),
    ("crime", "crime drama"),
    ("love", "romantic drama"),
    ("family", "family drama"),
    ("war", "historical drama"),
    ("school", "coming-of-age drama"),
    ("theater", "mystery drama"),
    ("magic", "fantasy drama"),
    ("haunted", "supernatural drama"),
    ("mystery", "mystery drama"),
]

TONE_HINTS = [
    ("haunted", "haunting and intimate"),
    ("ghost", "uncanny and atmospheric"),
    ("murder", "tense and cinematic"),
    ("crime", "sharp and suspenseful"),
    ("love", "tender and emotionally direct"),
    ("family", "warm and emotionally grounded"),
    ("future", "speculative and moody"),
    ("magic", "lyrical and vivid"),
    ("school", "youthful and bittersweet"),
]

THEME_HINTS = [
    ("memory", "memory"),
    ("ghost", "grief"),
    ("family", "family"),
    ("love", "love"),
    ("crime", "truth"),
    ("murder", "guilt"),
    ("future", "identity"),
    ("war", "survival"),
    ("school", "growing up"),
    ("theater", "performance"),
]


def quickstart_settings(settings: RuntimeSettings) -> tuple[RuntimeSettings, str]:
    if settings.provider == ProviderType.OPENAI_COMPATIBLE:
        fast_settings = settings.model_copy(update={"candidate_count": 1}) if settings.candidate_count > 1 else settings
        return (
            fast_settings,
            f"Trying your local model first: {settings.model}. Quickstart uses 1 draft candidate for speed; if it does not answer, the built-in storyteller fills in.",
        )
    return settings, "Using the built-in storyteller. No model setup is required."


def build_story_from_prompt(request: QuickstartRequest) -> StoryCreate:
    prompt = re.sub(r"\s+", " ", request.prompt).strip()
    title = _derive_title(prompt)
    return StoryCreate(
        title=title,
        genre=_match_hint(prompt, GENRE_HINTS, "speculative drama"),
        premise=prompt,
        tone=_match_hint(prompt, TONE_HINTS, "cinematic and emotionally grounded"),
        themes=_extract_themes(prompt),
        characters=_extract_characters(prompt),
        notes="Created from quickstart prompt.",
        target_length_scenes=max(3, request.scene_count),
    )


def outline_to_scene_request(card: OutlineCard, desired_length_words: int) -> SceneRequest:
    return SceneRequest(
        title=card.title,
        pov=card.pov,
        goal=card.goal,
        location=card.location,
        time_label=card.time_label,
        summary_request=card.summary_request,
        beats=card.beats,
        must_include=card.must_include,
        must_avoid=card.must_avoid,
        desired_length_words=desired_length_words,
        outline_card_id=card.id,
    )


def continue_request_to_words(request: ContinueStoryRequest) -> int:
    return request.desired_length_words


def next_planned_outline_card(cards: list[OutlineCard]) -> OutlineCard | None:
    for card in cards:
        if card.status != "used":
            return card
    return None


def _derive_title(prompt: str) -> str:
    first_clause = re.split(r"[.!?\n]+", prompt, maxsplit=1)[0].strip(" -:;,")
    if not first_clause:
        return "New Story"
    words = first_clause.split()
    if len(words) <= 5:
        candidate = first_clause
    else:
        candidate = " ".join(words[:5])
    candidate = candidate.strip()
    if not candidate:
        return "New Story"
    if len(candidate) > 48:
        candidate = clip_text(candidate, 48)
    return candidate


def _extract_characters(prompt: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
    if len(matches) >= 2:
        return normalize_list(matches[:4])
    return ["Protagonist", "Counterpart"]


def _extract_themes(prompt: str) -> list[str]:
    themes = [theme for keyword, theme in THEME_HINTS if keyword in prompt.lower()]
    return normalize_list(themes[:4])


def _match_hint(prompt: str, hints: list[tuple[str, str]], fallback: str) -> str:
    lowered = prompt.lower()
    for keyword, value in hints:
        if keyword in lowered:
            return value
    return fallback
