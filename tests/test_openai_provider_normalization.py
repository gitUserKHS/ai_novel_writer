from __future__ import annotations

from conarrative.llm import OllamaNativeProvider, OpenAICompatibleProvider, build_provider
from conarrative.models import RuntimeSettings, SceneRequest, Severity
from conarrative.runtime_settings import RuntimeSettingsStore
from conarrative.utils import extract_json_object


def make_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        RuntimeSettings(
            provider="openai_compatible",
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            model="gemma4:e4b",
            cache_responses=False,
        )
    )


def test_outline_normalization_accepts_alternative_keys() -> None:
    provider = make_provider()
    payload = {
        "plan_summary": "irrelevant",
        "next_scene_plan": [
            {
                "id": "scene_1_initial_discovery",
                "scene_index": 1,
                "title": "먼지 쌓인 무대와 낡은 사진",
                "pov": "서윤",
                "location": "무대 뒤편",
                "time_label": "늦은 오후",
                "goal": "동생의 흔적을 찾는다.",
                "beat": "극장에 들어와 첫 단서를 발견한다.",
                "foreshadowing": ["오케스트라 박스의 비밀"],
                "required_facts": ["극장은 폐관 위기다."],
            }
        ],
    }
    memory_bundle = {"story": {"id": "moon-theater"}}

    cards = provider._normalize_outline_cards(payload, memory_bundle, scene_count=1)

    assert len(cards) == 1
    assert cards[0].id == "scene_1_initial_discovery"
    assert cards[0].title == "먼지 쌓인 무대와 낡은 사진"
    assert cards[0].foreshadowing == ["오케스트라 박스의 비밀"]


def test_report_normalization_handles_loose_shapes() -> None:
    provider = make_provider()

    consistency = provider._normalize_consistency_report(
        {
            "consistency_score": 0.77,
            "plausibility": 0.72,
            "problems": [
                {
                    "type": "timeline_gap",
                    "level": "moderate",
                    "description": "시간 전환이 약하다.",
                    "suggestion": "시간 표식을 추가해라.",
                }
            ],
            "summary": "critic summary",
        }
    )
    creativity = provider._normalize_creativity_report(
        {
            "creativity_score": 0.66,
            "novelty": 0.7,
            "hook": 0.61,
            "emotion_score": 0.63,
            "style_score": 0.68,
        }
    )
    revision = provider._normalize_revision_output({"text": "수정본", "changes": ["장면 정리"]}, "원본")
    extraction = provider._normalize_extraction_output(
        {
            "scene_summary": "장면 요약",
            "state": {"current_location": "분장실"},
            "open_threads": ["숨겨진 문"],
            "edges": [{"subject": "서윤", "predicate": "finds", "object": "사진"}],
        },
        "본문",
    )
    plan = provider._normalize_plan_output(
        {
            "title": "분장실의 장면",
            "steps": ["들어간다", "찾는다", "의심한다"],
            "constraints": ["분장실", "D1 22:00"],
            "target_length": 500,
        },
        SceneRequest(location="분장실", min_words=320, max_words=680),
    )

    assert consistency.score == 0.77
    assert consistency.world_plausibility_score == 0.72
    assert consistency.issues[0].severity == Severity.MEDIUM
    assert creativity.novelty_score == 0.7
    assert revision.revised_text == "수정본"
    assert extraction.summary == "장면 요약"
    assert extraction.kg_edges[0].relation == "finds"
    assert plan.scene_title == "분장실의 장면"
    assert plan.must_include == ["분장실", "D1 22:00"]


def test_runtime_settings_store_merges_new_defaults(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text('{"provider":"openai_compatible","model":"qwen3:4b"}', encoding="utf-8")

    store = RuntimeSettingsStore(
        path,
        RuntimeSettings(
            provider="openai_compatible",
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            model="qwen3:4b",
            think=False,
            timeout_seconds=600,
        ),
    )

    loaded = store.load()

    assert loaded.model == "qwen3:4b"
    assert loaded.think is False
    assert loaded.timeout_seconds == 600


def test_extract_json_object_prefers_last_top_level_object() -> None:
    text = 'Thinking about {"prompt":"echo"} before final.\n</think>\n{"ok": true, "items": [1, 2]}'

    payload = extract_json_object(text)

    assert payload == {"ok": True, "items": [1, 2]}


def test_build_provider_supports_ollama_native() -> None:
    provider = build_provider(
        RuntimeSettings(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            api_key="ollama",
            model="qwen3:4b",
            think=False,
        )
    )

    assert isinstance(provider, OllamaNativeProvider)
