from __future__ import annotations

from scripts.train_qlora import build_sft_rows


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ANN001
        return "\n".join(f"{item['role']}::{item['content']}" for item in messages)


def test_build_sft_rows_world_model_format() -> None:
    rows = [
        {
            "story_id": "story-a",
            "scene_id": "story-a-sc001",
            "scene_index": 1,
            "request": {"goal": "find the trace"},
            "plan": {"scene_title": "First turn"},
            "previous_state": {"current_location": "lobby"},
            "next_state": {"current_location": "archive"},
            "accepted_text": "The archivist enters the archive.",
            "extraction": {"summary": "The archivist moves to the archive."},
        }
    ]

    formatted = build_sft_rows(rows, DummyTokenizer(), dataset_format="world_model")

    assert len(formatted) == 1
    assert "system::You are a narrative world model." in formatted[0]["text"]
    assert "\"previous_state\": {\"current_time_label\": \"\", \"current_location\": \"lobby\"" in formatted[0]["text"]
    assert "\"next_state\": {\"current_location\": \"archive\"}" in formatted[0]["text"]
    assert "\"schema\": {\"next_state\": {}" in formatted[0]["text"]
