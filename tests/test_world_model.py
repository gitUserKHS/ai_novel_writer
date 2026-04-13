from __future__ import annotations

from conarrative.models import SceneRequest, WorldModelForecast, WorldModelScore
from conarrative.world_model import NarrativeWorldModel


def test_blend_forecast_adds_learned_world_model_details() -> None:
    model = NarrativeWorldModel()
    base_score = WorldModelScore(plausibility=0.7, novelty=0.5, surprise=0.59)
    memory_bundle = {
        "state": {
            "current_location": "Lobby",
            "current_time_label": "D1 19:00",
            "active_threads": [],
            "resolved_threads": [],
            "character_knowledge": {},
            "inventory": {},
            "emotional_state": {},
            "summary_memory": [],
        }
    }
    request = SceneRequest(
        location="Archive",
        time_label="D1 20:00",
        foreshadowing=["red key"],
    )
    forecast = WorldModelForecast(
        next_state={
            "current_location": "Archive",
            "current_time_label": "D1 20:00",
            "active_threads": ["red key"],
            "emotional_state": {"Seoyun": "tense"},
        },
        extraction={"summary": "Seoyun reaches the archive."},
        notes=["forecast ok"],
    )

    blended = model.blend_forecast(memory_bundle, request, base_score, forecast)

    assert blended.plausibility > base_score.plausibility
    assert blended.details["learned_world_model"]["predicted_next_state"]["current_location"] == "Archive"
    assert blended.details["learned_world_model"]["notes"] == ["forecast ok"]
