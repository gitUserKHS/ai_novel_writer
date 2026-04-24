from __future__ import annotations

import json
from typing import Any, Dict

from .models import AcceptedScene


def accepted_scene_to_multi_target_payload(scene: AcceptedScene) -> Dict[str, Any]:
    """Build a structured Narrative-MTP target without changing model heads."""

    plan = scene.plan.model_dump()
    extraction = scene.extraction.model_dump()
    return {
        "draft": scene.accepted_text,
        "summary": scene.summary,
        "state_delta": extraction.get("state_updates", {}),
        "new_threads": extraction.get("new_threads", []),
        "resolved_threads": extraction.get("resolved_threads", []),
        "kg_edges": extraction.get("kg_edges", []),
        "future_predictions": {
            "future_state_predictions": plan.get("future_state_predictions", []),
            "backward_prerequisites": plan.get("backward_prerequisites", []),
            "payoff_targets": plan.get("payoff_targets", []),
            "contradiction_risks": plan.get("contradiction_risks", []),
            "expected_reveals": plan.get("expected_reveals", []),
            "expected_state_delta": plan.get("expected_state_delta", {}),
        },
    }


def dump_multi_target_payload(scene: AcceptedScene) -> str:
    return json.dumps(accepted_scene_to_multi_target_payload(scene), ensure_ascii=False, indent=2)
