
from __future__ import annotations

import json
from typing import Any, Dict, List

from .utils import short_text


def compact_memory(memory_bundle: Dict[str, Any]) -> Dict[str, Any]:
    recent = []
    for scene in memory_bundle.get("recent_scenes", [])[-3:]:
        recent.append(
            {
                "scene_index": scene.get("scene_index"),
                "title": scene.get("title"),
                "summary": short_text(scene.get("summary", ""), 200),
                "location": scene.get("location"),
                "time_label": scene.get("time_label"),
            }
        )
    return {
        "story": memory_bundle.get("story", {}),
        "bible": memory_bundle.get("bible", {}),
        "state": memory_bundle.get("state", {}),
        "recent_scenes": recent,
    }


def make_json_messages(system: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
