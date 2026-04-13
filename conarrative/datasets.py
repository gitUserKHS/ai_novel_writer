from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .db import Storage
from .utils import ensure_dir
from .models import PoolType, utcnow_iso


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _critic_label(report: Dict[str, Any]) -> str:
    issues = report.get("issues", []) or []
    severities = {str(issue.get("severity", "low")).lower() for issue in issues if isinstance(issue, dict)}
    if "high" in severities:
        return "hard_fail"
    if "medium" in severities:
        return "soft_fail"
    return "pass"


def _critic_messages(request: Dict[str, Any], plan: Dict[str, Any], text: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a scene continuity critic. Return only structured JSON describing continuity, rule, state-transition, and required-fact issues.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": request,
                        "plan": plan,
                        "text": text,
                        "rubric": [
                            "location and time anchors",
                            "required facts and must-include items",
                            "goal and beat alignment",
                            "scene transition cues",
                            "thread continuity",
                            "world-rule compliance",
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": json.dumps(report, ensure_ascii=False)},
        ]
    }


def export_training_bundle(storage: Storage, story_id: str, output_dir: str | Path) -> Dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    story = storage.get_story(story_id)
    if story is None:
        raise ValueError(f"Story not found: {story_id}")
    scenes = storage.list_scenes(story_id)
    dataset_records = storage.list_dataset_records(story_id, limit=100_000)
    snapshots = storage.list_state_snapshots(story_id)

    accepted_rows = []
    distill_rows = []
    world_rows = []
    pairwise_rows = []
    critic_rows = []
    critic_sft_rows = []

    previous_state = snapshots[0] if snapshots else None
    snapshot_by_scene = {snap.get("scene_id"): snap for snap in snapshots}

    for scene in scenes:
        scene_detail = storage.get_scene(scene["id"]) or scene
        request = scene.get("input", {})
        plan = scene.get("plan", {})
        accepted_text = scene.get("accepted_text", "")
        accepted_rows.append(
            {
                "messages": [
                    {"role": "system", "content": "너는 장면 단위 장편 소설 작성 모델이다."},
                    {"role": "user", "content": json.dumps({"request": request, "plan": plan}, ensure_ascii=False)},
                    {"role": "assistant", "content": accepted_text},
                ],
                "metadata": {"story_id": story_id, "scene_id": scene["id"], "scene_index": scene["scene_index"]},
            }
        )
        distill_rows.append(
            {
                "prompt": json.dumps({"request": request, "plan": plan}, ensure_ascii=False),
                "teacher_trace": {
                    "plan_reasoning": plan.get("reasoning", []),
                    "consistency": scene.get("consistency", {}),
                    "creativity": scene.get("creativity", {}),
                    "revision": scene.get("revision", {}),
                },
                "completion": accepted_text,
                "metadata": {"story_id": story_id, "scene_id": scene["id"]},
            }
        )
        next_state = snapshot_by_scene.get(scene["id"]) or {}
        world_rows.append(
            {
                "story_id": story_id,
                "scene_id": scene["id"],
                "scene_index": scene["scene_index"],
                "request": request,
                "plan": plan,
                "accepted_text": accepted_text,
                "previous_state": previous_state,
                "next_state": next_state,
                "extraction": scene.get("extraction", {}),
            }
        )
        previous_state = next_state

        seen_texts: set[str] = set()
        candidate_reports = [
            {
                "text": accepted_text,
                "report": scene.get("consistency", {}) or {},
                "accepted": True,
            }
        ]
        for candidate in scene_detail.get("candidates", []):
            candidate_reports.append(
                {
                    "text": candidate.get("text", ""),
                    "report": candidate.get("consistency", {}) or {},
                    "accepted": bool(candidate.get("accepted")),
                }
            )
        for item in candidate_reports:
            text = str(item.get("text", "")).strip()
            report = item.get("report", {}) if isinstance(item.get("report"), dict) else {}
            if not text or text in seen_texts or not report:
                continue
            seen_texts.add(text)
            critic_sft_rows.append(
                {
                    **_critic_messages(request, plan, text, report),
                    "metadata": {
                        "story_id": story_id,
                        "scene_id": scene["id"],
                        "scene_index": scene["scene_index"],
                        "label": _critic_label(report),
                        "accepted": bool(item.get("accepted")),
                    },
                }
            )

    for record in dataset_records:
        payload = record.get("payload", {})
        if record.get("pool_type") == PoolType.PAIRWISE.value:
            pairwise_rows.append(
                {
                    "prompt": json.dumps(payload.get("request", {}), ensure_ascii=False),
                    "chosen": payload.get("accepted_text", ""),
                    "rejected": payload.get("rejected_text", ""),
                    "metadata": {"story_id": story_id, "scene_id": record.get("scene_id"), "created_at": record.get("created_at")},
                }
            )
        if record.get("pool_type") == PoolType.HARD_NEGATIVE.value:
            critic_rows.append(
                {
                    "text": payload.get("text", ""),
                    "issues": payload.get("issues", []),
                    "request": payload.get("request", {}),
                    "plan": payload.get("plan", {}),
                    "label": "hard_negative",
                    "metadata": {"story_id": story_id, "scene_id": record.get("scene_id")},
                }
            )

    files = {
        "writer_sft": out_dir / f"{story_id}_writer_sft.jsonl",
        "writer_dpo": out_dir / f"{story_id}_writer_dpo.jsonl",
        "distill_stepwise": out_dir / f"{story_id}_distill_stepwise.jsonl",
        "critic_hard_negative": out_dir / f"{story_id}_critic_hard_negative.jsonl",
        "critic_consistency_sft": out_dir / f"{story_id}_critic_consistency_sft.jsonl",
        "world_model_transitions": out_dir / f"{story_id}_world_model_transitions.jsonl",
        "manifest": out_dir / f"{story_id}_training_manifest.json",
    }
    counts = {
        "writer_sft": _write_jsonl(files["writer_sft"], accepted_rows),
        "writer_dpo": _write_jsonl(files["writer_dpo"], pairwise_rows),
        "distill_stepwise": _write_jsonl(files["distill_stepwise"], distill_rows),
        "critic_hard_negative": _write_jsonl(files["critic_hard_negative"], critic_rows),
        "critic_consistency_sft": _write_jsonl(files["critic_consistency_sft"], critic_sft_rows),
        "world_model_transitions": _write_jsonl(files["world_model_transitions"], world_rows),
    }
    manifest = {
        "story_id": story_id,
        "generated_at": utcnow_iso(),
        "counts": counts,
        "files": {key: str(path) for key, path in files.items() if key != "manifest"},
    }
    files["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact = storage.save_artifact(story_id, "training_bundle", str(files["manifest"]), manifest)
    return {"manifest": manifest, "artifact": artifact.model_dump(), "paths": {key: str(path) for key, path in files.items()}}
