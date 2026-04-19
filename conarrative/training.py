from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import PoolType

NOVELIST_SYSTEM_PROMPT = (
    "당신은 한국어 장편소설용 장면을 쓰는 작가다. "
    "항상 자연스러운 한국어로만 답하고, 사용자의 요구와 기억 문맥을 강하게 따른다. "
    "설명, 메타 발화, 제목, 주석 없이 장면 본문만 작성한다."
)


def build_training_user_prompt(
    request: Dict[str, Any],
    memory_snapshot: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> str:
    sections: List[str] = [
        "다음 정보를 바탕으로 한국어 소설 장면 하나를 작성하세요.",
        "",
        "장면 요청",
        _render_request(request),
    ]
    if memory_snapshot:
        sections.extend(["", "기억 문맥", _render_memory_snapshot(memory_snapshot)])
    if plan:
        sections.extend(["", "장면 계획", _render_plan(plan)])
    sections.extend(
        [
            "",
            "지침",
            "- 출력은 자연스러운 한국어 장면 본문만 작성합니다.",
            "- POV, 시간, 장소, 목표, 감정선, 포함/금지 조건을 지킵니다.",
            "- 개연성과 인물 지식 상태를 유지합니다.",
            "- 불필요한 영어 문장, 메타 설명, 소제목을 넣지 않습니다.",
        ]
    )
    return "\n".join(sections).strip()


def export_training_corpus(records: Iterable[Dict[str, Any]], out_dir: str | Path) -> Dict[str, Any]:
    rows = list(records)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    prompt_only_index = _build_prompt_only_index(rows)
    accepted_sft: List[Dict[str, Any]] = []
    prompt_only_teacher: List[Dict[str, Any]] = []
    pairwise_dpo: List[Dict[str, Any]] = []
    hard_negative: List[Dict[str, Any]] = []

    for row in rows:
        pool_type = row["pool_type"]
        payload = row["payload"]

        if pool_type == PoolType.PROMPT_ONLY.value:
            prompt_only_teacher.append(_prompt_only_to_teacher_sample(row))
        elif pool_type == PoolType.ACCEPTED.value:
            accepted_sft.append(_accepted_to_sft_sample(row, prompt_only_index))
        elif pool_type == PoolType.PAIRWISE.value:
            pairwise_dpo.append(_pairwise_to_dpo_sample(row, prompt_only_index))
        elif pool_type == PoolType.HARD_NEGATIVE.value:
            hard_negative.append(_hard_negative_sample(row, prompt_only_index))
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")

    files = {
        "accepted_sft": target / "accepted_sft.jsonl",
        "prompt_only_teacher": target / "prompt_only_teacher.jsonl",
        "pairwise_dpo": target / "pairwise_dpo.jsonl",
        "hard_negative": target / "hard_negative.jsonl",
    }
    _write_jsonl(files["accepted_sft"], accepted_sft)
    _write_jsonl(files["prompt_only_teacher"], prompt_only_teacher)
    _write_jsonl(files["pairwise_dpo"], pairwise_dpo)
    _write_jsonl(files["hard_negative"], hard_negative)

    manifest = {
        "counts": {
            "accepted_sft": len(accepted_sft),
            "prompt_only_teacher": len(prompt_only_teacher),
            "pairwise_dpo": len(pairwise_dpo),
            "hard_negative": len(hard_negative),
        },
        "files": {name: str(path) for name, path in files.items()},
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def _accepted_to_sft_sample(
    row: Dict[str, Any],
    prompt_only_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    payload = row["payload"]
    accepted_scene = payload["accepted_scene"]
    scene_index = int(accepted_scene["scene_index"])
    prompt_only = prompt_only_index.get((row["story_id"], scene_index), {})
    memory_snapshot = prompt_only.get("memory_snapshot")
    request = payload["request"]
    plan = payload.get("plan")

    return {
        "messages": [
            {"role": "system", "content": NOVELIST_SYSTEM_PROMPT},
            {"role": "user", "content": build_training_user_prompt(request, memory_snapshot=memory_snapshot, plan=plan)},
            {"role": "assistant", "content": accepted_scene["accepted_text"]},
        ],
        "metadata": {
            "story_id": row["story_id"],
            "scene_id": row["scene_id"],
            "scene_index": scene_index,
            "pool_type": row["pool_type"],
            "source": "accepted",
        },
    }


def _prompt_only_to_teacher_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row["payload"]
    request = payload["request"]
    memory_snapshot = payload.get("memory_snapshot")
    scene_index = int(payload.get("scene_index", 0))
    return {
        "messages": [
            {"role": "system", "content": NOVELIST_SYSTEM_PROMPT},
            {"role": "user", "content": build_training_user_prompt(request, memory_snapshot=memory_snapshot)},
        ],
        "metadata": {
            "story_id": row["story_id"],
            "scene_id": row["scene_id"],
            "scene_index": scene_index,
            "pool_type": row["pool_type"],
            "source": "prompt_only",
        },
    }


def _pairwise_to_dpo_sample(
    row: Dict[str, Any],
    prompt_only_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    payload = row["payload"]
    request = payload["request"]
    scene_index = int(request.get("scene_index") or 0)
    prompt_only = prompt_only_index.get((row["story_id"], scene_index), {})
    prompt = build_training_user_prompt(request, memory_snapshot=prompt_only.get("memory_snapshot"))
    return {
        "prompt": prompt,
        "chosen": payload["accepted_text"],
        "rejected": payload["rejected_text"],
        "metadata": {
            "story_id": row["story_id"],
            "scene_id": row["scene_id"],
            "scene_index": scene_index,
            "pool_type": row["pool_type"],
            "accepted_score": payload.get("accepted_score"),
            "rejected_score": payload.get("rejected_score"),
        },
    }


def _hard_negative_sample(
    row: Dict[str, Any],
    prompt_only_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    payload = row["payload"]
    request = payload["request"]
    scene_index = int(request.get("scene_index") or 0)
    prompt_only = prompt_only_index.get((row["story_id"], scene_index), {})
    return {
        "messages": [
            {"role": "system", "content": NOVELIST_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_training_user_prompt(
                    request,
                    memory_snapshot=prompt_only.get("memory_snapshot"),
                    plan=payload.get("plan"),
                ),
            },
            {"role": "assistant", "content": payload["text"]},
        ],
        "issues": payload.get("issues", []),
        "metadata": {
            "story_id": row["story_id"],
            "scene_id": row["scene_id"],
            "scene_index": scene_index,
            "pool_type": row["pool_type"],
        },
    }


def _build_prompt_only_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        if row["pool_type"] != PoolType.PROMPT_ONLY.value:
            continue
        payload = row["payload"]
        scene_index = int(payload.get("scene_index", 0))
        index[(row["story_id"], scene_index)] = payload
    return index


def _render_request(request: Dict[str, Any]) -> str:
    lines = [
        f"- 제목: {request.get('title', '')}",
        f"- POV: {request.get('pov', '')}",
        f"- 목표: {request.get('goal', '')}",
        f"- 장소: {request.get('location', '')}",
        f"- 시간: {request.get('time_label', '')}",
        f"- 요약 요청: {request.get('summary_request', '')}",
        f"- 목표 분량(단어): {request.get('desired_length_words', '')}",
    ]
    if request.get("beats"):
        lines.append("- 비트: " + " | ".join(request["beats"]))
    if request.get("must_include"):
        lines.append("- 반드시 포함: " + " | ".join(request["must_include"]))
    if request.get("must_avoid"):
        lines.append("- 금지: " + " | ".join(request["must_avoid"]))
    if request.get("emotion_targets"):
        lines.append("- 감정 목표: " + " | ".join(request["emotion_targets"]))
    return "\n".join(lines)


def _render_plan(plan: Dict[str, Any]) -> str:
    lines = [
        f"- 장면 제목: {plan.get('scene_title', '')}",
        f"- 시놉시스: {plan.get('synopsis', '')}",
    ]
    beats = plan.get("beats") or []
    if beats:
        lines.append("- 계획 비트: " + " | ".join(f"{beat.get('label', '')}:{beat.get('purpose', '')}" for beat in beats))
    if plan.get("expected_reveals"):
        lines.append("- 예상 드러남: " + " | ".join(plan["expected_reveals"]))
    if plan.get("expected_new_threads"):
        lines.append("- 새 떡밥: " + " | ".join(plan["expected_new_threads"]))
    if plan.get("expected_resolved_threads"):
        lines.append("- 회수 떡밥: " + " | ".join(plan["expected_resolved_threads"]))
    return "\n".join(lines)


def _render_memory_snapshot(snapshot: Dict[str, Any]) -> str:
    state = snapshot.get("state", {})
    bible = snapshot.get("bible", {})
    recent = snapshot.get("recent_scenes", [])
    lines: List[str] = []
    if bible.get("static_facts"):
        lines.append("- 고정 사실: " + " | ".join(bible["static_facts"]))
    if bible.get("rules"):
        lines.append("- 규칙: " + " | ".join(bible["rules"]))
    if bible.get("motifs"):
        lines.append("- 모티프: " + " | ".join(bible["motifs"]))
    if bible.get("voice_notes"):
        lines.append("- 문체 메모: " + " | ".join(bible["voice_notes"]))
    if state.get("current_time_label"):
        lines.append(f"- 현재 시간: {state['current_time_label']}")
    if state.get("current_location"):
        lines.append(f"- 현재 장소: {state['current_location']}")
    if state.get("active_threads"):
        lines.append("- 열린 떡밥: " + " | ".join(state["active_threads"]))
    if state.get("resolved_threads"):
        lines.append("- 회수된 떡밥: " + " | ".join(state["resolved_threads"]))
    if recent:
        lines.append("- 최근 장면 요약:")
        for scene in recent[-3:]:
            lines.append(
                f"  - Scene {scene.get('scene_index', '')} / {scene.get('time_label', '')} / {scene.get('location', '')}: {scene.get('summary', '')}"
            )
    return "\n".join(lines) if lines else "- 별도 기억 정보 없음"


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
