from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from .models import BibleContent, ConsistencyIssue, PlanOutput, SceneRequest, Severity, StoryState, WorldModelForecast, WorldModelScore
from .utils import clamp, lexical_jaccard, normalize_list, transition_cue_present


class NarrativeWorldModel:
    """Lightweight abstract-state scorer for narrative transitions.

    This is not a generative JEPA. It is a symbolic / heuristic state-transition model inspired by
    abstract latent prediction work: we rank a draft by whether its *implied next state* is plausible,
    novel, and causally connected to the existing story state.
    """

    item_words = ["열쇠", "편지", "메모", "칼", "목걸이", "사진", "서류", "지도", "반지"]

    def score(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> WorldModelScore:
        bible = BibleContent(**memory_bundle.get("bible", {})) if isinstance(memory_bundle.get("bible"), dict) else memory_bundle.get("bible", BibleContent())
        state = StoryState(**memory_bundle.get("state", {})) if isinstance(memory_bundle.get("state"), dict) else memory_bundle.get("state", StoryState())
        recent_summaries = [scene.get("summary", "") for scene in memory_bundle.get("recent_scenes", [])]

        issues: List[ConsistencyIssue] = []
        plausibility = 0.84
        timeline = 0.88
        knowledge = 0.86
        constraint = 0.9

        if request.location and request.location not in text:
            issues.append(
                ConsistencyIssue(
                    issue_type="world_weak_location_anchor",
                    severity=Severity.LOW,
                    message="장면이 지정된 장소에 충분히 고정되지 않았다.",
                    evidence_span=request.location,
                    suggested_fix="도입부나 행동선에 장소 단서를 더 넣어라.",
                )
            )
            plausibility -= 0.05

        previous_location = (state.current_location or "").strip()
        if previous_location and request.location and previous_location != request.location and not transition_cue_present(text):
            issues.append(
                ConsistencyIssue(
                    issue_type="world_transition_gap",
                    severity=Severity.MEDIUM,
                    message="직전 장면과 현재 장면 사이의 이동 단서가 약하다.",
                    evidence_span=f"{previous_location} -> {request.location}",
                    suggested_fix="걷기, 이동, 도착 같은 짧은 전환 문장을 추가해라.",
                )
            )
            plausibility -= 0.10
            timeline -= 0.08

        if request.time_label and request.time_label not in text:
            timeline -= 0.04

        forbidden_hits = [rule for rule in normalize_list(bible.rules + bible.forbidden) if rule and rule in text]
        for hit in forbidden_hits:
            issues.append(
                ConsistencyIssue(
                    issue_type="world_forbidden_rule",
                    severity=Severity.HIGH,
                    message="금지 규칙과 직접 충돌하는 표현이 등장했다.",
                    evidence_span=hit,
                    suggested_fix="해당 설정을 삭제하거나 규칙 안에서 다시 설명해라.",
                )
            )
            plausibility -= 0.20
            constraint -= 0.22

        state_change_size = 0
        new_items = [item for item in self.item_words if item in text]
        state_change_size += len(new_items)
        state_change_size += len(request.foreshadowing)
        state_change_size += len(request.required_facts)
        if state_change_size >= 6:
            issues.append(
                ConsistencyIssue(
                    issue_type="world_overcompressed_delta",
                    severity=Severity.MEDIUM,
                    message="한 장면에 너무 많은 상태 변화가 몰려 있어 압축감이 강하다.",
                    evidence_span=f"delta_size={state_change_size}",
                    suggested_fix="단서 하나를 다음 장면으로 미루거나 감정 처리 분량을 늘려라.",
                )
            )
            plausibility -= 0.08
            timeline -= 0.03

        # Approximate theory-of-mind / knowledge stability.
        known_facts = " ".join(sum(state.character_knowledge.values(), []))
        if known_facts and any(phrase in text for phrase in ["이미 알고", "처음부터 알", "당연하다는 듯"]) and request.pov not in text:
            knowledge -= 0.08
        if request.required_facts and not any(fact in text for fact in request.required_facts[:1]):
            knowledge -= 0.06

        lexical_overlap = 0.0
        if recent_summaries:
            lexical_overlap = max(lexical_jaccard(text, summary) for summary in recent_summaries if summary)
        novelty = 0.45 + 0.25 * (1.0 - lexical_overlap)
        if request.location and request.location != state.current_location:
            novelty += 0.08
        if request.pov and request.pov not in state.emotional_state:
            novelty += 0.05
        novelty += min(0.16, 0.05 * len(new_items))
        novelty += min(0.12, 0.04 * len(request.foreshadowing))
        novelty = clamp(novelty)

        # Reward novel-but-plausible moves more than random weirdness.
        plausibility = clamp((plausibility + timeline + knowledge + constraint) / 4.0)
        surprise = clamp(math.sqrt(max(0.0, novelty * plausibility)))

        details = {
            "lexical_overlap_to_recent": round(lexical_overlap, 3),
            "state_change_size": state_change_size,
            "new_items": new_items,
            "previous_location": previous_location,
        }
        return WorldModelScore(plausibility=round(plausibility, 3), novelty=round(novelty, 3), surprise=round(surprise, 3), issues=issues, details=details)

    @staticmethod
    def _delta_size(previous: StoryState, predicted: Dict[str, Any]) -> int:
        delta = 0
        predicted_location = str(predicted.get("current_location", "") or "").strip()
        predicted_time = str(predicted.get("current_time_label", "") or "").strip()
        if predicted_location and predicted_location != previous.current_location:
            delta += 1
        if predicted_time and predicted_time != previous.current_time_label:
            delta += 1

        predicted_active = set(normalize_list(predicted.get("active_threads", [])))
        predicted_resolved = set(normalize_list(predicted.get("resolved_threads", [])))
        delta += len(predicted_active - set(previous.active_threads))
        delta += len(predicted_resolved - set(previous.resolved_threads))

        predicted_inventory = predicted.get("inventory", {}) if isinstance(predicted.get("inventory"), dict) else {}
        for character, items in predicted_inventory.items():
            delta += len(set(normalize_list(items)) - set(previous.inventory.get(str(character), [])))

        predicted_emotions = predicted.get("emotional_state", {}) if isinstance(predicted.get("emotional_state"), dict) else {}
        for character, emotion in predicted_emotions.items():
            if str(emotion or "").strip() and str(previous.emotional_state.get(str(character), "") or "").strip() != str(emotion).strip():
                delta += 1
        return delta

    def blend_forecast(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        base_score: WorldModelScore,
        forecast: WorldModelForecast,
    ) -> WorldModelScore:
        previous = StoryState(**memory_bundle.get("state", {})) if isinstance(memory_bundle.get("state"), dict) else memory_bundle.get("state", StoryState())
        predicted = forecast.next_state if isinstance(forecast.next_state, dict) else {}
        extraction = forecast.extraction if isinstance(forecast.extraction, dict) else {}

        issues = list(base_score.issues)
        plausibility = base_score.plausibility
        novelty = base_score.novelty

        predicted_location = str(predicted.get("current_location", "") or "").strip()
        if request.location and predicted_location:
            if predicted_location == request.location:
                plausibility += 0.03
            else:
                issues.append(
                    ConsistencyIssue(
                        issue_type="forecast_location_mismatch",
                        severity=Severity.MEDIUM,
                        message="Learned world-model predicted a different next location.",
                        evidence_span=f"{predicted_location} != {request.location}",
                        suggested_fix="Strengthen transition and location anchoring.",
                    )
                )
                plausibility -= 0.08

        predicted_time = str(predicted.get("current_time_label", "") or "").strip()
        if request.time_label and predicted_time:
            if predicted_time == request.time_label:
                plausibility += 0.02
            else:
                issues.append(
                    ConsistencyIssue(
                        issue_type="forecast_time_mismatch",
                        severity=Severity.LOW,
                        message="Learned world-model predicted a different next time label.",
                        evidence_span=f"{predicted_time} != {request.time_label}",
                        suggested_fix="Anchor the time progression more explicitly.",
                    )
                )
                plausibility -= 0.05

        predicted_active = set(normalize_list(predicted.get("active_threads", [])))
        predicted_resolved = set(normalize_list(predicted.get("resolved_threads", [])))
        if not predicted_active and not predicted_resolved:
            predicted_active = set(normalize_list(extraction.get("new_threads", [])))
            predicted_resolved = set(normalize_list(extraction.get("resolved_threads", [])))
        expected_threads = set(normalize_list(request.foreshadowing))
        missing_threads = sorted(expected_threads - (predicted_active | predicted_resolved))
        for thread in missing_threads[:2]:
            issues.append(
                ConsistencyIssue(
                    issue_type="forecast_missing_thread",
                    severity=Severity.LOW,
                    message="Learned world-model did not carry the requested foreshadowing thread forward.",
                    evidence_span=thread,
                    suggested_fix="Make the foreshadowing beat explicit in the scene outcome.",
                )
            )
            plausibility -= 0.03

        delta_size = self._delta_size(previous, predicted)
        novelty = max(novelty, clamp(0.35 + 0.07 * delta_size))
        plausibility = clamp(plausibility)
        surprise = clamp(math.sqrt(max(0.0, novelty * plausibility)))

        details = dict(base_score.details)
        details["learned_world_model"] = {
            "predicted_next_state": predicted,
            "predicted_extraction": extraction,
            "notes": forecast.notes,
            "delta_size": delta_size,
        }
        return WorldModelScore(
            plausibility=round(plausibility, 3),
            novelty=round(novelty, 3),
            surprise=round(surprise, 3),
            issues=issues,
            details=details,
        )
