
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import AppConfig
from .datasets import export_training_bundle
from .db import Storage
from .llm import BaseLLMProvider
from .models import (
    AcceptedScene,
    BibleContent,
    CandidateEvaluation,
    ConsistencyIssue,
    ConsistencyReport,
    CreativityReport,
    EvaluationReport,
    ExtractionOutput,
    GenerationResult,
    OutlineCard,
    OutlineGenerateRequest,
    PlanOutput,
    PoolType,
    RevisionOutput,
    SceneRequest,
    StoryState,
    StoryStatus,
    StoryUpdate,
    WorldModelScore,
    utcnow_iso,
)
from .utils import count_words, ensure_dir, normalize_list, short_text, slugify
from .world_model import NarrativeWorldModel


LogFn = Callable[[str, float], None]


class Orchestrator:
    def __init__(self, storage: Storage, provider: BaseLLMProvider, config: AppConfig) -> None:
        self.storage = storage
        self.provider = provider
        self.config = config
        self.world_model = NarrativeWorldModel()

    def build_memory_bundle(self, story_id: str) -> Dict[str, Any]:
        story = self.storage.get_story(story_id)
        if story is None:
            raise ValueError(f"Story not found: {story_id}")
        bible = self.storage.get_bible(story_id)
        state = self.storage.get_latest_state(story_id)
        scenes = self.storage.list_scenes(story_id)
        recent = scenes[-self.config.orchestration.recent_scene_memory :]
        outline = self.storage.list_outline(story_id)
        return {
            "story": story.model_dump(mode="json"),
            "bible": bible.model_dump(mode="json"),
            "state": state.model_dump(mode="json"),
            "recent_scenes": recent,
            "outline": [card.model_dump(mode="json") for card in outline],
        }

    def generate_outline(self, story_id: str, request: OutlineGenerateRequest) -> List[OutlineCard]:
        cards = self.provider.generate_outline(self.build_memory_bundle(story_id), request.scene_count)
        cards = self._normalize_outline_ids(story_id, cards)
        return self.storage.replace_outline(story_id, cards)

    def scene_request_from_card(self, card: OutlineCard) -> SceneRequest:
        return SceneRequest(
            title_hint=card.title,
            pov=card.pov,
            location=card.location,
            time_label=card.time_label,
            goal=card.goal,
            beat=card.beat,
            foreshadowing=card.foreshadowing,
            required_facts=card.required_facts,
            outline_card_id=card.id,
            min_words=self.config.orchestration.scene_min_words,
            max_words=self.config.orchestration.scene_max_words,
        )

    def _adapt_outline_card(self, story_id: str, card: OutlineCard) -> OutlineCard:
        if not self.config.orchestration.adaptive_outline:
            return card
        state = self.storage.get_latest_state(story_id)
        if not state.active_threads:
            return card
        highest = state.active_threads[0]
        beat = card.beat if highest in card.beat else f"{card.beat}. 그리고 '{highest}'의 여진이 계속된다."
        goal = card.goal
        if card.scene_index > 1 and highest not in goal:
            goal = f"{goal}. 동시에 '{highest}'의 진실을 좁혀 간다."
        return card.model_copy(update={"beat": beat, "goal": goal})

    @staticmethod
    def _normalize_outline_ids(story_id: str, cards: Sequence[OutlineCard]) -> List[OutlineCard]:
        normalized: List[OutlineCard] = []
        seen_ids: set[str] = set()
        for idx, card in enumerate(cards, start=1):
            suffix = slugify(card.id or "") or f"oc{idx:03d}"
            candidate_id = suffix if suffix.startswith(f"{story_id}-") else f"{story_id}-{suffix}"
            if candidate_id in seen_ids:
                candidate_id = f"{story_id}-oc{card.scene_index:03d}-{idx:02d}"
            seen_ids.add(candidate_id)
            normalized.append(card.model_copy(update={"id": candidate_id}))
        return normalized

    def auto_write_novel(self, story_id: str, scene_limit: Optional[int] = None, log: Optional[LogFn] = None) -> Dict[str, Any]:
        def emit(message: str, progress: float) -> None:
            if log:
                log(message, progress)

        story = self.storage.get_story(story_id)
        if story is None:
            raise ValueError(f"Story not found: {story_id}")
        if not self.storage.list_outline(story_id):
            emit("Generating outline", 0.02)
            self.generate_outline(story_id, OutlineGenerateRequest(scene_count=story.target_scene_count))

        outline = self.storage.list_outline(story_id)
        pending = [card for card in outline if card.status != "done"]
        if scene_limit is not None:
            pending = pending[:scene_limit]
        total = max(1, len(pending))
        completed_scene_ids: List[str] = []

        for idx, card in enumerate(pending, start=1):
            adapted = self._adapt_outline_card(story_id, card)
            emit(f"Writing scene {adapted.scene_index}: {adapted.title}", 0.05 + idx / (total + 1) * 0.78)
            result = self.run_scene(story_id, self.scene_request_from_card(adapted), log=log)
            completed_scene_ids.append(result.accepted_scene.scene_id)

        final_scenes = self.storage.list_scenes(story_id)
        if final_scenes and len(final_scenes) >= story.target_scene_count:
            self.storage.update_story(story_id, StoryUpdate(status=StoryStatus.COMPLETED))

        emit("Exporting manuscript", 0.9)
        manuscript = self.write_export_files(story_id)
        emit("Computing evaluation", 0.96)
        evaluation = self.write_evaluation_file(story_id)
        emit("Exporting training bundle", 0.985)
        training_bundle = self.write_training_bundle(story_id)
        emit("Novel run completed", 1.0)

        return {
            "story_id": story_id,
            "completed_scene_ids": completed_scene_ids,
            "scene_count": len(final_scenes),
            "manuscript": manuscript,
            "evaluation": evaluation,
            "training_bundle": training_bundle,
        }

    def run_scene(self, story_id: str, request: SceneRequest, log: Optional[LogFn] = None) -> GenerationResult:
        def emit(message: str, progress: float) -> None:
            if log:
                log(message, progress)

        emit("Loading memory", 0.05)
        memory_bundle = self.build_memory_bundle(story_id)
        next_scene_index = self.storage.get_next_scene_index(story_id)

        self.storage.add_dataset_record(
            story_id,
            None,
            PoolType.PROMPT_ONLY,
            {
                "story_id": story_id,
                "scene_index": next_scene_index,
                "request": request.model_dump(mode="json"),
                "memory_snapshot": {
                    "bible": memory_bundle["bible"],
                    "state": memory_bundle["state"],
                    "recent_scenes": memory_bundle["recent_scenes"],
                },
            },
        )

        emit("Planning scene", 0.15)
        plan = self.provider.plan_scene(memory_bundle, request)

        emit("Generating candidates", 0.35)
        candidates = self._generate_candidates(memory_bundle, request, plan)
        if not candidates:
            raise RuntimeError("Provider returned zero candidates")

        emit("Evaluating candidates", 0.6)
        evaluated = self._evaluate_candidates(memory_bundle, request, plan, candidates, start_index=1)
        evaluated = [
            self._finalize_candidate(memory_bundle, request, plan, item, emit if idx == 0 else None)
            for idx, item in enumerate(evaluated)
        ]
        evaluated = self._rank_candidates(evaluated)

        rescue_round = 0
        while evaluated and not self._passes_release_gate(evaluated[0].consistency) and rescue_round < self.config.orchestration.release_gate_rescue_rounds:
            rescue_round += 1
            emit(f"Release gate retry {rescue_round}", 0.69 + rescue_round * 0.03)
            rescue_count = max(1, self.config.orchestration.release_gate_rescue_candidate_count)
            rescue_candidates = self._generate_candidates(memory_bundle, request, plan, desired_count=rescue_count)
            if not rescue_candidates:
                break
            rescue_evaluated = self._evaluate_candidates(
                memory_bundle,
                request,
                plan,
                rescue_candidates,
                start_index=len(evaluated) + 1,
            )
            rescue_evaluated = [self._finalize_candidate(memory_bundle, request, plan, item) for item in rescue_evaluated]
            evaluated.extend(rescue_evaluated)
            evaluated = self._rank_candidates(evaluated)

        winner = evaluated[0]
        if not self._passes_release_gate(winner.consistency):
            if self.config.orchestration.strict_release_gate:
                raise RuntimeError("Release gate failed after rescue rounds")
            winner.consistency = winner.consistency.model_copy(
                update={
                    "notes": normalize_list(list(winner.consistency.notes) + ["release gate forced accept after rescue exhaustion"])
                }
            )

        emit("Extracting memory updates", 0.82)
        final_text = winner.revised.revised_text if winner.revised else winner.draft.text
        extraction = self.provider.extract_scene(memory_bundle, request, plan, final_text)

        emit("Updating memory", 0.9)
        new_bible = self._merge_bible(self.storage.get_bible(story_id), extraction)
        new_state = self._merge_state(self.storage.get_latest_state(story_id), request, extraction, next_scene_index)

        scene_id = f"{story_id}-sc{next_scene_index:03d}"
        accepted = AcceptedScene(
            scene_id=scene_id,
            scene_index=next_scene_index,
            title=plan.scene_title,
            pov=request.pov,
            location=request.location,
            time_label=request.time_label,
            goal=request.goal,
            request=request,
            plan=plan,
            accepted_text=final_text,
            summary=extraction.summary,
            extraction=extraction,
            consistency=winner.consistency,
            creativity=winner.creativity,
            revision=winner.revised,
            created_at=utcnow_iso(),
        )

        self.storage.save_scene(
            {
                "id": scene_id,
                "story_id": story_id,
                "scene_index": next_scene_index,
                "title": accepted.title,
                "pov": accepted.pov,
                "location": accepted.location,
                "time_label": accepted.time_label,
                "goal": accepted.goal,
                "input": request.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "accepted_text": final_text,
                "summary": extraction.summary,
                "extraction": extraction.model_dump(mode="json"),
                "consistency": winner.consistency.model_dump(mode="json"),
                "creativity": winner.creativity.model_dump(mode="json"),
                "revision": winner.revised.model_dump(mode="json") if winner.revised else None,
                "created_at": accepted.created_at,
            },
            [
                {
                    "id": f"{scene_id}-cand{item.candidate_index}",
                    "candidate_index": item.candidate_index,
                    "text": item.revised.revised_text if item.revised and item is winner else item.draft.text,
                    "score": item.combined_score,
                    "accepted": item is winner,
                    "consistency": item.consistency.model_dump(mode="json"),
                    "creativity": item.creativity.model_dump(mode="json"),
                    "revision": item.revised.model_dump(mode="json") if item.revised else None,
                    "world_model": item.world_model,
                }
                for item in evaluated
            ],
        )
        self.storage.save_state_snapshot(story_id, scene_id, next_scene_index, new_state)
        self.storage.save_bible(story_id, new_bible)
        self.storage.save_kg_edges(story_id, scene_id, [edge.model_dump(mode="json") for edge in extraction.kg_edges])
        if request.outline_card_id:
            self.storage.mark_outline_used(story_id, request.outline_card_id)
        self._store_training_pools(story_id, scene_id, request, plan, evaluated, accepted)
        emit("Scene accepted", 1.0)
        return GenerationResult(accepted_scene=accepted, candidate_evaluations=evaluated, updated_state=new_state, logs=[])

    def _generate_candidates(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        desired_count: int | None = None,
    ) -> List[Any]:
        desired_count = max(1, desired_count or self.config.orchestration.candidate_count)
        candidates: List[Any] = []
        seen_texts: set[str] = set()
        attempts = 0
        max_attempts = max(2, desired_count + 1)

        while len(candidates) < desired_count and attempts < max_attempts:
            batch = self.provider.write_candidates(
                memory_bundle,
                request,
                plan,
                count=max(1, desired_count - len(candidates)),
            )
            attempts += 1
            if not batch:
                break
            for draft in batch:
                text = getattr(draft, "text", "").strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                candidates.append(draft)
                if len(candidates) >= desired_count:
                    break
        return candidates

    def _merge_world_model(
        self,
        consistency: ConsistencyReport,
        creativity: CreativityReport,
        world_score: WorldModelScore,
    ) -> tuple[ConsistencyReport, CreativityReport]:
        issues = list(consistency.issues) + list(world_score.issues)
        adjusted_consistency = consistency.model_copy(
            update={
                "issues": issues,
                "world_plausibility_score": world_score.plausibility,
                "notes": normalize_list(list(consistency.notes) + [f"world_surprise={world_score.surprise}"]),
            }
        )
        adjusted_creativity = creativity.model_copy(
            update={
                "novelty_score": round(max(creativity.novelty_score, world_score.novelty), 3),
                "surprise_score": world_score.surprise,
                "notes": normalize_list(list(creativity.notes) + ["world-model transition score attached"]),
            }
        )
        if world_score.issues:
            penalty = 0.04 * sum(1 for issue in world_score.issues if issue.severity.value in {"medium", "high"})
            adjusted_consistency = adjusted_consistency.model_copy(update={"score": round(max(0.0, consistency.score - penalty), 3)})
        return adjusted_consistency, adjusted_creativity

    def _attach_learned_world_model(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        text: str,
        consistency: ConsistencyReport,
        creativity: CreativityReport,
        world_score: WorldModelScore,
    ) -> tuple[ConsistencyReport, CreativityReport, Dict[str, Any]]:
        try:
            forecast = self.provider.forecast_world_model(memory_bundle, request, plan, text)
        except Exception as exc:
            payload = world_score.model_dump(mode="json")
            payload["forecast_error"] = str(exc)
            updated_consistency = consistency.model_copy(
                update={
                    "notes": normalize_list(
                        list(consistency.notes) + [f"learned world-model skipped: {short_text(str(exc), 180)}"]
                    )
                }
            )
            return updated_consistency, creativity, payload
        if forecast is None:
            return consistency, creativity, world_score.model_dump(mode="json")

        blended = self.world_model.blend_forecast(memory_bundle, request, world_score, forecast)
        seen = {(issue.issue_type, issue.message, issue.evidence_span) for issue in consistency.issues}
        extra_issues = [
            issue
            for issue in blended.issues
            if (issue.issue_type, issue.message, issue.evidence_span) not in seen
        ]
        updated_consistency = consistency.model_copy(
            update={
                "issues": list(consistency.issues) + extra_issues,
                "world_plausibility_score": blended.plausibility,
                "notes": normalize_list(list(consistency.notes) + ["learned world-model forecast attached"]),
            }
        )
        updated_creativity = creativity.model_copy(
            update={
                "novelty_score": round(max(creativity.novelty_score, blended.novelty), 3),
                "surprise_score": blended.surprise,
                "notes": normalize_list(list(creativity.notes) + ["learned world-model forecast attached"]),
            }
        )
        payload = blended.model_dump(mode="json")
        payload["forecast"] = forecast.model_dump(mode="json")
        return updated_consistency, updated_creativity, payload

    def _passes_release_gate(self, consistency: ConsistencyReport) -> bool:
        cfg = self.config.orchestration
        high_count = sum(1 for issue in consistency.issues if issue.severity.value == "high")
        medium_count = sum(1 for issue in consistency.issues if issue.severity.value == "medium")
        return (
            consistency.score >= cfg.minimum_release_consistency
            and consistency.world_plausibility_score >= cfg.release_gate_world_min_plausibility
            and high_count == 0
            and medium_count <= cfg.release_gate_max_medium_issues
        )

    def _rank_candidates(self, candidates: Sequence[CandidateEvaluation]) -> List[CandidateEvaluation]:
        return sorted(
            candidates,
            key=lambda item: (
                1 if self._passes_release_gate(item.consistency) else 0,
                item.consistency.world_plausibility_score,
                item.consistency.score,
                item.combined_score,
                item.creativity.surprise_score,
            ),
            reverse=True,
        )

    def _finalize_candidate(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        item: CandidateEvaluation,
        emit: Optional[LogFn] = None,
    ) -> CandidateEvaluation:
        final_text = item.draft.text
        final_consistency = item.consistency
        final_creativity = item.creativity
        final_world_score = WorldModelScore(**item.world_model) if item.world_model else self.world_model.score(memory_bundle, request, plan, final_text)
        revised_payload: Optional[RevisionOutput] = None

        if self.config.orchestration.auto_revision and self._needs_revision(final_consistency.issues):
            if emit:
                emit("Revising winning candidate", 0.72)
            revised_payload = self.provider.revise_scene(memory_bundle, request, plan, final_text, final_consistency.issues)
            final_text = revised_payload.revised_text
            consistency = self.provider.critique_consistency(memory_bundle, request, plan, final_text)
            creativity = self.provider.critique_creativity(memory_bundle, request, plan, final_text)
            world_score = self.world_model.score(memory_bundle, request, plan, final_text)
            final_consistency, final_creativity = self._merge_world_model(consistency, creativity, world_score)
            final_world_score = world_score

        final_consistency, final_creativity, final_world_payload = self._attach_learned_world_model(
            memory_bundle,
            request,
            plan,
            final_text,
            final_consistency,
            final_creativity,
            final_world_score,
        )
        return item.model_copy(
            update={
                "draft": item.draft.model_copy(update={"text": final_text}) if revised_payload else item.draft,
                "consistency": final_consistency,
                "creativity": final_creativity,
                "combined_score": self._combined_score(final_consistency, final_creativity),
                "revised": revised_payload,
                "world_model": final_world_payload,
            }
        )

    def _evaluate_candidates(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        candidates: Sequence[Any],
        start_index: int = 1,
    ) -> List[CandidateEvaluation]:
        evaluated: List[CandidateEvaluation] = []
        for idx, draft in enumerate(candidates, start=start_index):
            consistency = self.provider.critique_consistency(memory_bundle, request, plan, draft.text)
            creativity = self.provider.critique_creativity(memory_bundle, request, plan, draft.text)
            world_score = self.world_model.score(memory_bundle, request, plan, draft.text)
            consistency, creativity = self._merge_world_model(consistency, creativity, world_score)
            evaluated.append(
                CandidateEvaluation(
                    candidate_index=idx,
                    draft=draft,
                    consistency=consistency,
                    creativity=creativity,
                    combined_score=self._combined_score(consistency, creativity),
                    world_model=world_score.model_dump(mode="json"),
                )
            )
        return evaluated

    def _combined_score(self, consistency: ConsistencyReport, creativity: CreativityReport) -> float:
        cfg = self.config.orchestration
        score = (
            cfg.consistency_weight * consistency.score
            + cfg.creativity_weight
            * (
                0.35 * creativity.novelty_score
                + 0.25 * creativity.hook_score
                + 0.2 * creativity.emotional_depth_score
                + 0.1 * creativity.style_fit_score
                + 0.1 * creativity.surprise_score
            )
            + cfg.world_model_weight * consistency.world_plausibility_score
        )
        high_penalty = 0.18 * sum(1 for issue in consistency.issues if issue.severity.value == "high")
        medium_penalty = 0.05 * sum(1 for issue in consistency.issues if issue.severity.value == "medium")
        return round(max(0.0, min(1.0, score - high_penalty - medium_penalty)), 3)

    @staticmethod
    def _needs_revision(issues: Sequence[ConsistencyIssue]) -> bool:
        return any(issue.severity.value in {"high", "medium"} for issue in issues)

    @staticmethod
    def _merge_bible(bible: BibleContent, extraction: ExtractionOutput) -> BibleContent:
        return bible.model_copy(update={"static_facts": normalize_list(bible.static_facts + extraction.new_static_facts)})

    def _merge_state(self, previous: StoryState, request: SceneRequest, extraction: ExtractionOutput, scene_index: int) -> StoryState:
        updates = extraction.state_updates or {}
        current_time_label = updates.get("current_time_label") or request.time_label or previous.current_time_label
        current_location = updates.get("current_location") or request.location or previous.current_location
        active_threads = normalize_list(previous.active_threads + extraction.new_threads + updates.get("active_threads_add", []))
        resolved_threads = normalize_list(previous.resolved_threads + extraction.resolved_threads + updates.get("resolved_threads_add", []))
        active_threads = [thread for thread in active_threads if thread not in set(resolved_threads)]
        knowledge = dict(previous.character_knowledge)
        for char, facts in extraction.knowledge_updates.items():
            knowledge.setdefault(char, [])
            knowledge[char] = normalize_list(knowledge[char] + facts)
        inventory = dict(previous.inventory)
        for char, items in extraction.inventory_updates.items():
            inventory.setdefault(char, [])
            inventory[char] = normalize_list(inventory[char] + items)
        emotional = dict(previous.emotional_state)
        emotional.update(extraction.emotional_updates)
        summary_memory = (previous.summary_memory + [extraction.summary])[-self.config.orchestration.max_summary_memory :]
        return StoryState(
            last_scene_index=scene_index,
            current_time_label=current_time_label,
            current_location=current_location,
            active_threads=active_threads,
            resolved_threads=resolved_threads,
            character_knowledge=knowledge,
            inventory=inventory,
            emotional_state=emotional,
            summary_memory=summary_memory,
        )

    def _store_training_pools(
        self,
        story_id: str,
        scene_id: str,
        request: SceneRequest,
        plan: PlanOutput,
        candidate_evaluations: List[CandidateEvaluation],
        accepted: AcceptedScene,
    ) -> None:
        winner = candidate_evaluations[0]
        self.storage.add_dataset_record(
            story_id,
            scene_id,
            PoolType.ACCEPTED,
            {
                "request": request.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "accepted_scene": accepted.model_dump(mode="json"),
            },
        )
        for candidate in candidate_evaluations[1:]:
            self.storage.add_dataset_record(
                story_id,
                scene_id,
                PoolType.PAIRWISE,
                {
                    "request": request.model_dump(mode="json"),
                    "accepted_text": accepted.accepted_text,
                    "rejected_text": candidate.draft.text,
                    "accepted_score": winner.combined_score,
                    "rejected_score": candidate.combined_score,
                },
            )
        for candidate in candidate_evaluations:
            if any(issue.severity.value == "high" for issue in candidate.consistency.issues):
                self.storage.add_dataset_record(
                    story_id,
                    scene_id,
                    PoolType.HARD_NEGATIVE,
                    {
                        "request": request.model_dump(mode="json"),
                        "plan": plan.model_dump(mode="json"),
                        "text": candidate.draft.text,
                        "issues": [issue.model_dump(mode="json") for issue in candidate.consistency.issues],
                    },
                )

    def evaluate_story(self, story_id: str) -> EvaluationReport:
        scenes = self.storage.list_scenes(story_id)
        if not scenes:
            return EvaluationReport(
                story_id=story_id,
                scene_count=0,
                average_consistency_score=0.0,
                average_novelty_score=0.0,
                average_hook_score=0.0,
                average_emotional_depth_score=0.0,
                average_world_plausibility_score=0.0,
                average_surprise_score=0.0,
                unresolved_threads=[],
                resolved_threads=[],
                issue_counts={},
                dataset_counts=self.storage.dataset_counts(story_id),
                notes=["No scenes generated yet."],
            )
        avg_consistency = sum(scene["consistency"].get("score", 0.0) for scene in scenes) / len(scenes)
        avg_novelty = sum(scene["creativity"].get("novelty_score", 0.0) for scene in scenes) / len(scenes)
        avg_hook = sum(scene["creativity"].get("hook_score", 0.0) for scene in scenes) / len(scenes)
        avg_emotion = sum(scene["creativity"].get("emotional_depth_score", 0.0) for scene in scenes) / len(scenes)
        avg_world = sum(scene["consistency"].get("world_plausibility_score", 0.0) for scene in scenes) / len(scenes)
        avg_surprise = sum(scene["creativity"].get("surprise_score", 0.0) for scene in scenes) / len(scenes)

        issue_counts: Dict[str, int] = {}
        total_words = 0
        for scene in scenes:
            total_words += count_words(scene["accepted_text"])
            for issue in scene["consistency"].get("issues", []):
                key = issue.get("issue_type", "other")
                issue_counts[key] = issue_counts.get(key, 0) + 1

        state = self.storage.get_latest_state(story_id)
        notes = []
        if avg_consistency < self.config.orchestration.minimum_release_consistency:
            notes.append("Continuity score is below the recommended release threshold.")
        if len(state.active_threads) > max(2, len(scenes)):
            notes.append("Unresolved threads are accumulating faster than they are closing.")
        if issue_counts.get("required_fact", 0) > 0:
            notes.append("Some mandatory scene facts were not rendered strongly enough.")
        story = self.storage.get_story(story_id)
        if story and total_words < story.target_word_count * 0.6:
            notes.append("The manuscript is shorter than the requested target; consider longer scenes or a higher min_words.")
        if avg_world < 0.68:
            notes.append("World-model plausibility is low; inspect transition gaps and overcompressed deltas.")

        return EvaluationReport(
            story_id=story_id,
            scene_count=len(scenes),
            average_consistency_score=round(avg_consistency, 3),
            average_novelty_score=round(avg_novelty, 3),
            average_hook_score=round(avg_hook, 3),
            average_emotional_depth_score=round(avg_emotion, 3),
            average_world_plausibility_score=round(avg_world, 3),
            average_surprise_score=round(avg_surprise, 3),
            unresolved_threads=state.active_threads,
            resolved_threads=state.resolved_threads,
            issue_counts=issue_counts,
            dataset_counts=self.storage.dataset_counts(story_id),
            notes=notes,
        )

    def export_story_markdown(self, story_id: str) -> Dict[str, Any]:
        story = self.storage.get_story(story_id)
        if story is None:
            raise ValueError(f"Story not found: {story_id}")
        bible = self.storage.get_bible(story_id)
        state = self.storage.get_latest_state(story_id)
        scenes = self.storage.list_scenes(story_id)
        lines = [
            f"# {story.title}",
            "",
            f"- Genre: {story.genre}",
            f"- Tone: {story.tone}",
            f"- Themes: {', '.join(story.themes)}",
            f"- Characters: {', '.join(story.characters)}",
            f"- Target scene count: {story.target_scene_count}",
            "",
            "## Premise",
            story.premise,
            "",
            "## Story Bible",
        ]
        for item in bible.static_facts:
            lines.append(f"- {item}")
        if bible.rules:
            lines.extend(["", "## Rules", *[f"- {item}" for item in bible.rules]])
        if bible.motifs:
            lines.extend(["", "## Motifs", *[f"- {item}" for item in bible.motifs]])
        lines.append("")
        lines.append("## Manuscript")
        for scene in scenes:
            lines.extend(
                [
                    "",
                    f"### Scene {scene['scene_index']}: {scene['title']}",
                    f"*POV:* {scene['pov']}  ",
                    f"*Time:* {scene['time_label']}  ",
                    f"*Location:* {scene['location']}  ",
                    f"*Consistency:* {scene['consistency'].get('score', 0)} / *World plausibility:* {scene['consistency'].get('world_plausibility_score', 0)}",
                    "",
                    scene["accepted_text"],
                ]
            )
        lines.extend(
            [
                "",
                "## Final State Snapshot",
                f"- Current time label: {state.current_time_label}",
                f"- Current location: {state.current_location}",
                f"- Active threads: {', '.join(state.active_threads)}",
                f"- Resolved threads: {', '.join(state.resolved_threads)}",
            ]
        )
        return {
            "filename": f"{story_id}.md",
            "content": "\n".join(lines).strip() + "\n",
            "metadata": {"story_id": story_id, "scene_count": len(scenes), "generated_at": utcnow_iso()},
        }

    def write_export_files(self, story_id: str) -> Dict[str, Any]:
        artifact = self.export_story_markdown(story_id)
        export_dir = ensure_dir(Path(self.config.workspace.exports_dir) / story_id)
        path = export_dir / artifact["filename"]
        path.write_text(artifact["content"], encoding="utf-8")
        saved = self.storage.save_artifact(story_id, "manuscript", str(path), artifact["metadata"])
        return {"path": str(path), "artifact": saved.model_dump(mode="json"), **artifact}

    def write_evaluation_file(self, story_id: str) -> Dict[str, Any]:
        report = self.evaluate_story(story_id)
        export_dir = ensure_dir(Path(self.config.workspace.exports_dir) / story_id)
        path = export_dir / f"{story_id}_evaluation.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        saved = self.storage.save_artifact(story_id, "evaluation", str(path), {"story_id": story_id, "generated_at": utcnow_iso()})
        return {"path": str(path), "report": report.model_dump(mode="json"), "artifact": saved.model_dump(mode="json")}

    def write_training_bundle(self, story_id: str) -> Dict[str, Any]:
        export_dir = ensure_dir(Path(self.config.workspace.exports_dir) / story_id / "training")
        return export_training_bundle(self.storage, story_id, export_dir)
