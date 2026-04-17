from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import AppConfig
from .db import Storage
from .llm import BaseLLMProvider
from .models import (
    AcceptedScene,
    BibleContent,
    CandidateEvaluation,
    ConsistencyIssue,
    ConsistencyReport,
    CreativityReport,
    DraftCandidate,
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
    utcnow_iso,
)
from .utils.text import normalize_list

LogFn = Callable[[str, float], None]


class Orchestrator:
    def __init__(self, storage: Storage, provider: BaseLLMProvider, config: AppConfig) -> None:
        self.storage = storage
        self.provider = provider
        self.config = config

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
            "story": story.model_dump(),
            "bible": bible.model_dump(),
            "state": state.model_dump(),
            "recent_scenes": [scene.model_dump() for scene in recent],
            "outline": [card.model_dump() for card in outline],
        }

    def generate_outline(self, story_id: str, request: OutlineGenerateRequest) -> List[OutlineCard]:
        memory_bundle = self.build_memory_bundle(story_id)
        cards = self.provider.generate_outline(memory_bundle, request.scene_count)
        return self.storage.replace_outline(story_id, cards)

    def run_scene(self, story_id: str, request: SceneRequest, log: Optional[LogFn] = None) -> GenerationResult:
        def emit(message: str, progress: float) -> None:
            if log is not None:
                log(message, progress)

        emit("Loading story memory", 0.05)
        memory_bundle = self.build_memory_bundle(story_id)
        story = memory_bundle["story"]
        next_scene_index = self.storage.get_next_scene_index(story_id)

        self.storage.add_dataset_record(
            story_id,
            None,
            PoolType.PROMPT_ONLY,
            {
                "story_id": story_id,
                "scene_index": next_scene_index,
                "request": request.model_dump(),
                "memory_snapshot": {
                    "state": memory_bundle["state"],
                    "bible": memory_bundle["bible"],
                    "recent_scenes": memory_bundle["recent_scenes"],
                },
            },
        )

        emit("Planning scene", 0.15)
        plan = self.provider.plan_scene(memory_bundle, request)

        emit("Generating draft candidates", 0.35)
        candidates = self.provider.write_candidates(memory_bundle, request, plan)
        if not candidates:
            raise RuntimeError("Provider returned no draft candidates")

        emit("Running critics", 0.55)
        candidate_evaluations: List[CandidateEvaluation] = []
        for idx, draft in enumerate(candidates, start=1):
            consistency = self.provider.critique_consistency(memory_bundle, request, plan, draft.text)
            creativity = self.provider.critique_creativity(memory_bundle, request, plan, draft.text)
            score = self._combined_score(consistency, creativity)
            candidate_evaluations.append(
                CandidateEvaluation(
                    candidate_index=idx,
                    draft=draft,
                    consistency=consistency,
                    creativity=creativity,
                    combined_score=score,
                )
            )

        candidate_evaluations.sort(key=lambda item: item.combined_score, reverse=True)
        winner = candidate_evaluations[0]

        emit("Revision pass", 0.72)
        revised_payload: Optional[RevisionOutput] = None
        final_text = winner.draft.text
        final_consistency = winner.consistency
        final_creativity = winner.creativity
        if self.config.orchestration.auto_revision and self._needs_revision(winner.consistency.issues):
            revised_payload = self.provider.revise_scene(memory_bundle, request, plan, winner.draft.text, winner.consistency.issues)
            final_text = revised_payload.revised_text
            final_consistency = self.provider.critique_consistency(memory_bundle, request, plan, final_text)
            final_creativity = self.provider.critique_creativity(memory_bundle, request, plan, final_text)
            winner.revised = revised_payload
            winner.combined_score = self._combined_score(final_consistency, final_creativity)
        emit("Extracting memory update", 0.84)
        extraction = self.provider.extract_scene(memory_bundle, request, plan, final_text)

        emit("Updating memory", 0.92)
        new_state = self._merge_state(self.storage.get_latest_state(story_id), request, extraction, next_scene_index)
        new_bible = self._merge_bible(self.storage.get_bible(story_id), extraction)

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
            consistency=final_consistency,
            creativity=final_creativity,
            revision=revised_payload,
            created_at=utcnow_iso(),
        )

        persisted_scene = self.storage.save_scene(
            {
                "id": scene_id,
                "story_id": story_id,
                "scene_index": next_scene_index,
                "title": plan.scene_title,
                "pov": request.pov,
                "location": request.location,
                "time_label": request.time_label,
                "goal": request.goal,
                "input": request.model_dump(),
                "plan": plan.model_dump(),
                "accepted_text": final_text,
                "summary": extraction.summary,
                "extraction": extraction.model_dump(),
                "consistency": final_consistency.model_dump(),
                "creativity": final_creativity.model_dump(),
                "revision": revised_payload.model_dump() if revised_payload else None,
            },
            [
                {
                    "id": f"{scene_id}-cand{item.candidate_index}",
                    "candidate_index": item.candidate_index,
                    "text": item.revised.revised_text if item.revised and item is winner else item.draft.text,
                    "score": item.combined_score,
                    "accepted": item is winner,
                    "consistency": (final_consistency if item is winner else item.consistency).model_dump(),
                    "creativity": (final_creativity if item is winner else item.creativity).model_dump(),
                    "revision": item.revised.model_dump() if item.revised else None,
                }
                for item in candidate_evaluations
            ],
        )

        self.storage.save_state_snapshot(story_id, scene_id, next_scene_index, new_state)
        self.storage.save_bible(story_id, new_bible)
        if request.outline_card_id:
            self.storage.mark_outline_used(story_id, request.outline_card_id)
        self._store_training_pools(story_id, scene_id, request, plan, candidate_evaluations, accepted)

        emit("Scene accepted", 1.0)
        return GenerationResult(
            accepted_scene=accepted,
            candidate_evaluations=candidate_evaluations,
            updated_state=new_state,
            logs=[],
        )

    def _combined_score(self, consistency: ConsistencyReport, creativity: CreativityReport) -> float:
        creativity_avg = (
            creativity.novelty_score + creativity.hook_score + creativity.emotional_depth_score + creativity.language_score
        ) / 4.0
        score = (
            self.config.orchestration.consistency_weight * consistency.score
            + self.config.orchestration.creativity_weight * creativity_avg
        )
        high_penalty = 0.18 * sum(1 for issue in consistency.issues if issue.severity.value == "high")
        medium_penalty = 0.05 * sum(1 for issue in consistency.issues if issue.severity.value == "medium")
        return round(max(0.0, score - high_penalty - medium_penalty), 3)

    @staticmethod
    def _needs_revision(issues: Sequence[ConsistencyIssue]) -> bool:
        return any(issue.severity.value in {"high", "medium"} for issue in issues)

    @staticmethod
    def _merge_bible(bible: BibleContent, extraction: ExtractionOutput) -> BibleContent:
        bible.static_facts = normalize_list(bible.static_facts + extraction.new_static_facts)
        return bible

    @staticmethod
    def _merge_state(previous: StoryState, request: SceneRequest, extraction: ExtractionOutput, scene_index: int) -> StoryState:
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
        emotion = dict(previous.emotional_state)
        emotion.update(extraction.emotional_updates)
        summary_memory = previous.summary_memory[-6:] + [extraction.summary]
        return StoryState(
            last_scene_index=scene_index,
            current_time_label=current_time_label,
            current_location=current_location,
            active_threads=active_threads,
            resolved_threads=resolved_threads,
            character_knowledge=knowledge,
            inventory=inventory,
            emotional_state=emotion,
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
                "request": request.model_dump(),
                "plan": plan.model_dump(),
                "accepted_scene": accepted.model_dump(),
            },
        )
        for candidate in candidate_evaluations[1:]:
            self.storage.add_dataset_record(
                story_id,
                scene_id,
                PoolType.PAIRWISE,
                {
                    "request": request.model_dump(),
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
                        "request": request.model_dump(),
                        "plan": plan.model_dump(),
                        "text": candidate.draft.text,
                        "issues": [issue.model_dump() for issue in candidate.consistency.issues],
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
                unresolved_threads=[],
                resolved_threads=[],
                issue_counts={},
                dataset_counts=self.storage.dataset_counts(story_id),
                notes=["No scenes have been generated yet."],
            )
        avg_consistency = sum(scene.consistency.get("score", 0.0) for scene in scenes) / len(scenes)
        avg_novelty = sum(scene.creativity.get("novelty_score", 0.0) for scene in scenes) / len(scenes)
        avg_hook = sum(scene.creativity.get("hook_score", 0.0) for scene in scenes) / len(scenes)
        avg_emotion = sum(scene.creativity.get("emotional_depth_score", 0.0) for scene in scenes) / len(scenes)
        issue_counts: Dict[str, int] = {}
        for scene in scenes:
            for issue in scene.consistency.get("issues", []):
                key = issue.get("issue_type", "other")
                issue_counts[key] = issue_counts.get(key, 0) + 1
        state = self.storage.get_latest_state(story_id)
        notes = []
        if issue_counts.get("coverage", 0) > 0:
            notes.append("Some scenes under-hit requested coverage or length; consider one more revision pass.")
        if len(state.active_threads) > max(2, len(scenes)):
            notes.append("Active threads are accumulating faster than they are being resolved.")
        if avg_consistency < 0.7:
            notes.append("Continuity stability is below the recommended comfort line for release exports.")
        return EvaluationReport(
            story_id=story_id,
            scene_count=len(scenes),
            average_consistency_score=round(avg_consistency, 3),
            average_novelty_score=round(avg_novelty, 3),
            average_hook_score=round(avg_hook, 3),
            average_emotional_depth_score=round(avg_emotion, 3),
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
            "",
            "## Premise",
            story.premise,
            "",
            "## Story Bible",
        ]
        for item in bible.static_facts:
            lines.append(f"- {item}")
        if bible.rules:
            lines.append("\n## Rules")
            for item in bible.rules:
                lines.append(f"- {item}")
        lines.append("\n## Manuscript")
        for scene in scenes:
            lines.extend(
                [
                    "",
                    f"### Scene {scene.scene_index}: {scene.title}",
                    f"*POV:* {scene.pov}  ",
                    f"*Time:* {scene.time_label}  ",
                    f"*Location:* {scene.location}",
                    "",
                    scene.accepted_text,
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
            "metadata": {
                "story_id": story_id,
                "scene_count": len(scenes),
                "generated_at": utcnow_iso(),
            },
        }
