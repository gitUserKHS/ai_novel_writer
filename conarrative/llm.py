from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel

from .models import (
    BibleContent,
    ConsistencyIssue,
    ConsistencyReport,
    CreativityReport,
    DraftCandidate,
    ExtractionOutput,
    KGEdge,
    OutlineCard,
    PlanBeat,
    PlanOutput,
    ProviderType,
    RevisionOutput,
    RuntimeSettings,
    SceneRequest,
    Severity,
    StoryState,
)
from .prompts import (
    consistency_system_prompt,
    consistency_user_prompt,
    creativity_system_prompt,
    creativity_user_prompt,
    extraction_system_prompt,
    extraction_user_prompt,
    outline_system_prompt,
    outline_user_prompt,
    planner_system_prompt,
    planner_user_prompt,
    revision_system_prompt,
    revision_user_prompt,
    writer_system_prompt,
    writer_user_prompt,
)
from .utils.jsonfix import loads_json_loose
from .utils.text import clip_text, join_korean_and, korean_particle, normalize_list, stable_seed

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

    def set_stream_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        return None

    def close(self) -> None:
        return None

    @abstractmethod
    def health(self) -> Tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        raise NotImplementedError

    @abstractmethod
    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        raise NotImplementedError

    @abstractmethod
    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput) -> List[DraftCandidate]:
        raise NotImplementedError

    @abstractmethod
    def critique_consistency(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> ConsistencyReport:
        raise NotImplementedError

    @abstractmethod
    def critique_creativity(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> CreativityReport:
        raise NotImplementedError

    @abstractmethod
    def revise_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
        issues: Sequence[ConsistencyIssue],
    ) -> RevisionOutput:
        raise NotImplementedError

    @abstractmethod
    def extract_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> ExtractionOutput:
        raise NotImplementedError


class MockLLMProvider(BaseLLMProvider):
    VARIANT_HINTS = [
        "lean into tactile atmosphere and withheld tension",
        "lean into dialog and emotional contradiction",
        "lean into symbolic detail and a lingering final image",
        "lean into motion and sensory compression",
    ]

    def health(self) -> Tuple[bool, str]:
        return True, "Mock backend active"

    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        story = memory_bundle["story"]
        premise = story["premise"]
        characters = story.get("characters") or ["주인공"]
        primary = characters[0]
        secondary = characters[1] if len(characters) > 1 else "또 다른 인물"
        settings = [
            "낡은 역 대합실",
            "비에 젖은 골목",
            "작은 아파트 부엌",
            "강변 산책로",
            "버려진 극장 로비",
            "새벽의 옥상",
            "잠겨 있던 기록 보관실",
            "눈 오는 버스 정류장",
        ]
        cards: List[OutlineCard] = []
        for idx in range(scene_count):
            act = "setup" if idx < max(1, scene_count // 3) else "complication" if idx < max(2, scene_count * 2 // 3) else "payoff"
            location = settings[idx % len(settings)]
            if idx == 0:
                goal = f"{primary}가 균열의 시작점을 감지한다"
                summary = f"{premise}를 구체적인 사건으로 열어 주고, 중심 갈등의 첫 흔들림을 보여준다."
            elif idx == scene_count - 1:
                goal = f"{primary}가 가장 아픈 진실을 선택한다"
                summary = f"초반에 심어 둔 정서와 상징을 회수하며 결말의 잔향을 남긴다."
            else:
                goal = f"{primary}와 {secondary}의 관계를 {act} 단계로 밀어붙인다"
                summary = f"갈등을 한 단계 심화하고, 다음 장면을 부르는 새 질문을 남긴다."
            cards.append(
                OutlineCard(
                    id=f"outline-{idx+1:02d}",
                    title=f"Scene {idx+1}: {primary}의 {['징후','접촉','균열','도주','대면','전환','회수','선택'][idx % 8]}",
                    pov=primary,
                    goal=goal,
                    location=location,
                    time_label=f"Day {idx+1} / {['morning','afternoon','evening','night'][idx % 4]}",
                    summary_request=summary,
                    beats=[
                        f"{primary}가 현재 문제를 손으로 만질 수 있는 행동으로 시작한다",
                        f"{secondary} 또는 환경이 예상 밖의 저항을 만든다",
                        f"장면 말미에 다음 장면으로 이어지는 정서적 질문을 남긴다",
                    ],
                    must_include=[(story.get("themes") or ["감정의 여진"])[0]],
                    must_avoid=["설명조 요약"],
                    status="planned",
                )
            )
        return cards

    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        beats = request.beats or [
            f"{request.pov}가 {request.location}에 들어서며 장면이 열린다",
            f"{request.goal}와 연결된 저항이 드러난다",
            "마지막 문단에서 다음 장면으로 이어질 여진을 남긴다",
        ]
        plan_beats = []
        for idx, beat in enumerate(beats, start=1):
            plan_beats.append(PlanBeat(label=f"Beat {idx}", purpose=beat))
        return PlanOutput(
            scene_title=request.title or f"{request.pov} - {request.goal}",
            synopsis=request.summary_request,
            beats=plan_beats,
            expected_reveals=normalize_list(request.must_include[:2] + request.emotion_targets[:1]),
            expected_new_threads=[f"{request.goal}의 대가"],
            expected_resolved_threads=[],
            expected_state_delta={
                "current_time_label": request.time_label,
                "current_location": request.location,
                "goal_pressure": request.goal,
            },
        )

    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput) -> List[DraftCandidate]:
        count = max(1, min(4, self.settings.candidate_count))
        outputs: List[DraftCandidate] = []
        for idx in range(count):
            seed = stable_seed(request.pov, request.goal, request.location, request.time_label, str(idx))
            rng = random.Random(seed)
            outputs.append(self._mock_scene_candidate(memory_bundle, request, plan, idx, rng))
        return outputs

    def _mock_scene_candidate(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        idx: int,
        rng: random.Random,
    ) -> DraftCandidate:
        story = memory_bundle["story"]
        motifs = memory_bundle["bible"].get("motifs") or story.get("themes") or ["잔향"]
        motif = motifs[idx % len(motifs)]
        if request.emotion_targets:
            emotional = join_korean_and(request.emotion_targets)
        else:
            emotional = "조용한 긴장"
        topic = korean_particle(request.pov, "은/는")
        subject = korean_particle(request.pov, "이/가")
        obj_goal = korean_particle(request.goal, "을/를")
        pov_obj = korean_particle(request.pov, "을/를")
        opener = [
            f"{request.time_label}의 {request.location}는 숨을 죽인 채 {request.pov}{pov_obj} 받아들였다.",
            f"{request.pov}{topic} {request.location}에 발을 들이는 순간, 오늘의 공기가 평소보다 한 겹 더 얇다는 걸 알아차렸다.",
            f"{request.location}에 남은 빛은 거의 없었고, {request.pov}{topic} 그 어둠이 오히려 자신을 또렷하게 비춘다고 느꼈다.",
            f"{request.time_label}, {request.location}는 너무 조용해서 누군가의 결심이 소리처럼 들릴 것 같았다.",
        ][idx % 4]
        middle_lines = []
        for beat in plan.beats:
            verb = rng.choice(["만졌다", "밀어붙였다", "되짚었다", "붙잡았다", "견뎠다"])
            beat_sentence = beat.purpose.rstrip(". ")
            if not beat_sentence.endswith(("다", "했다", "였다")):
                beat_sentence = beat_sentence + "다"
            middle_lines.append(
                f"{beat_sentence}. {request.pov}{topic} 그 움직임의 결을 놓치지 않은 채, {request.goal}라는 목표를 향해 한 걸음 더 {verb}."
            )
        include_lines = [f"그 순간 떠오른 것은 {item}였다." for item in request.must_include[:2]]
        if not include_lines:
            include_lines = [f"{motif} 같은 작은 징후가 장면의 방향을 바꾸었다."]
        conflict_line = (
            f"하지만 {request.pov}{subject} 원하는 것과 지금 감당할 수 있는 것은 달랐고, 그 틈새에서 {emotional} 같은 감정이 서서히 얼굴을 드러냈다."
        )
        closer = [
            f"장면이 끝날 무렵, {request.pov}{topic} 다음에 잃게 될 것이 무엇인지 어렴풋이 알았다.",
            f"마지막에 남은 것은 대답이 아니라, {request.pov}{subject} 더는 모른 척할 수 없는 질문이었다.",
            f"그리고 {request.pov}{topic} 방금 지나간 침묵이 앞으로의 모든 선택값을 바꾸리라는 예감을 떨치지 못했다.",
            f"문득 {request.pov}{topic} 지금의 결심이 훗날 자신을 구할지 무너뜨릴지 아직 모른다는 사실만은 분명히 이해했다.",
        ][rng.randrange(4)]
        text = "\n\n".join(
            [
                opener,
                " ".join(middle_lines[:2]),
                " ".join(include_lines + [conflict_line]),
                closer,
            ]
        )
        return DraftCandidate(
            text=text,
            strengths=[self.VARIANT_HINTS[idx % len(self.VARIANT_HINTS)], f"motif={motif}"],
            risks=["may under-specify concrete action" if idx == 2 else "may need stronger causal marker"],
            strategy=self.VARIANT_HINTS[idx % len(self.VARIANT_HINTS)],
        )

    def critique_consistency(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> ConsistencyReport:
        issues: List[ConsistencyIssue] = []
        lowered = scene_text.lower()
        if request.must_include:
            for item in request.must_include:
                if item.lower() not in lowered:
                    issues.append(
                        ConsistencyIssue(
                            issue_type="coverage",
                            severity=Severity.HIGH,
                            message=f"Required element missing: {item}",
                            evidence="must_include check",
                            suggested_fix=f"Integrate {item} concretely into the scene.",
                        )
                    )
        for item in request.must_avoid:
            if item and item.lower() in lowered:
                issues.append(
                    ConsistencyIssue(
                        issue_type="rule",
                        severity=Severity.HIGH,
                        message=f"Forbidden element present: {item}",
                        evidence=item,
                        suggested_fix=f"Remove or replace {item}.",
                    )
                )
        forbidden = memory_bundle["story"].get("forbidden_facts") or []
        for item in forbidden:
            if item and item.lower() in lowered:
                issues.append(
                    ConsistencyIssue(
                        issue_type="rule",
                        severity=Severity.MEDIUM,
                        message=f"Scene may violate story rule: {item}",
                        evidence=item,
                        suggested_fix="Rephrase the moment to avoid breaking a locked fact.",
                    )
                )
        if request.location and request.location.lower() not in lowered:
            issues.append(
                ConsistencyIssue(
                    issue_type="location",
                    severity=Severity.LOW,
                    message="Location is not clearly grounded in the prose.",
                    evidence=request.location,
                    suggested_fix="Anchor the scene with one or two location details.",
                )
            )
        if len(scene_text.split()) < int(request.desired_length_words * 0.45):
            issues.append(
                ConsistencyIssue(
                    issue_type="coverage",
                    severity=Severity.MEDIUM,
                    message="Scene is much shorter than requested.",
                    evidence=f"target={request.desired_length_words}",
                    suggested_fix="Expand action, sensory detail, or emotional turn.",
                )
            )
        high_count = sum(1 for issue in issues if issue.severity == Severity.HIGH)
        score = max(0.0, 0.92 - 0.22 * high_count - 0.08 * (len(issues) - high_count))
        verdict = "accept" if high_count == 0 else "needs_revision"
        return ConsistencyReport(score=round(score, 3), issues=issues, checks_run=["must_include", "must_avoid", "rule", "location", "length"], verdict=verdict)

    def critique_creativity(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> CreativityReport:
        unique_ratio = len(set(scene_text.split())) / max(1, len(scene_text.split()))
        novelty = min(1.0, 0.55 + unique_ratio * 0.5)
        hook = 0.62 + (0.12 if "?" in scene_text or "예감" in scene_text else 0.0)
        emotion = 0.6 + (0.15 if request.emotion_targets else 0.05)
        language = 0.58 + (0.15 if len(scene_text.split("\n\n")) >= 3 else 0.05)
        return CreativityReport(
            novelty_score=round(min(novelty, 0.95), 3),
            hook_score=round(min(hook, 0.95), 3),
            emotional_depth_score=round(min(emotion, 0.95), 3),
            language_score=round(min(language, 0.95), 3),
            summary="Image-driven mock evaluation with emphasis on emotional continuity.",
            strengths=[
                f"Strong scene premise around {request.goal}",
                f"Motif recall supports tone: {', '.join(memory_bundle['story'].get('themes') or ['implicit motif'])}",
            ],
            opportunities=[
                "Increase concrete action beats if the draft feels too reflective.",
                "Tighten one image so the closing line lands harder.",
            ],
        )

    def revise_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
        issues: Sequence[ConsistencyIssue],
    ) -> RevisionOutput:
        revised = scene_text
        change_log: List[str] = []
        addressed: List[str] = []
        lowered = revised.lower()
        for issue in issues:
            if issue.issue_type.value == "coverage" and "Required element missing:" in issue.message:
                missing = issue.message.split(":", 1)[1].strip()
                if missing and missing.lower() not in lowered:
                    pov_topic = korean_particle(request.pov, "은/는")
                    revised = revised.rstrip() + f"\n\n마침내 {missing}이 장면의 표면으로 떠올랐고, {request.pov}{pov_topic} 그 사실을 외면할 수 없었다."
                    change_log.append(f"Added missing required element: {missing}")
                    addressed.append(issue.message)
                    lowered = revised.lower()
            elif issue.issue_type.value == "location" and request.location.lower() not in lowered:
                revised = f"{request.location}의 공기부터 이상했다.\n\n" + revised
                change_log.append("Strengthened location anchoring in the opening paragraph.")
                addressed.append(issue.message)
                lowered = revised.lower()
        if not change_log:
            revised = revised.rstrip() + f"\n\n{request.pov}{korean_particle(request.pov, '은/는')} 방금의 선택이 앞으로 남은 장면들을 바꾸리라는 걸 천천히 받아들였다."
            change_log.append("Added a clearer closing causal hook.")
            addressed.extend([issue.message for issue in issues[:1]])
        return RevisionOutput(revised_text=revised, change_log=change_log, addressed_issues=addressed)

    def extract_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> ExtractionOutput:
        summary = clip_text(scene_text.replace("\n", " "), 220)
        knowledge = {}
        if request.pov:
            knowledge[request.pov] = normalize_list([request.goal] + request.must_include[:2])
        inventory_updates = {}
        emotional_updates = {}
        if request.emotion_targets:
            emotional_updates[request.pov] = ", ".join(request.emotion_targets)
        else:
            emotional_updates[request.pov] = "tense but controlled"
        state_updates = {
            "current_time_label": request.time_label,
            "current_location": request.location,
            "active_threads_add": normalize_list(plan.expected_new_threads + request.must_include[:1]),
            "resolved_threads_add": normalize_list(plan.expected_resolved_threads),
        }
        kg_edges = [
            KGEdge(source=request.pov, relation="pursues", target=request.goal, edge_type="intent", metadata={"location": request.location}),
            KGEdge(source=request.location, relation="frames", target=request.pov, edge_type="setting", metadata={"time_label": request.time_label}),
        ]
        return ExtractionOutput(
            summary=summary,
            new_static_facts=[],
            state_updates=state_updates,
            new_threads=normalize_list(plan.expected_new_threads),
            resolved_threads=normalize_list(plan.expected_resolved_threads),
            knowledge_updates=knowledge,
            inventory_updates=inventory_updates,
            emotional_updates=emotional_updates,
            kg_edges=kg_edges,
            tags=normalize_list([request.pov, request.location] + request.emotion_targets[:1]),
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__(settings)
        self.base_url = settings.base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.timeout_seconds)
        self.stream_callback: Optional[Callable[[str], None]] = None

    def close(self) -> None:
        self.client.close()

    def set_stream_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self.stream_callback = callback

    def health(self) -> Tuple[bool, str]:
        models_url = f"{self.base_url}/models"
        headers = self._headers()
        health_timeout = min(max(self.settings.timeout_seconds, 1), 5)
        try:
            response = self.client.get(models_url, headers=headers, timeout=health_timeout)
            if response.status_code < 400:
                return True, "Connected to OpenAI-compatible backend"
        except Exception as exc:  # pragma: no cover - network dependent
            detail = str(exc)
        else:  # pragma: no cover - network dependent
            detail = f"HTTP {response.status_code}: {response.text[:240]}"
        try:
            text = self._chat_text(
                system="Return the single word OK.",
                user="Health check.",
                temperature=0.0,
                max_tokens=8,
                timeout=health_timeout,
            )
            return True, f"Chat completion OK: {text[:80]}"
        except Exception as exc:  # pragma: no cover - network dependent
            return False, detail + f" | chat failed: {exc}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _chat_payload(self, system: str, user: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return payload

    def _chat_text(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int = 1500,
        stream: bool = False,
        timeout: float | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = self._chat_payload(system, user, temperature=temperature, max_tokens=max_tokens)
        if stream and self.stream_callback is not None:
            try:
                return self._chat_text_streaming(url, payload)
            except Exception as exc:
                if self.stream_callback is not None:
                    self.stream_callback(f"\n[stream fallback: {exc}]\n")
        request_kwargs: Dict[str, Any] = {"headers": self._headers(), "json": payload}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        response = self.client.post(url, **request_kwargs)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - depends on backend response shape
            raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:400]}") from exc

    def _chat_text_streaming(self, url: str, payload: Dict[str, Any]) -> str:
        streamed_payload = {**payload, "stream": True}
        chunks: list[str] = []
        with self.client.stream("POST", url, headers=self._headers(), json=streamed_payload) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = self._extract_stream_delta(item)
                if not delta:
                    continue
                chunks.append(delta)
                if self.stream_callback is not None:
                    self.stream_callback(delta)
        return "".join(chunks)

    @staticmethod
    def _extract_stream_delta(item: Dict[str, Any]) -> str:
        choices = item.get("choices") or []
        if not choices:
            return ""
        first = choices[0]
        delta = first.get("delta") or {}
        if isinstance(delta, dict) and delta.get("content"):
            return str(delta["content"])
        if first.get("text"):
            return str(first["text"])
        message = first.get("message") or {}
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        return ""

    def _chat_json(self, schema: Type[T], system: str, user: str, temperature: float, max_tokens: int = 1800) -> T:
        text = self._chat_text(system, user, temperature=temperature, max_tokens=max_tokens)
        payload = loads_json_loose(text)
        return schema.model_validate(payload)

    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        class _Envelope(BaseModel):
            cards: List[OutlineCard]

        response = self._chat_json(
            _Envelope,
            outline_system_prompt(),
            outline_user_prompt(memory_bundle, scene_count),
            temperature=self.settings.temperature_planner,
            max_tokens=2200,
        )
        return response.cards

    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        return self._chat_json(
            PlanOutput,
            planner_system_prompt(),
            planner_user_prompt(memory_bundle, request.model_dump()),
            temperature=self.settings.temperature_planner,
            max_tokens=1600,
        )

    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput) -> List[DraftCandidate]:
        candidates: List[DraftCandidate] = []
        hints = [
            "prioritize tactile atmosphere and patient tension",
            "prioritize dialog pressure and emotional contradiction",
            "prioritize symbolic detail and a resonant final image",
            "prioritize action clarity and causal flow",
        ]
        count = max(1, min(4, self.settings.candidate_count))
        for idx in range(count):
            text = self._chat_text(
                writer_system_prompt(),
                writer_user_prompt(memory_bundle, request.model_dump(), plan.model_dump(), hints[idx % len(hints)]),
                temperature=self.settings.temperature_writer + idx * 0.05,
                max_tokens=min(3000, max(1200, request.desired_length_words * 2)),
                stream=True,
            )
            candidates.append(
                DraftCandidate(
                    text=text.strip(),
                    strengths=[hints[idx % len(hints)]],
                    risks=[],
                    strategy=hints[idx % len(hints)],
                )
            )
        return candidates

    def critique_consistency(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> ConsistencyReport:
        report = self._chat_json(
            ConsistencyReport,
            consistency_system_prompt(),
            consistency_user_prompt(memory_bundle, request.model_dump(), plan.model_dump(), scene_text),
            temperature=self.settings.temperature_critic,
            max_tokens=1800,
        )
        if not report.checks_run:
            report.checks_run = ["timeline", "causality", "coverage"]
        return report

    def critique_creativity(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> CreativityReport:
        return self._chat_json(
            CreativityReport,
            creativity_system_prompt(),
            creativity_user_prompt(memory_bundle, request.model_dump(), plan.model_dump(), scene_text),
            temperature=self.settings.temperature_critic,
            max_tokens=1600,
        )

    def revise_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
        issues: Sequence[ConsistencyIssue],
    ) -> RevisionOutput:
        revised = self._chat_text(
            revision_system_prompt(),
            revision_user_prompt(
                memory_bundle,
                request.model_dump(),
                plan.model_dump(),
                scene_text,
                [issue.model_dump() for issue in issues],
            ),
            temperature=self.settings.temperature_revision,
            max_tokens=min(3200, max(1200, request.desired_length_words * 2)),
            stream=True,
        )
        return RevisionOutput(
            revised_text=revised.strip(),
            change_log=[issue.message for issue in issues],
            addressed_issues=[issue.message for issue in issues],
        )

    def extract_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
    ) -> ExtractionOutput:
        return self._chat_json(
            ExtractionOutput,
            extraction_system_prompt(),
            extraction_user_prompt(memory_bundle, request.model_dump(), plan.model_dump(), scene_text),
            temperature=self.settings.temperature_critic,
            max_tokens=2000,
        )


class HybridProvider(BaseLLMProvider):
    """Wraps a real provider with deterministic guardrails for robustness."""

    def __init__(self, primary: BaseLLMProvider, fallback: BaseLLMProvider) -> None:
        super().__init__(primary.settings)
        self.primary = primary
        self.fallback = fallback

    def health(self) -> Tuple[bool, str]:
        return self.primary.health()

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()

    def set_stream_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self.primary.set_stream_callback(callback)
        self.fallback.set_stream_callback(callback)

    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        try:
            cards = self.primary.generate_outline(memory_bundle, scene_count)
            return cards or self.fallback.generate_outline(memory_bundle, scene_count)
        except Exception:
            return self.fallback.generate_outline(memory_bundle, scene_count)

    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        try:
            return self.primary.plan_scene(memory_bundle, request)
        except Exception:
            return self.fallback.plan_scene(memory_bundle, request)

    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput) -> List[DraftCandidate]:
        try:
            candidates = self.primary.write_candidates(memory_bundle, request, plan)
            return candidates or self.fallback.write_candidates(memory_bundle, request, plan)
        except Exception:
            return self.fallback.write_candidates(memory_bundle, request, plan)

    def critique_consistency(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, scene_text: str) -> ConsistencyReport:
        try:
            primary = self.primary.critique_consistency(memory_bundle, request, plan, scene_text)
        except Exception:
            primary = self.fallback.critique_consistency(memory_bundle, request, plan, scene_text)
        fallback = self.fallback.critique_consistency(memory_bundle, request, plan, scene_text)
        merged = list(primary.issues)
        existing = {(issue.message, issue.issue_type.value) for issue in merged}
        for issue in fallback.issues:
            key = (issue.message, issue.issue_type.value)
            if key not in existing:
                merged.append(issue)
        high_count = sum(1 for issue in merged if issue.severity == Severity.HIGH)
        penalty = 0.22 * high_count + 0.08 * max(0, len(merged) - high_count)
        score = max(0.0, min(primary.score, fallback.score, 0.96) - penalty * 0.1)
        return ConsistencyReport(
            score=round(score, 3),
            issues=merged,
            checks_run=normalize_list(primary.checks_run + fallback.checks_run),
            verdict="accept" if high_count == 0 else "needs_revision",
        )

    def critique_creativity(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, scene_text: str) -> CreativityReport:
        try:
            primary = self.primary.critique_creativity(memory_bundle, request, plan, scene_text)
        except Exception:
            return self.fallback.critique_creativity(memory_bundle, request, plan, scene_text)
        return primary

    def revise_scene(
        self,
        memory_bundle: Dict[str, Any],
        request: SceneRequest,
        plan: PlanOutput,
        scene_text: str,
        issues: Sequence[ConsistencyIssue],
    ) -> RevisionOutput:
        try:
            return self.primary.revise_scene(memory_bundle, request, plan, scene_text, issues)
        except Exception:
            return self.fallback.revise_scene(memory_bundle, request, plan, scene_text, issues)

    def extract_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, scene_text: str) -> ExtractionOutput:
        try:
            primary = self.primary.extract_scene(memory_bundle, request, plan, scene_text)
        except Exception:
            return self.fallback.extract_scene(memory_bundle, request, plan, scene_text)
        fallback = self.fallback.extract_scene(memory_bundle, request, plan, scene_text)
        if not primary.summary:
            primary.summary = fallback.summary
        primary.new_threads = normalize_list(primary.new_threads + fallback.new_threads)
        primary.resolved_threads = normalize_list(primary.resolved_threads + fallback.resolved_threads)
        for character, facts in fallback.knowledge_updates.items():
            primary.knowledge_updates.setdefault(character, [])
            primary.knowledge_updates[character] = normalize_list(primary.knowledge_updates[character] + facts)
        primary.tags = normalize_list(primary.tags + fallback.tags)
        if not primary.kg_edges:
            primary.kg_edges = fallback.kg_edges
        return primary


def build_provider(settings: RuntimeSettings) -> BaseLLMProvider:
    if settings.provider == ProviderType.MOCK:
        return MockLLMProvider(settings)
    if settings.provider == ProviderType.OPENAI_COMPATIBLE:
        primary = OpenAICompatibleProvider(settings)
        fallback = MockLLMProvider(settings)
        return HybridProvider(primary=primary, fallback=fallback)
    raise ValueError(f"Unsupported provider: {settings.provider}")
