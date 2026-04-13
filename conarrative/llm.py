
from __future__ import annotations

import gc
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import urlparse

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
    PlanOutput,
    RevisionOutput,
    RuntimeSettings,
    SceneRequest,
    Severity,
    StoryState,
    WorldModelForecast,
)
from .prompts import compact_memory, make_json_messages
from .utils import clamp, ensure_dir, extract_json_object, normalize_list, short_text, stable_hash


_LOCAL_ROLE_RUNTIME_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def _adapter_checkpoint_exists(path: str | Path) -> bool:
    resolved = Path(path)
    return resolved.exists() and (resolved / "adapter_config.json").exists()


def _looks_like_local_ollama(base_url: str) -> bool:
    parsed = urlparse(base_url or "")
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost"} and parsed.port == 11434


def _parse_ollama_ps_models(output: str) -> list[str]:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    models: list[str] = []
    for line in lines[1:]:
        name = line.split()[0].strip()
        if name and name != "NAME":
            models.append(name)
    return models


class BaseLLMProvider(ABC):
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

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
    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, count: int = 3) -> List[DraftCandidate]:
        raise NotImplementedError

    @abstractmethod
    def critique_consistency(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ConsistencyReport:
        raise NotImplementedError

    @abstractmethod
    def critique_creativity(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> CreativityReport:
        raise NotImplementedError

    @abstractmethod
    def revise_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str, issues: Sequence[ConsistencyIssue]) -> RevisionOutput:
        raise NotImplementedError

    @abstractmethod
    def extract_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ExtractionOutput:
        raise NotImplementedError

    def forecast_world_model(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> WorldModelForecast | None:
        return None


class MockProvider(BaseLLMProvider):
    sensory = ["비 냄새", "낡은 먼지", "젖은 커튼", "차가운 금속", "숨 막히는 정적", "미세한 발자국 소리"]
    emotions = ["불안", "그리움", "의심", "결의", "수치심", "안도", "질투", "경외"]
    items = ["열쇠", "편지", "메모", "목걸이", "사진", "서류"]

    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__(settings)
        self.random = random.Random(7)

    def health(self) -> Tuple[bool, str]:
        return True, "mock provider active"

    def _story(self, memory_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return memory_bundle.get("story", {})

    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        story = self._story(memory_bundle)
        chars = story.get("characters") or ["주인공"]
        locations = [
            "극장 로비", "분장실", "지하 통로", "관장실", "무대 뒤편", "옥상", "폐도서관", "비밀 창고"
        ]
        beats = [
            "단서를 발견한다",
            "불신이 커진다",
            "금지된 사실을 엿본다",
            "관계를 시험한다",
            "과거의 흔적과 맞선다",
            "결정적인 선택을 준비한다",
            "숨은 적의를 확인한다",
            "결말의 문턱에 선다",
        ]
        cards: List[OutlineCard] = []
        for idx in range(scene_count):
            pov = chars[idx % len(chars)]
            location = locations[idx % len(locations)]
            beat = beats[idx % len(beats)]
            title = f"{idx+1}장 {beat}"
            goal = f"{beat} 그리고 {story.get('premise', '문제')}의 중심에 더 가까이 간다"
            foreshadow = []
            if idx == 0:
                foreshadow.append("잠긴 문 뒤에서 반복되는 발소리")
            if idx == scene_count - 1:
                foreshadow.append("마지막 장면에서 밝혀질 숨겨진 증언")
            cards.append(
                OutlineCard(
                    id=f"{story.get('id', 'story')}-oc{idx+1:03d}",
                    scene_index=idx + 1,
                    title=title,
                    pov=pov,
                    location=location,
                    time_label=f"D{1 + idx//3} {20 + (idx % 3) * 2:02d}:00",
                    goal=goal,
                    beat=beat,
                    foreshadowing=foreshadow,
                    required_facts=[story.get("premise", "")] if idx == 0 else [],
                    status="pending",
                )
            )
        return cards

    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        story = self._story(memory_bundle)
        scene_title = request.title_hint or f"{request.location}의 장면"
        must_include = normalize_list([request.location, request.time_label] + request.required_facts + request.foreshadowing)
        beat_sheet = [
            f"{request.pov}이/가 {request.location}에 들어선다.",
            f"{request.goal}를 향한 대화나 조사 장면이 나온다.",
            f"감정의 흔들림과 다음 장면의 갈고리가 남는다.",
        ]
        reasoning = [
            f"장르={story.get('genre', '')}와 톤={story.get('tone', '')}을 유지한다.",
            "장면 단위로 상태를 조금만 움직여 과압축을 피한다.",
            "세계 규칙을 깨지 않는 선에서 단서와 감정 곡선을 동시에 밀어 올린다.",
        ]
        return PlanOutput(
            scene_title=scene_title,
            beat_sheet=beat_sheet,
            must_include=must_include,
            reasoning=reasoning,
            target_word_count=max(request.min_words, min(request.max_words, 480)),
        )

    def _paragraphs(self, request: SceneRequest, plan: PlanOutput, variant: int, memory_bundle: Dict[str, Any]) -> str:
        story = self._story(memory_bundle)
        sensory = self.sensory[(variant - 1) % len(self.sensory)]
        emotion = self.emotions[(variant + 1) % len(self.emotions)]
        item = self.items[(variant + len(request.location)) % len(self.items)] if request.location else self.items[variant % len(self.items)]
        recent = memory_bundle.get("recent_scenes", [])
        callback = ""
        if recent:
            callback = f"직전 장면의 여운인 '{short_text(recent[-1].get('summary', ''), 40)}'이 아직 {request.pov}의 귓가에 남아 있었다. "
        hook = request.foreshadowing[0] if request.foreshadowing else f"{item} 하나가 다음 장면의 문을 연다"

        p1 = (
            f"{request.time_label}의 {request.location}에는 {sensory}가 눅눅하게 맴돌고 있었다. "
            f"{request.pov}은 {callback}{story.get('premise', '어떤 문제')}을 다시 떠올리며 걸음을 늦췄다. "
            f"{request.pov}이 원하는 것은 분명했다. {request.goal}. 하지만 그 바람은 늘 누군가의 침묵과 맞부딪혔다."
        )
        p2 = (
            f"장면의 중심에는 {plan.beat_sheet[1]}라는 압력이 있었다. "
            f"{request.pov}은 작은 소리와 시선의 흔들림을 따라가며 상대의 숨은 뜻을 재고, "
            f"자신도 모르게 {emotion} 쪽으로 기울었다. "
            f"손끝에 닿은 것은 {item}였고, 그것은 단순한 물건이 아니라 오늘 밤의 균형을 바꿀 징후처럼 느껴졌다."
        )
        fact_sentences = []
        for fact in request.required_facts:
            fact_sentences.append(f"그 순간 {fact}라는 사실이 장면 속에 또렷하게 드러났다.")
        if not fact_sentences and variant == 2:
            fact_sentences.append("하지만 중요한 사실 하나는 끝내 분명하게 말해지지 않은 채 맴돌았다.")
        foreshadow_line = f"그리고 {hook}라는 예감이 대사와 정적 사이에 엷게 깔렸다."
        p3 = (
            "대화는 짧았지만 파장은 길었다. "
            + " ".join(fact_sentences)
            + " "
            + foreshadow_line
            + " "
            + "마지막에 남은 것은 해답보다도, 다음 장면에서 누군가 반드시 대가를 치르게 되리라는 조용한 확신이었다."
        )
        if variant == 3:
            p2 += " 다만 이동의 맥락을 거의 설명하지 않아 연결이 다소 거칠게 느껴졌다."
        return "\n\n".join([p1, p2, p3]).strip()

    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, count: int = 3) -> List[DraftCandidate]:
        candidates = []
        for variant in range(1, max(1, count) + 1):
            candidates.append(
                DraftCandidate(
                    text=self._paragraphs(request, plan, variant, memory_bundle),
                    notes=[f"mock_variant={variant}"],
                )
            )
        return candidates

    def critique_consistency(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ConsistencyReport:
        bible = BibleContent(**memory_bundle.get("bible", {})) if isinstance(memory_bundle.get("bible"), dict) else BibleContent()
        state = StoryState(**memory_bundle.get("state", {})) if isinstance(memory_bundle.get("state"), dict) else StoryState()
        issues: List[ConsistencyIssue] = []

        if request.location and request.location not in text:
            issues.append(
                ConsistencyIssue(
                    issue_type="missing_location_anchor",
                    severity=Severity.MEDIUM,
                    message="지정된 장소가 장면에 충분히 드러나지 않았다.",
                    evidence_span=request.location,
                    suggested_fix="도입 문단에 장소 표식을 분명히 넣어라.",
                )
            )
        if request.time_label and request.time_label not in text:
            issues.append(
                ConsistencyIssue(
                    issue_type="missing_time_anchor",
                    severity=Severity.LOW,
                    message="시간 레이블이 장면에 약하게 반영되었다.",
                    evidence_span=request.time_label,
                    suggested_fix="첫 문단에 시간 표식을 넣어라.",
                )
            )
        for fact in request.required_facts:
            if fact and fact not in text:
                issues.append(
                    ConsistencyIssue(
                        issue_type="required_fact",
                        severity=Severity.HIGH,
                        message="필수 사실이 장면에 반영되지 않았다.",
                        evidence_span=fact,
                        suggested_fix="이 사실을 대사나 서술로 명시해라.",
                    )
                )
        for rule in normalize_list(bible.forbidden + bible.rules):
            if rule and rule in text and ("없음" in rule or "금지" in rule):
                issues.append(
                    ConsistencyIssue(
                        issue_type="forbidden_rule_echo",
                        severity=Severity.LOW,
                        message="금지 규칙이 그대로 서술돼 메타적으로 튀어 보인다.",
                        evidence_span=rule,
                        suggested_fix="규칙 문구 대신 서사 안에서 간접적으로 표현해라.",
                    )
                )
        if state.current_location and request.location and state.current_location != request.location:
            if not any(token in text for token in ["걸음을", "향했다", "도착", "돌아왔", "이동", "내려", "올라"]):
                issues.append(
                    ConsistencyIssue(
                        issue_type="transition_gap",
                        severity=Severity.MEDIUM,
                        message="직전 장소와의 전환 단서가 약하다.",
                        evidence_span=f"{state.current_location} -> {request.location}",
                        suggested_fix="장소 전환 문장을 한두 줄 추가해라.",
                    )
                )

        score = 0.9 - 0.2 * sum(1 for i in issues if i.severity == Severity.HIGH) - 0.08 * sum(1 for i in issues if i.severity == Severity.MEDIUM) - 0.03 * sum(1 for i in issues if i.severity == Severity.LOW)
        notes = ["scene-level consistency critique", f"issue_count={len(issues)}"]
        return ConsistencyReport(score=round(clamp(score), 3), issues=issues, notes=notes, world_plausibility_score=round(clamp(score + 0.04), 3))

    def critique_creativity(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> CreativityReport:
        unique_tokens = len(set(re.findall(r"[0-9A-Za-z가-힣]+", text)))
        novelty = clamp(0.45 + unique_tokens / 220)
        hook = clamp(0.45 + (0.15 if request.foreshadowing else 0.05) + (0.08 if "마지막" in text or "예감" in text else 0.0))
        emotion = clamp(0.4 + 0.08 * sum(1 for token in self.emotions if token in text))
        style = clamp(0.55 + (0.08 if request.location in text else 0.0) + (0.08 if request.time_label in text else 0.0))
        surprise = clamp((novelty * hook) ** 0.5)
        return CreativityReport(
            novelty_score=round(novelty, 3),
            hook_score=round(hook, 3),
            emotional_depth_score=round(emotion, 3),
            style_fit_score=round(style, 3),
            surprise_score=round(surprise, 3),
            notes=["mock creativity pass"],
        )

    def revise_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str, issues: Sequence[ConsistencyIssue]) -> RevisionOutput:
        revised = text
        change_log: List[str] = []
        fixed_issue_types: List[str] = []
        if any(issue.issue_type == "missing_location_anchor" for issue in issues) and request.location and request.location not in revised:
            revised = f"{request.location}의 공기는 처음부터 장면을 붙들고 있었다.\n\n{revised}"
            change_log.append("장소 앵커 추가")
            fixed_issue_types.append("missing_location_anchor")
        if any(issue.issue_type == "missing_time_anchor" for issue in issues) and request.time_label and request.time_label not in revised:
            revised = f"{request.time_label}. {revised}"
            change_log.append("시간 앵커 추가")
            fixed_issue_types.append("missing_time_anchor")
        missing_facts = [issue.evidence_span for issue in issues if issue.issue_type == "required_fact" and issue.evidence_span]
        if missing_facts:
            revised += "\n\n" + " ".join([f"무엇보다 {fact}라는 사실이 더는 숨겨지지 않았다." for fact in missing_facts])
            change_log.append("필수 사실 보강")
            fixed_issue_types.append("required_fact")
        if any(issue.issue_type == "transition_gap" for issue in issues):
            revised = f"{request.pov}은 숨을 고르고 {request.location} 쪽으로 걸음을 옮겼다.\n\n{revised}"
            change_log.append("장면 전환 문장 추가")
            fixed_issue_types.append("transition_gap")
        return RevisionOutput(revised_text=revised, change_log=change_log, fixed_issue_types=fixed_issue_types)

    def extract_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ExtractionOutput:
        summary = short_text(re.sub(r"\s+", " ", text), 220)
        new_static_facts = [fact for fact in request.required_facts if fact and fact in text]
        new_threads = normalize_list(request.foreshadowing)
        resolved_threads = []
        if "해답" in text or "밝혀" in text:
            resolved_threads = [thread for thread in new_threads[:1]]
        knowledge_updates = {request.pov: normalize_list(request.required_facts)}
        inventory_updates = {}
        for item in self.items:
            if item in text:
                inventory_updates.setdefault(request.pov, []).append(item)
        emotional_updates = {request.pov: next((emotion for emotion in self.emotions if emotion in text), "복합적")}
        kg_edges = [
            KGEdge(source=request.pov or "주인공", relation="located_in", target=request.location or "미상"),
            KGEdge(source=request.pov or "주인공", relation="pursues", target=request.goal or "목표"),
        ]
        for fact in request.required_facts:
            kg_edges.append(KGEdge(source=request.pov or "주인공", relation="learns", target=fact))
        for thread in request.foreshadowing:
            kg_edges.append(KGEdge(source=plan.scene_title, relation="foreshadows", target=thread))
        return ExtractionOutput(
            summary=summary,
            new_static_facts=new_static_facts,
            state_updates={
                "current_time_label": request.time_label,
                "current_location": request.location,
            },
            new_threads=new_threads,
            resolved_threads=resolved_threads,
            knowledge_updates=knowledge_updates,
            inventory_updates={key: normalize_list(value) for key, value in inventory_updates.items()},
            emotional_updates=emotional_updates,
            kg_edges=kg_edges,
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__(settings)
        self.client = httpx.Client(timeout=settings.timeout_seconds, headers=settings.extra_headers or {})
        self.cache_dir = ensure_dir(settings.cache_dir) if settings.cache_responses else None
        self._ollama_unloaded_for_local_role = False

    def _role_model(self, role: str) -> str:
        return self.settings.role_models.get(role, self.settings.model)

    def _has_explicit_role_model(self, role: str) -> bool:
        value = self.settings.role_models.get(role)
        return bool(str(value or "").strip())

    def _role_temperature(self, role: str) -> float:
        if role in {"consistency_critic", "creativity_critic", "world_model"}:
            return self.settings.critic_temperature
        return self.settings.temperature

    def _local_role_model_path(self, role: str) -> str | None:
        model_ref = str(self._role_model(role) or "").strip()
        if not model_ref:
            return None
        candidate = Path(model_ref).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        return None

    def _configured_remote_model_names(self) -> list[str]:
        names = {str(self.settings.model or "").strip()}
        for value in (self.settings.role_models or {}).values():
            text = str(value or "").strip()
            if text and "/" not in text and "\\" not in text:
                names.add(text)
        return sorted(name for name in names if name)

    def _unload_local_ollama_before_local_role(self) -> None:
        if not _looks_like_local_ollama(self.settings.base_url):
            return
        target_models = self._configured_remote_model_names()
        for model_name in target_models:
            try:
                subprocess.run(["ollama", "stop", model_name], check=False, capture_output=True, text=True)
            except Exception:
                continue
        deadline = time.time() + 15.0
        while time.time() < deadline:
            try:
                ps_result = subprocess.run(["ollama", "ps"], check=False, capture_output=True, text=True)
                running = set(_parse_ollama_ps_models(ps_result.stdout))
            except Exception:
                break
            if not any(model_name in running for model_name in target_models):
                break
            time.sleep(0.5)
        self._ollama_unloaded_for_local_role = True

    def _evict_other_local_role_runtimes(self, active_model_ref: str) -> None:
        stale_refs = [model_ref for model_ref in _LOCAL_ROLE_RUNTIME_CACHE if model_ref != active_model_ref]
        if not stale_refs:
            return
        for model_ref in stale_refs:
            runtime = _LOCAL_ROLE_RUNTIME_CACHE.pop(model_ref, None)
            if runtime is None:
                continue
            _, _, model = runtime
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _local_model_device(model: Any) -> Any:
        try:
            return next(model.parameters()).device
        except Exception:
            return getattr(model, "device", "cpu")

    def _load_local_role_runtime(self, model_ref: str) -> tuple[Any, Any, Any]:
        cached = _LOCAL_ROLE_RUNTIME_CACHE.get(model_ref)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import AutoPeftModelForCausalLM
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                f"Missing local inference dependencies for role model '{model_ref}': {type(exc).__name__}: {exc}"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["device_map"] = "cpu"

        if _adapter_checkpoint_exists(model_ref):
            model = AutoPeftModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
        model.eval()
        runtime = (torch, tokenizer, model)
        _LOCAL_ROLE_RUNTIME_CACHE[model_ref] = runtime
        return runtime

    def _remote_repair_model_name(self) -> str:
        for role in ["extractor", "planner", "writer"]:
            model_name = str(self.settings.role_models.get(role, "") or "").strip()
            if model_name and "/" not in model_name and "\\" not in model_name:
                return model_name
        model_name = str(self.settings.model or "").strip()
        if model_name and "/" not in model_name and "\\" not in model_name:
            return model_name
        raise RuntimeError("No remote repair model is configured.")

    def _repair_raw_structured_output(
        self,
        role: str,
        system: str,
        payload: Dict[str, Any],
        raw_text: str,
        schema: Dict[str, Any] | str,
        max_tokens: int | None = None,
    ) -> Any:
        model_name = self._remote_repair_model_name()
        messages = make_json_messages(
            "You repair malformed structured model outputs. Convert the raw text into valid JSON that matches the provided schema. Return JSON only.",
            {
                "role": role,
                "original_system": system,
                "original_payload": payload,
                "raw_text": raw_text,
                "schema": schema,
            },
        )
        request_payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "response_format": {"type": "json_object"},
        }
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        response = self.client.post(url, json=request_payload, headers=headers)
        if response.status_code >= 400:
            fallback_payload = dict(request_payload)
            fallback_payload.pop("response_format", None)
            response = self.client.post(url, json=fallback_payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return extract_json_object(content)

    def _chat_local_role_json(
        self,
        role: str,
        system: str,
        payload: Dict[str, Any],
        max_tokens: int | None = None,
        schema: Dict[str, Any] | str | None = None,
    ) -> Any:
        model_ref = self._local_role_model_path(role)
        if model_ref is None:
            raise RuntimeError(f"Local role model path not found for role '{role}'")

        request_payload = dict(payload)
        if schema is not None and "schema" not in request_payload:
            request_payload["schema"] = schema
        assistant_prefill = "{" if schema is not None else ""
        messages = make_json_messages(
            f"{system}\nReturn valid JSON only. Do not emit markdown fences, commentary, or <think> tags.",
            request_payload,
        )
        cache_key = stable_hash(
            {
                "role": role,
                "model_ref": model_ref,
                "messages": messages,
                "max_tokens": max_tokens or self.settings.max_tokens,
            }
        )
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached

        self._unload_local_ollama_before_local_role()
        request_file = None
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "local_role_infer.py"
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
                request_file = handle.name
                json.dump(
                    {
                        "model_ref": model_ref,
                        "messages": messages,
                        "max_tokens": max_tokens or self.settings.max_tokens,
                        "temperature": self._role_temperature(role),
                        "assistant_prefill": assistant_prefill,
                    },
                    handle,
                    ensure_ascii=False,
                )
            result = subprocess.run(
                [sys.executable, str(script_path), "--request-file", request_file],
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Local role subprocess failed for '{role}' ({model_ref}): {result.stderr.strip() or result.stdout.strip()}"
                )
            parsed = json.loads(result.stdout)
            if isinstance(parsed, dict) and "__raw_text__" in parsed and schema is not None:
                parsed = self._repair_raw_structured_output(
                    role,
                    system,
                    request_payload,
                    str(parsed.get("__raw_text__", "")),
                    schema,
                    max_tokens=max_tokens,
                )
        finally:
            if request_file:
                try:
                    Path(request_file).unlink(missing_ok=True)
                except Exception:
                    pass
        self._cache_write(cache_key, parsed)
        return parsed

    def _cache_read(self, key: str) -> Any | None:
        if not self.cache_dir:
            return None
        path = Path(self.cache_dir) / f"{key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload in ({}, [], None):
            return None
        return payload

    def _cache_write(self, key: str, payload: Any) -> None:
        if not self.cache_dir:
            return
        if payload in ({}, [], None):
            return
        path = Path(self.cache_dir) / f"{key}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _pick(payload: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return default

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return normalize_list(str(item) for item in value)
        if isinstance(value, dict):
            flattened: List[str] = []
            for key, item in value.items():
                if isinstance(item, list):
                    flattened.extend(f"{key}: {entry}" for entry in item)
                elif isinstance(item, dict):
                    flattened.extend(f"{key}.{sub_key}: {sub_value}" for sub_key, sub_value in item.items())
                elif item is not None:
                    flattened.append(f"{key}: {item}")
            return normalize_list(flattened)
        if isinstance(value, str):
            return normalize_list(line.strip() for line in value.replace(",", "\n").splitlines())
        return normalize_list([value])

    @staticmethod
    def _int_value(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float_value(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clean_scene_text(text: Any) -> str:
        cleaned = str(text or "").strip()
        fence_index = cleaned.find("```")
        if fence_index != -1:
            cleaned = cleaned[:fence_index].rstrip()
        return cleaned.strip()

    @staticmethod
    def _severity_value(value: Any) -> Severity:
        text = str(value or "").strip().lower()
        if text in {"critical", "high", "severe"}:
            return Severity.HIGH
        if text in {"medium", "med", "moderate"}:
            return Severity.MEDIUM
        return Severity.LOW

    def _normalize_outline_items(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = self._pick(
                data,
                "items",
                "outline",
                "cards",
                "scenes",
                "next_scene_plan",
                "scene_breakdown",
                "scene_plan",
                default=[],
            )
            if isinstance(items, dict):
                items = self._pick(items, "items", "cards", "scenes", default=[])
            if not isinstance(items, list) and all(key in data for key in ["scene_index", "title"]):
                items = [data]
        else:
            items = []
        return [item for item in items if isinstance(item, dict)]

    def _normalize_outline_cards(self, data: Any, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        story = memory_bundle.get("story", {}) if isinstance(memory_bundle, dict) else {}
        story_id = story.get("id", "story")
        items = self._normalize_outline_items(data)
        cards: List[OutlineCard] = []
        for idx, item in enumerate(items[:scene_count], start=1):
            scene_index = self._int_value(self._pick(item, "scene_index", "index", "order", default=idx), idx)
            title = str(self._pick(item, "title", "scene_title", "name", default=f"{scene_index}장"))
            cards.append(
                OutlineCard(
                    id=str(self._pick(item, "id", default=f"{story_id}-oc{scene_index:03d}")),
                    scene_index=scene_index,
                    title=title,
                    pov=str(self._pick(item, "pov", "viewpoint", "character", default="주인공")),
                    location=str(self._pick(item, "location", "scene_location", "place", default="")),
                    time_label=str(self._pick(item, "time_label", "time", "time_of_day", default="")),
                    goal=str(self._pick(item, "goal", "objective", "purpose", default=title)),
                    beat=str(self._pick(item, "beat", "summary", "action", default=title)),
                    foreshadowing=self._string_list(self._pick(item, "foreshadowing", "foreshadow", default=[])),
                    required_facts=self._string_list(self._pick(item, "required_facts", "facts", "must_include", default=[])),
                    status=str(self._pick(item, "status", default="pending")),
                )
            )
        return cards

    def _normalize_plan_output(self, data: Any, request: SceneRequest) -> PlanOutput:
        payload = data if isinstance(data, dict) else {}
        title = str(self._pick(payload, "scene_title", "title", "name", default=request.title_hint or f"{request.location} 장면"))
        target_word_count = self._int_value(
            self._pick(payload, "target_word_count", "word_count", "target_length", default=500),
            default=500,
        )
        return PlanOutput(
            scene_title=title,
            beat_sheet=self._string_list(self._pick(payload, "beat_sheet", "beats", "scene_beats", "steps", default=[])),
            must_include=self._string_list(self._pick(payload, "must_include", "required_facts", "constraints", default=[])),
            reasoning=self._string_list(self._pick(payload, "reasoning", "rationale", "notes", default=[])),
            target_word_count=max(request.min_words, min(request.max_words, target_word_count)),
        )

    def _normalize_candidates(self, data: Any) -> List[DraftCandidate]:
        payload = data if isinstance(data, dict) else {}
        items = self._pick(payload, "items", "candidates", "drafts", "variations", "outputs", default=[])
        if isinstance(data, list):
            items = data
        if isinstance(items, dict):
            items = self._pick(items, "items", "candidates", default=[])
        if not items and isinstance(payload, dict):
            single_text = self._pick(payload, "text", "draft", "content")
            if single_text:
                items = [{"text": single_text, "notes": self._string_list(self._pick(payload, "notes", "reasoning", default=[]))}]
        candidates: List[DraftCandidate] = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str):
                candidates.append(DraftCandidate(text=item, notes=[]))
                continue
            if not isinstance(item, dict):
                continue
            text = self._clean_scene_text(self._pick(item, "text", "draft", "content", default=""))
            if not text:
                continue
            candidates.append(DraftCandidate(text=text, notes=self._string_list(self._pick(item, "notes", "reasoning", default=[]))))
        return candidates

    def _normalize_issues(self, value: Any) -> List[ConsistencyIssue]:
        items = value if isinstance(value, list) else []
        issues: List[ConsistencyIssue] = []
        for item in items:
            if isinstance(item, str):
                issues.append(
                    ConsistencyIssue(
                        issue_type="model_reported_issue",
                        severity=Severity.LOW,
                        message=item,
                    )
                )
                continue
            if not isinstance(item, dict):
                continue
            issues.append(
                ConsistencyIssue(
                    issue_type=str(self._pick(item, "issue_type", "type", "category", default="model_reported_issue")),
                    severity=self._severity_value(self._pick(item, "severity", "level", default="low")),
                    message=str(self._pick(item, "message", "description", "issue", default="Model reported an issue.")),
                    evidence_span=str(self._pick(item, "evidence_span", "evidence", "span", default="")),
                    suggested_fix=str(self._pick(item, "suggested_fix", "fix", "suggestion", default="")),
                )
            )
        return issues

    def _normalize_consistency_report(self, data: Any) -> ConsistencyReport:
        payload = data if isinstance(data, dict) else {}
        score = self._float_value(self._pick(payload, "score", "consistency_score", default=0.8), 0.8)
        world_score = self._float_value(self._pick(payload, "world_plausibility_score", "plausibility", default=score), score)
        return ConsistencyReport(
            score=round(clamp(score), 3),
            issues=self._normalize_issues(self._pick(payload, "issues", "problems", "warnings", default=[])),
            notes=self._string_list(self._pick(payload, "notes", "reasoning", "summary", default=[])),
            world_plausibility_score=round(clamp(world_score), 3),
        )

    def _normalize_creativity_report(self, data: Any) -> CreativityReport:
        payload = data if isinstance(data, dict) else {}
        overall = self._float_value(self._pick(payload, "score", "creativity_score", default=0.6), 0.6)
        novelty = self._float_value(self._pick(payload, "novelty_score", "novelty", default=overall), overall)
        hook = self._float_value(self._pick(payload, "hook_score", "hook", default=overall), overall)
        emotion = self._float_value(self._pick(payload, "emotional_depth_score", "emotional_depth", "emotion_score", default=overall), overall)
        style = self._float_value(self._pick(payload, "style_fit_score", "style_fit", "style_score", default=overall), overall)
        surprise = self._float_value(self._pick(payload, "surprise_score", "surprise", default=(novelty * hook) ** 0.5), (novelty * hook) ** 0.5)
        return CreativityReport(
            novelty_score=round(clamp(novelty), 3),
            hook_score=round(clamp(hook), 3),
            emotional_depth_score=round(clamp(emotion), 3),
            style_fit_score=round(clamp(style), 3),
            surprise_score=round(clamp(surprise), 3),
            notes=self._string_list(self._pick(payload, "notes", "reasoning", "summary", default=[])),
        )

    def _normalize_revision_output(self, data: Any, original_text: str) -> RevisionOutput:
        payload = data if isinstance(data, dict) else {}
        revised_text = self._clean_scene_text(self._pick(payload, "revised_text", "text", "revision_text", default=original_text)) or original_text
        return RevisionOutput(
            revised_text=revised_text,
            change_log=self._string_list(self._pick(payload, "change_log", "changes", "notes", default=[])),
            fixed_issue_types=self._string_list(self._pick(payload, "fixed_issue_types", "fixed_issues", default=[])),
        )

    def _normalize_kg_edges(self, value: Any) -> List[KGEdge]:
        items = value if isinstance(value, list) else []
        edges: List[KGEdge] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            edges.append(
                KGEdge(
                    source=str(self._pick(item, "source", "subject", default="")),
                    relation=str(self._pick(item, "relation", "predicate", default="related_to")),
                    target=str(self._pick(item, "target", "object", default="")),
                )
            )
        return [edge for edge in edges if edge.source and edge.target]

    def _normalize_mapping_of_lists(self, value: Any) -> Dict[str, List[str]]:
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, List[str]] = {}
        for key, item in value.items():
            values = self._string_list(item)
            if values:
                normalized[str(key)] = values
        return normalized

    @staticmethod
    def _normalize_mapping_of_strings(value: Any) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, str] = {}
        for key, item in value.items():
            if isinstance(item, dict):
                text = "; ".join(f"{sub_key}: {sub_value}" for sub_key, sub_value in item.items())
            elif isinstance(item, list):
                text = ", ".join(str(entry) for entry in item if str(entry).strip())
            else:
                text = str(item or "").strip()
            if text:
                normalized[str(key)] = text
        return normalized

    def _normalize_extraction_output(self, data: Any, text: str) -> ExtractionOutput:
        payload = data if isinstance(data, dict) else {}
        return ExtractionOutput(
            summary=str(self._pick(payload, "summary", "scene_summary", default=short_text(text, 220))),
            new_static_facts=self._string_list(self._pick(payload, "new_static_facts", "static_facts", default=[])),
            state_updates=self._pick(payload, "state_updates", "state", default={}) if isinstance(self._pick(payload, "state_updates", "state", default={}), dict) else {},
            new_threads=self._string_list(self._pick(payload, "new_threads", "open_threads", default=[])),
            resolved_threads=self._string_list(self._pick(payload, "resolved_threads", "closed_threads", default=[])),
            knowledge_updates=self._normalize_mapping_of_lists(self._pick(payload, "knowledge_updates", "knowledge", default={})),
            inventory_updates=self._normalize_mapping_of_lists(self._pick(payload, "inventory_updates", "inventory", default={})),
            emotional_updates=self._normalize_mapping_of_strings(self._pick(payload, "emotional_updates", "emotions", default={})),
            kg_edges=self._normalize_kg_edges(self._pick(payload, "kg_edges", "edges", default=[])),
        )

    def _normalize_world_model_forecast(self, data: Any) -> WorldModelForecast:
        payload = data if isinstance(data, dict) else {}
        next_state_raw = self._pick(payload, "next_state", "state", "predicted_next_state", default={})
        next_state_payload = next_state_raw if isinstance(next_state_raw, dict) else {}
        next_state: Dict[str, Any] = {}

        if "last_scene_index" in next_state_payload:
            next_state["last_scene_index"] = self._int_value(next_state_payload.get("last_scene_index"), 0)
        for key in ["current_time_label", "current_location"]:
            value = str(next_state_payload.get(key, "") or "").strip()
            if value:
                next_state[key] = value
        for key in ["active_threads", "resolved_threads", "summary_memory"]:
            values = self._string_list(next_state_payload.get(key, []))
            if values:
                next_state[key] = values

        knowledge = self._normalize_mapping_of_lists(next_state_payload.get("character_knowledge", {}))
        if knowledge:
            next_state["character_knowledge"] = knowledge
        inventory = self._normalize_mapping_of_lists(next_state_payload.get("inventory", {}))
        if inventory:
            next_state["inventory"] = inventory
        emotional = self._normalize_mapping_of_strings(next_state_payload.get("emotional_state", {}))
        if emotional:
            next_state["emotional_state"] = emotional

        extraction_raw = self._pick(payload, "extraction", "predicted_extraction", default={})
        extraction = {}
        if isinstance(extraction_raw, dict):
            extraction = self._normalize_extraction_output(extraction_raw, "").model_dump(mode="json")

        return WorldModelForecast(
            next_state=next_state,
            extraction=extraction,
            notes=self._string_list(self._pick(payload, "notes", "reasoning", "summary", default=[])),
        )

    def _chat_json(self, role: str, system: str, payload: Dict[str, Any], max_tokens: int | None = None) -> Any:
        if self._local_role_model_path(role):
            return self._chat_local_role_json(role, system, payload, max_tokens=max_tokens)
        messages = make_json_messages(system, payload)
        request_payload = {
            "model": self._role_model(role),
            "messages": messages,
            "temperature": self._role_temperature(role),
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if self.settings.use_response_format:
            request_payload["response_format"] = {"type": "json_object"}
        if self.settings.think is not None:
            request_payload["think"] = self.settings.think
        if self.settings.reasoning_effort is not None:
            request_payload["reasoning_effort"] = self.settings.reasoning_effort
        cache_key = stable_hash({"role": role, "payload": request_payload, "base_url": self.settings.base_url})
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached

        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        try:
            response = self.client.post(url, json=request_payload, headers=headers)
            if response.status_code >= 400:
                # Retry without response_format for compatibility with some local servers.
                fallback_payload = dict(request_payload)
                fallback_payload.pop("response_format", None)
                response = self.client.post(url, json=fallback_payload, headers=headers)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - depends on external local server
            raise RuntimeError(f"Failed to call local model server: {exc}") from exc

        data = response.json()
        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected response shape: {data}") from exc
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not str(content or "").strip():
            reasoning = ""
            if isinstance(message, dict):
                reasoning = str(message.get("reasoning") or message.get("thinking") or "").strip()
            if reasoning:
                raise RuntimeError("Model returned thinking trace without final JSON content. Disable thinking for this model, for example by setting think=false.")
        parsed = extract_json_object(content)
        self._cache_write(cache_key, parsed)
        return parsed

    def health(self) -> Tuple[bool, str]:
        base = self.settings.base_url.rstrip("/")
        for candidate in [base + "/models", base.replace("/v1", "") + "/health"]:
            try:
                response = self.client.get(candidate)
                if response.status_code < 500:
                    return True, f"connected to {candidate}"
            except Exception:
                continue
        return False, f"could not reach {base}"

    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        payload = {
            "memory": compact_memory(memory_bundle),
            "scene_count": scene_count,
            "schema": {
                "items": [{
                    "id": "string",
                    "scene_index": "int",
                    "title": "string",
                    "pov": "string",
                    "location": "string",
                    "time_label": "string",
                    "goal": "string",
                    "beat": "string",
                    "foreshadowing": ["string"],
                    "required_facts": ["string"],
                    "status": "pending"
                }]
            },
        }
        data = self._chat_json(
            "planner",
            "너는 장편 소설의 시퀀스 플래너다. 반드시 JSON 객체로만 답하고 key는 English snake_case를 사용해. 요청한 scene_count를 넘기지 말고 각 필드는 짧고 구체적으로 써라.",
            payload,
            max_tokens=max(self.settings.max_tokens, 3072),
        )
        return self._normalize_outline_cards(data, memory_bundle, scene_count)

    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "schema": {
                "scene_title": "string",
                "beat_sheet": ["string"],
                "must_include": ["string"],
                "reasoning": ["string"],
                "target_word_count": "int",
            },
        }
        data = self._chat_json(
            "planner",
            "너는 scene planner다. 인물, 시간, 장소, 필수 사실을 잃지 말고 JSON만 반환해. plan은 간결하게 유지하고 beat_sheet와 reasoning은 짧은 bullet 수준으로 써라.",
            payload,
            max_tokens=max(self.settings.max_tokens, 3072),
        )
        return self._normalize_plan_output(data, request)

    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, count: int = 3) -> List[DraftCandidate]:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "count": max(1, count),
            "schema": {
                "items": [{"text": "string", "notes": ["string"]}],
            },
        }
        data = self._chat_json(
            "writer",
            "너는 scene writer다. 각 후보는 plan.target_word_count 근처의 짧고 응축된 한국어 산문으로 쓰고 JSON만 반환해. 설명, 코드펜스, 주석 없이 items[].text에 산문만 넣어라.",
            payload,
            max_tokens=max(self.settings.max_tokens, 4096),
        )
        return self._normalize_candidates(data)

    def critique_consistency(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ConsistencyReport:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "rubric": [
                "location and time anchors",
                "required facts and must-include items",
                "goal and beat alignment",
                "scene transition cues",
                "thread continuity",
                "world-rule compliance",
                "state-delta compression",
            ],
            "schema": {
                "score": 0.0,
                "world_plausibility_score": 0.0,
                "notes": ["string"],
                "issues": [{
                    "issue_type": "string",
                    "severity": "low|medium|high",
                    "message": "string",
                    "evidence_span": "string",
                    "suggested_fix": "string",
                }],
            },
        }
        data = self._chat_json(
            "consistency_critic",
            "너는 continuity critic이다. 설정, 시간, 장소, 인지 누수, 목표 불일치를 검사하고 JSON만 반환해.",
            payload,
        )
        return self._normalize_consistency_report(data)

    def critique_creativity(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> CreativityReport:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "schema": {
                "novelty_score": 0.0,
                "hook_score": 0.0,
                "emotional_depth_score": 0.0,
                "style_fit_score": 0.0,
                "surprise_score": 0.0,
                "notes": ["string"],
            },
        }
        data = self._chat_json(
            "creativity_critic",
            "너는 literary critic이다. 새로움, 갈고리, 감정 깊이, 문체 적합도를 평가하고 JSON만 반환해.",
            payload,
        )
        return self._normalize_creativity_report(data)

    def revise_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str, issues: Sequence[ConsistencyIssue]) -> RevisionOutput:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "issues": [issue.model_dump() for issue in issues],
            "schema": {
                "revised_text": "string",
                "change_log": ["string"],
                "fixed_issue_types": ["string"],
            },
        }
        data = self._chat_json(
            "revision",
            "너는 revision writer다. 원문의 문체를 가능한 유지하면서 이슈만 수정해. revised_text는 원문과 비슷한 길이의 산문만 포함하고 코드펜스나 설명을 넣지 마라. JSON만 반환해.",
            payload,
            max_tokens=max(self.settings.max_tokens, 4096),
        )
        return self._normalize_revision_output(data, text)

    def extract_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ExtractionOutput:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "schema": {
                "summary": "string",
                "new_static_facts": ["string"],
                "state_updates": {},
                "new_threads": ["string"],
                "resolved_threads": ["string"],
                "knowledge_updates": {},
                "inventory_updates": {},
                "emotional_updates": {},
                "kg_edges": [{"source": "string", "relation": "string", "target": "string"}],
            },
        }
        data = self._chat_json(
            "extractor",
            "너는 narrative state extractor다. 장면에서 추상 상태 변화만 뽑아 JSON으로 반환해. 값은 짧고 구조적으로 유지해.",
            payload,
        )
        return self._normalize_extraction_output(data, text)

    def forecast_world_model(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> WorldModelForecast | None:
        if not self._has_explicit_role_model("world_model"):
            return None
        compact_request = {
            "pov": request.pov,
            "location": request.location,
            "time_label": request.time_label,
            "goal": request.goal,
            "foreshadowing": request.foreshadowing,
            "required_facts": request.required_facts,
        }
        compact_plan = {
            "scene_title": plan.scene_title,
            "must_include": plan.must_include,
        }
        previous_state = memory_bundle.get("state", {}) if isinstance(memory_bundle.get("state"), dict) else {}
        compact_previous_state = {
            "current_time_label": previous_state.get("current_time_label", ""),
            "current_location": previous_state.get("current_location", ""),
            "active_threads": previous_state.get("active_threads", []),
            "resolved_threads": previous_state.get("resolved_threads", []),
            "inventory": previous_state.get("inventory", {}),
            "emotional_state": previous_state.get("emotional_state", {}),
        }
        payload = {
            "request": compact_request,
            "plan": compact_plan,
            "accepted_text": short_text(text, 1600),
            "previous_state": compact_previous_state,
            "requirements": [
                "Predict the next story state implied by the scene.",
                "Return only JSON.",
                "Keep fields concise and structured.",
            ],
        }
        data = self._chat_structured(
            "world_model",
            "You are a narrative world model. Predict the next state implied by the scene and return JSON only.",
            payload,
            WorldModelForecast.model_json_schema(),
            max_tokens=min(max(self.settings.max_tokens, 768), 1024),
        )
        return self._normalize_world_model_forecast(data)

    def forecast_world_model(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> WorldModelForecast | None:
        if not self._has_explicit_role_model("world_model"):
            return None
        compact_request = {
            "pov": request.pov,
            "location": request.location,
            "time_label": request.time_label,
            "goal": request.goal,
            "foreshadowing": request.foreshadowing,
            "required_facts": request.required_facts,
        }
        compact_plan = {
            "scene_title": plan.scene_title,
            "must_include": plan.must_include,
        }
        previous_state = memory_bundle.get("state", {}) if isinstance(memory_bundle.get("state"), dict) else {}
        compact_previous_state = {
            "current_time_label": previous_state.get("current_time_label", ""),
            "current_location": previous_state.get("current_location", ""),
            "active_threads": previous_state.get("active_threads", []),
            "resolved_threads": previous_state.get("resolved_threads", []),
            "inventory": previous_state.get("inventory", {}),
            "emotional_state": previous_state.get("emotional_state", {}),
        }
        payload = {
            "request": compact_request,
            "plan": compact_plan,
            "accepted_text": short_text(text, 1600),
            "previous_state": compact_previous_state,
            "schema": {
                "next_state": {},
                "extraction": {
                    "summary": "string",
                    "new_static_facts": ["string"],
                    "state_updates": {},
                    "new_threads": ["string"],
                    "resolved_threads": ["string"],
                    "knowledge_updates": {},
                    "inventory_updates": {},
                    "emotional_updates": {},
                    "kg_edges": [{"source": "string", "relation": "string", "target": "string"}],
                },
                "notes": ["string"],
            },
        }
        data = self._chat_json(
            "world_model",
            "You are a narrative world model. Predict the next state implied by the scene and return JSON only.",
            payload,
            max_tokens=min(max(self.settings.max_tokens, 768), 1024),
        )
        return self._normalize_world_model_forecast(data)


class _OutlineItems(BaseModel):
    items: List[OutlineCard]


class _DraftCandidates(BaseModel):
    items: List[DraftCandidate]


class OllamaNativeProvider(OpenAICompatibleProvider):
    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__(settings)

    def _repair_raw_structured_output(
        self,
        role: str,
        system: str,
        payload: Dict[str, Any],
        raw_text: str,
        schema: Dict[str, Any] | str,
        max_tokens: int | None = None,
    ) -> Any:
        model_name = self._remote_repair_model_name()
        messages = make_json_messages(
            "You repair malformed structured model outputs. Convert the raw text into valid JSON that matches the provided schema. Return JSON only.",
            {
                "role": role,
                "original_system": system,
                "original_payload": payload,
                "raw_text": raw_text,
                "schema": schema,
            },
        )
        request_payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens or self.settings.max_tokens,
            },
        }
        if self.settings.think is not None:
            request_payload["think"] = False if isinstance(self.settings.think, bool) else self.settings.think

        url = self.settings.base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        url = url.rstrip("/") + "/api/chat"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        response = self.client.post(url, json=request_payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        content = data["message"].get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return extract_json_object(content)

    def _chat_structured(
        self,
        role: str,
        system: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | str,
        max_tokens: int | None = None,
    ) -> Any:
        if self._local_role_model_path(role):
            return self._chat_local_role_json(role, system, payload, max_tokens=max_tokens, schema=schema)
        messages = make_json_messages(system, payload)
        request_payload: Dict[str, Any] = {
            "model": self._role_model(role),
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": self._role_temperature(role),
                "num_predict": max_tokens or self.settings.max_tokens,
            },
        }
        if self.settings.think is not None:
            request_payload["think"] = self.settings.think

        cache_key = stable_hash({"role": role, "payload": request_payload, "base_url": self.settings.base_url})
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached

        url = self.settings.base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        url = url.rstrip("/") + "/api/chat"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        try:
            response = self.client.post(url, json=request_payload, headers=headers)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - depends on local server
            raise RuntimeError(f"Failed to call Ollama native API: {exc}") from exc

        data = response.json()
        try:
            message = data["message"]
            content = message.get("content")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected Ollama response shape: {data}") from exc
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        parsed = extract_json_object(content)
        self._cache_write(cache_key, parsed)
        return parsed

    def health(self) -> Tuple[bool, str]:
        base = self.settings.base_url.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        for candidate in [root + "/api/tags", base + "/models", root + "/api/version"]:
            try:
                response = self.client.get(candidate)
                if response.status_code < 500:
                    return True, f"connected to {candidate}"
            except Exception:
                continue
        return False, f"could not reach {root}"

    def generate_outline(self, memory_bundle: Dict[str, Any], scene_count: int) -> List[OutlineCard]:
        payload = {
            "memory": compact_memory(memory_bundle),
            "scene_count": scene_count,
            "requirements": [
                "Return exactly the requested number of scenes when possible.",
                "Use concise Korean strings.",
                "Keep each field short and concrete.",
            ],
        }
        data = self._chat_structured(
            "planner",
            "너는 장편 소설의 시퀀스 플래너다. 반드시 JSON만 반환해.",
            payload,
            _OutlineItems.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 2048),
        )
        return self._normalize_outline_cards(data, memory_bundle, scene_count)

    def plan_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest) -> PlanOutput:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "requirements": [
                "Keep beat_sheet and reasoning concise.",
                "Respect the min/max word constraints.",
            ],
        }
        data = self._chat_structured(
            "planner",
            "너는 scene planner다. 장면 계획을 JSON으로만 반환해.",
            payload,
            PlanOutput.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 1536),
        )
        return self._normalize_plan_output(data, request)

    def write_candidates(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, count: int = 3) -> List[DraftCandidate]:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "count": max(1, count),
            "requirements": [
                "Write Korean prose only in items[].text.",
                "Do not include chain-of-thought, explanations, or markdown fences.",
                "Stay near the target length.",
            ],
        }
        data = self._chat_structured(
            "writer",
            "너는 scene writer다. 응축된 한국어 산문 후보를 JSON으로만 반환해.",
            payload,
            _DraftCandidates.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 3072),
        )
        return self._normalize_candidates(data)

    def critique_consistency(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ConsistencyReport:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "rubric": [
                "location and time anchors",
                "required facts and must-include items",
                "goal and beat alignment",
                "scene transition cues",
                "thread continuity",
                "world-rule compliance",
                "state-delta compression",
            ],
        }
        data = self._chat_structured(
            "consistency_critic",
            "너는 continuity critic이다. 설정, 시간, 장소, 전환, 필수 사실 누락을 검사하고 JSON으로만 반환해.",
            payload,
            ConsistencyReport.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 1536),
        )
        return self._normalize_consistency_report(data)

    def critique_creativity(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> CreativityReport:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
        }
        data = self._chat_structured(
            "creativity_critic",
            "너는 literary critic이다. 창의성과 감정 밀도를 평가하고 JSON으로만 반환해.",
            payload,
            CreativityReport.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 1024),
        )
        return self._normalize_creativity_report(data)

    def revise_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str, issues: Sequence[ConsistencyIssue]) -> RevisionOutput:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "requirements": [
                "Only revise the scene text.",
                "Do not add markdown or explanations.",
            ],
        }
        data = self._chat_structured(
            "revision",
            "너는 revision writer다. 문제를 고친 수정본을 JSON으로만 반환해.",
            payload,
            RevisionOutput.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 3072),
        )
        return self._normalize_revision_output(data, text)

    def extract_scene(self, memory_bundle: Dict[str, Any], request: SceneRequest, plan: PlanOutput, text: str) -> ExtractionOutput:
        payload = {
            "memory": compact_memory(memory_bundle),
            "request": request.model_dump(),
            "plan": plan.model_dump(),
            "text": text,
            "requirements": [
                "Keep extracted values brief and structured.",
            ],
        }
        data = self._chat_structured(
            "extractor",
            "너는 narrative state extractor다. 장면의 상태 변화를 JSON으로만 반환해.",
            payload,
            ExtractionOutput.model_json_schema(),
            max_tokens=max(self.settings.max_tokens, 2048),
        )
        return self._normalize_extraction_output(data, text)


def build_provider(settings: RuntimeSettings) -> BaseLLMProvider:
    if settings.provider == "mock":
        return MockProvider(settings)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if settings.provider == "ollama":
        return OllamaNativeProvider(settings)
    raise ValueError(f"Unsupported provider: {settings.provider}")
