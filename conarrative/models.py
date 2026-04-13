
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StoryStatus(str, Enum):
    DRAFTING = "drafting"
    COMPLETED = "completed"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PoolType(str, Enum):
    PROMPT_ONLY = "prompt_only"
    ACCEPTED = "accepted"
    PAIRWISE = "pairwise"
    HARD_NEGATIVE = "hard_negative"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StoryCreate(BaseModel):
    id: Optional[str] = None
    title: str
    genre: str
    tone: str
    premise: str
    themes: List[str] = Field(default_factory=list)
    characters: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    target_scene_count: int = 6
    target_word_count: int = 8000
    language: str = "ko"

    @field_validator("themes", "characters", "constraints", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [line.strip() for line in value.replace(",", "\n").splitlines() if line.strip()]
        return [str(value)]


class StoryUpdate(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None
    tone: Optional[str] = None
    premise: Optional[str] = None
    themes: Optional[List[str]] = None
    characters: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    target_scene_count: Optional[int] = None
    target_word_count: Optional[int] = None
    language: Optional[str] = None
    status: Optional[StoryStatus] = None


class StoryMeta(StoryCreate):
    id: str
    status: StoryStatus = StoryStatus.DRAFTING
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class BibleContent(BaseModel):
    static_facts: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    forbidden: List[str] = Field(default_factory=list)
    motifs: List[str] = Field(default_factory=list)

    @field_validator("static_facts", "rules", "forbidden", "motifs", mode="before")
    @classmethod
    def _coerce_text_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [line.strip() for line in value.replace(",", "\n").splitlines() if line.strip()]
        return [str(value)]


class OutlineGenerateRequest(BaseModel):
    scene_count: int = 6


class OutlineCard(BaseModel):
    id: str
    scene_index: int
    title: str
    pov: str
    location: str
    time_label: str
    goal: str
    beat: str
    foreshadowing: List[str] = Field(default_factory=list)
    required_facts: List[str] = Field(default_factory=list)
    status: str = "pending"


class SceneRequest(BaseModel):
    title_hint: str = ""
    pov: str = ""
    location: str = ""
    time_label: str = ""
    goal: str = ""
    beat: str = ""
    foreshadowing: List[str] = Field(default_factory=list)
    required_facts: List[str] = Field(default_factory=list)
    outline_card_id: Optional[str] = None
    min_words: int = 320
    max_words: int = 680

    @field_validator("foreshadowing", "required_facts", mode="before")
    @classmethod
    def _coerce_list_fields(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [line.strip() for line in value.replace(",", "\n").splitlines() if line.strip()]
        return [str(value)]


class PlanOutput(BaseModel):
    scene_title: str
    beat_sheet: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    target_word_count: int = 500


class DraftCandidate(BaseModel):
    text: str
    notes: List[str] = Field(default_factory=list)


class ConsistencyIssue(BaseModel):
    issue_type: str
    severity: Severity
    message: str
    evidence_span: str = ""
    suggested_fix: str = ""


class ConsistencyReport(BaseModel):
    score: float = 0.8
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    world_plausibility_score: float = 0.8


class CreativityReport(BaseModel):
    novelty_score: float = 0.6
    hook_score: float = 0.6
    emotional_depth_score: float = 0.6
    style_fit_score: float = 0.7
    surprise_score: float = 0.5
    notes: List[str] = Field(default_factory=list)


class RevisionOutput(BaseModel):
    revised_text: str
    change_log: List[str] = Field(default_factory=list)
    fixed_issue_types: List[str] = Field(default_factory=list)


class KGEdge(BaseModel):
    source: str
    relation: str
    target: str
    scene_id: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionOutput(BaseModel):
    summary: str
    new_static_facts: List[str] = Field(default_factory=list)
    state_updates: Dict[str, Any] = Field(default_factory=dict)
    new_threads: List[str] = Field(default_factory=list)
    resolved_threads: List[str] = Field(default_factory=list)
    knowledge_updates: Dict[str, List[str]] = Field(default_factory=dict)
    inventory_updates: Dict[str, List[str]] = Field(default_factory=dict)
    emotional_updates: Dict[str, str] = Field(default_factory=dict)
    kg_edges: List[KGEdge] = Field(default_factory=list)


class StoryState(BaseModel):
    last_scene_index: int = 0
    current_time_label: str = ""
    current_location: str = ""
    active_threads: List[str] = Field(default_factory=list)
    resolved_threads: List[str] = Field(default_factory=list)
    character_knowledge: Dict[str, List[str]] = Field(default_factory=dict)
    inventory: Dict[str, List[str]] = Field(default_factory=dict)
    emotional_state: Dict[str, str] = Field(default_factory=dict)
    summary_memory: List[str] = Field(default_factory=list)


class WorldModelScore(BaseModel):
    plausibility: float = 0.8
    novelty: float = 0.6
    surprise: float = 0.6
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class WorldModelForecast(BaseModel):
    next_state: Dict[str, Any] = Field(default_factory=dict)
    extraction: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    candidate_index: int
    draft: DraftCandidate
    consistency: ConsistencyReport
    creativity: CreativityReport
    combined_score: float
    revised: Optional[RevisionOutput] = None
    world_model: Dict[str, Any] = Field(default_factory=dict)


class AcceptedScene(BaseModel):
    scene_id: str
    scene_index: int
    title: str
    pov: str
    location: str
    time_label: str
    goal: str
    request: SceneRequest
    plan: PlanOutput
    accepted_text: str
    summary: str
    extraction: ExtractionOutput
    consistency: ConsistencyReport
    creativity: CreativityReport
    revision: Optional[RevisionOutput] = None
    created_at: str = Field(default_factory=utcnow_iso)


class GenerationResult(BaseModel):
    accepted_scene: AcceptedScene
    candidate_evaluations: List[CandidateEvaluation]
    updated_state: StoryState
    logs: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    story_id: str
    scene_count: int
    average_consistency_score: float
    average_novelty_score: float
    average_hook_score: float
    average_emotional_depth_score: float
    average_world_plausibility_score: float
    average_surprise_score: float
    unresolved_threads: List[str] = Field(default_factory=list)
    resolved_threads: List[str] = Field(default_factory=list)
    issue_counts: Dict[str, int] = Field(default_factory=dict)
    dataset_counts: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class ArtifactRecord(BaseModel):
    id: str
    story_id: str
    kind: str
    path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)


class HealthOut(BaseModel):
    status: str
    provider: str
    model: str
    database_ok: bool = True
    backend_ok: bool = True
    detail: str = ""


class RuntimeSettings(BaseModel):
    provider: str = "mock"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "not-needed"
    model: str = "mock-story-engine"
    think: bool | str | None = None
    reasoning_effort: str | None = None
    timeout_seconds: int = 60
    temperature: float = 0.9
    critic_temperature: float = 0.2
    max_tokens: int = 2048
    use_response_format: bool = True
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    role_models: Dict[str, str] = Field(default_factory=dict)
    cache_responses: bool = False
    cache_dir: str = "workspace/cache"


class JobRecord(BaseModel):
    id: str
    job_type: str
    story_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class OneClickLoopRequest(BaseModel):
    preset: str = "qwen-native"
    mode: str = "smoke"
    train_action: str = "skip"
    train_preset: str = "none"
    story_file: str = "examples/story.yaml"
    scene_file: str = "examples/scene.yaml"
    story_id: Optional[str] = None
    scene_limit: Optional[int] = None
    run_tests: bool = False
    install_training_deps: bool = False


class GeneralistLoopRequest(BaseModel):
    preset: str = "qwen-loop"
    mode: str = "smoke"
    train_action: str = "skip"
    train_preset: str = "none"
    story_dir: str = "examples/story_pack"
    scene_file: Optional[str] = None
    corpus_output_dir: str = "outputs/generalist_corpus"
    validation_story_ratio: float = 0.34
    story_offset: int = 0
    story_limit: Optional[int] = None
    scene_limit: Optional[int] = None
    resume: bool = False
    run_tests: bool = False
    install_training_deps: bool = False


class TrainingRunRequest(BaseModel):
    config: str = "configs/training_qwen3_4b_sft.yaml"
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    output_dir: Optional[str] = None
    model_name_or_path: Optional[str] = None
    dry_run: bool = False
    print_config: bool = False


class HFPublishRequest(BaseModel):
    source_dir: str = "outputs/training_qwen3_4b_sft"
    repo_id: str = ""
    repo_type: str = "model"
    path_in_repo: str = ""
    revision: Optional[str] = None
    commit_message: Optional[str] = None
    private: bool = False
    exclude_checkpoints: bool = True
    ignore_patterns: List[str] = Field(default_factory=list)
    namespace: str = ""
    project: str = "conarrative"
    role: str = ""
    base_model: str = ""
    stage: str = ""
    auto_tag: bool = False
    release_tag: str = ""
    release_prefix: str = "v"
    bump: str = "patch"
    tag_message: str = ""


class HFPullRequest(BaseModel):
    repo_id: str
    repo_type: str = "model"
    local_dir: str = "outputs/hf_download"
    revision: Optional[str] = None
    allow_patterns: List[str] = Field(default_factory=list)
    ignore_patterns: List[str] = Field(default_factory=list)


class StoryImportRequest(BaseModel):
    yaml_text: str


class UIPresetSaveRequest(BaseModel):
    kind: str
    name: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class UIPresetRecord(BaseModel):
    kind: str
    name: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    saved_at: str = Field(default_factory=utcnow_iso)
