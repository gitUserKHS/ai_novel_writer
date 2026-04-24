from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class PoolType(str, Enum):
    ACCEPTED = "accepted"
    PAIRWISE = "pairwise"
    HARD_NEGATIVE = "hard_negative"
    PROMPT_ONLY = "prompt_only"


class IssueType(str, Enum):
    TIMELINE = "timeline"
    LOCATION = "location"
    CAUSALITY = "causality"
    KNOWLEDGE_LEAK = "knowledge_leak"
    INVENTORY = "inventory"
    CHARACTER = "character"
    RULE = "rule"
    STYLE = "style"
    COVERAGE = "coverage"
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProviderType(str, Enum):
    MOCK = "mock"
    OPENAI_COMPATIBLE = "openai_compatible"


class RuntimeSettings(BaseModel):
    provider: ProviderType = ProviderType.MOCK
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    timeout_seconds: int = Field(default=180, ge=1)
    temperature_planner: float = 0.2
    temperature_writer: float = 0.85
    temperature_critic: float = 0.2
    temperature_revision: float = 0.4
    candidate_count: int = Field(default=3, ge=1, le=4)


class StoryCreate(BaseModel):
    title: str
    genre: str = "literary fiction"
    premise: str
    tone: str = "lyrical and emotionally grounded"
    themes: List[str] = Field(default_factory=list)
    characters: List[str] = Field(default_factory=list)
    forbidden_facts: List[str] = Field(default_factory=list)
    notes: str = ""
    target_length_scenes: int = 12


class StoryUpdate(BaseModel):
    title: Optional[str] = None
    genre: Optional[str] = None
    premise: Optional[str] = None
    tone: Optional[str] = None
    themes: Optional[List[str]] = None
    characters: Optional[List[str]] = None
    forbidden_facts: Optional[List[str]] = None
    notes: Optional[str] = None
    target_length_scenes: Optional[int] = None


class StoryOut(BaseModel):
    id: str
    title: str
    genre: str
    premise: str
    tone: str
    themes: List[str]
    characters: List[str]
    forbidden_facts: List[str]
    notes: str
    target_length_scenes: int
    created_at: str
    updated_at: str


class BibleContent(BaseModel):
    static_facts: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    motifs: List[str] = Field(default_factory=list)
    voice_notes: List[str] = Field(default_factory=list)
    reference_snippets: List[str] = Field(default_factory=list)


class OutlineCard(BaseModel):
    id: Optional[str] = None
    title: str
    pov: str
    goal: str
    location: str
    time_label: str
    summary_request: str
    beats: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)
    status: str = "planned"


class OutlineGenerateRequest(BaseModel):
    scene_count: int = Field(default=6, ge=1, le=20)


class QuickstartRequest(BaseModel):
    prompt: str = Field(min_length=8, max_length=4000)
    scene_count: int = Field(default=4, ge=1, le=12)
    desired_length_words: int = Field(default=900, ge=250, le=3000)


class SceneRequest(BaseModel):
    title: str = ""
    pov: str
    goal: str
    location: str
    time_label: str
    summary_request: str
    beats: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)
    emotion_targets: List[str] = Field(default_factory=list)
    desired_length_words: int = Field(default=900, ge=250, le=3000)
    outline_card_id: Optional[str] = None


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


class PlanBeat(BaseModel):
    label: str
    purpose: str


class FutureStatePrediction(BaseModel):
    horizon: int = Field(default=1, ge=1, le=12)
    state_summary: str = ""
    required_setup: List[str] = Field(default_factory=list)
    payoff_signals: List[str] = Field(default_factory=list)
    risk_notes: List[str] = Field(default_factory=list)


class ReversePrerequisite(BaseModel):
    target: str = ""
    prerequisite: str = ""
    reason: str = ""


class PlanOutput(BaseModel):
    scene_title: str
    synopsis: str
    beats: List[PlanBeat] = Field(default_factory=list)
    expected_reveals: List[str] = Field(default_factory=list)
    expected_new_threads: List[str] = Field(default_factory=list)
    expected_resolved_threads: List[str] = Field(default_factory=list)
    expected_state_delta: Dict[str, Any] = Field(default_factory=dict)
    future_state_predictions: List[FutureStatePrediction] = Field(default_factory=list)
    backward_prerequisites: List[ReversePrerequisite] = Field(default_factory=list)
    payoff_targets: List[str] = Field(default_factory=list)
    contradiction_risks: List[str] = Field(default_factory=list)


class DraftCandidate(BaseModel):
    text: str
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    strategy: str = ""


class ConsistencyIssue(BaseModel):
    issue_type: IssueType = IssueType.OTHER
    severity: Severity = Severity.LOW
    message: str
    evidence: str = ""
    suggested_fix: str = ""


class ConsistencyReport(BaseModel):
    score: float = 0.0
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    checks_run: List[str] = Field(default_factory=list)
    verdict: str = "needs_revision"


class CreativityReport(BaseModel):
    novelty_score: float = 0.0
    hook_score: float = 0.0
    emotional_depth_score: float = 0.0
    language_score: float = 0.0
    summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)


class RevisionOutput(BaseModel):
    revised_text: str
    change_log: List[str] = Field(default_factory=list)
    addressed_issues: List[str] = Field(default_factory=list)


class KGEdge(BaseModel):
    source: str
    relation: str
    target: str
    edge_type: str = "event"
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
    tags: List[str] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    candidate_index: int
    draft: DraftCandidate
    consistency: ConsistencyReport
    creativity: CreativityReport
    combined_score: float
    revised: Optional[RevisionOutput] = None


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
    created_at: str


class SceneOut(BaseModel):
    id: str
    scene_index: int
    title: str
    pov: str
    location: str
    time_label: str
    goal: str
    accepted_text: str
    summary: str
    created_at: str
    plan: Dict[str, Any]
    extraction: Dict[str, Any]
    consistency: Dict[str, Any]
    creativity: Dict[str, Any]
    revision: Optional[Dict[str, Any]]
    candidates: List[Dict[str, Any]] = Field(default_factory=list)


class JobOut(BaseModel):
    id: str
    story_id: str
    kind: str
    status: JobStatus
    progress: float
    message: str
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error_text: str = ""
    created_at: str
    updated_at: str


class ArtifactOut(BaseModel):
    id: int
    artifact_type: str
    path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class EvaluationReport(BaseModel):
    story_id: str
    scene_count: int
    average_consistency_score: float
    average_novelty_score: float
    average_hook_score: float
    average_emotional_depth_score: float
    unresolved_threads: List[str] = Field(default_factory=list)
    resolved_threads: List[str] = Field(default_factory=list)
    issue_counts: Dict[str, int] = Field(default_factory=dict)
    dataset_counts: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class ContinueStoryRequest(BaseModel):
    desired_length_words: int = Field(default=900, ge=250, le=3000)


class ApiEnvelope(BaseModel):
    ok: bool = True
    message: str = ""
    data: Optional[Any] = None


class HealthOut(BaseModel):
    status: str
    provider: str
    model: str
    database_ok: bool
    backend_ok: bool
    detail: str = ""


class AutoConnectOut(BaseModel):
    found: bool = False
    source: str = ""
    detail: str = ""
    settings: Optional[RuntimeSettings] = None
    available_models: List[str] = Field(default_factory=list)


class LocalModelOption(BaseModel):
    source: str
    base_url: str
    model: str


class LocalModelCatalogOut(BaseModel):
    options: List[LocalModelOption] = Field(default_factory=list)
    current: Optional[LocalModelOption] = None
    detail: str = ""


class ModelSelectRequest(BaseModel):
    provider: ProviderType = ProviderType.OPENAI_COMPATIBLE
    base_url: str = ""
    model: str = ""


class TrainingEnvironmentOut(BaseModel):
    preferred_python_version: str = ""
    preferred_python_path: str = ""
    training_env_dir: str = ""
    training_python_path: str = ""
    training_env_exists: bool = False
    gpu_available: bool = False
    gpu_name: str = ""
    python_version: str = ""
    torch_installed: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_total_gib: float = 0.0
    cuda_free_gib: float = 0.0
    bitsandbytes_ok: bool = False
    transformers_ok: bool = False
    peft_ok: bool = False
    ready: bool = False
    training_profile: str = ""
    detail: str = ""


class TrainingSetupRequest(BaseModel):
    force_reinstall: bool = False


class TrainedAdapterOut(BaseModel):
    story_id: str = ""
    run_id: str = ""
    base_model: str = ""
    adapter_dir: str = ""
    metadata_path: str = ""
    created_at: str = ""
    training_profile: str = ""


class UseTrainedAdapterRequest(BaseModel):
    story_id: str = ""
    adapter_dir: str = ""
    host: str = "127.0.0.1"
    port: int = Field(default=5001, ge=1024, le=65535)


class UseTrainedAdapterOut(BaseModel):
    ok: bool = True
    detail: str = ""
    settings: RuntimeSettings
    adapter: TrainedAdapterOut
    log_path: str = ""


class OneClickTrainingRequest(BaseModel):
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_token: str = ""
    epochs: float = Field(default=1.0, ge=0.05, le=20.0)
    per_device_batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=256)
    max_seq_length: int = Field(default=0, ge=0, le=32768)
    lora_r: int = Field(default=0, ge=0, le=256)
    lora_alpha: int = Field(default=0, ge=0, le=512)
    use_distillation: bool = True
    teacher_base_url: str = ""
    teacher_model: str = "google/gemma-4-E2B-it"
    teacher_api_key: str = ""
    teacher_coaching: bool = True
    teacher_variants_per_prompt: int = Field(default=1, ge=0, le=3)


class MemoryBundle(BaseModel):
    story: StoryOut
    bible: BibleContent
    state: StoryState
    recent_scenes: List[SceneOut] = Field(default_factory=list)
    outline: List[OutlineCard] = Field(default_factory=list)


class QuickstartOut(BaseModel):
    story: StoryOut
    bible: BibleContent
    state: StoryState
    outline: List[OutlineCard] = Field(default_factory=list)
    recent_scenes: List[SceneOut] = Field(default_factory=list)
    provider: ProviderType
    detail: str = ""


class GenerationResult(BaseModel):
    accepted_scene: AcceptedScene
    candidate_evaluations: List[CandidateEvaluation]
    updated_state: StoryState
    logs: List[Dict[str, Any]] = Field(default_factory=list)


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
