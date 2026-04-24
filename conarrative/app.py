from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, closing, contextmanager
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig
from .db import Storage
from .jobs import JobManager
from .llm import build_provider
from .models import (
    AutoConnectOut,
    BibleContent,
    ContinueStoryRequest,
    HealthOut,
    LocalModelCatalogOut,
    ModelSelectRequest,
    OneClickTrainingRequest,
    OutlineGenerateRequest,
    QuickstartOut,
    QuickstartRequest,
    ProviderType,
    RuntimeSettings,
    SceneRequest,
    StoryCreate,
    StoryUpdate,
    TrainingEnvironmentOut,
    TrainingSetupRequest,
    UseTrainedAdapterOut,
    UseTrainedAdapterRequest,
    JobStatus,
)
from .autodetect import auto_connect_settings, build_catalog_from_settings, detect_runtime_settings
from .orchestrator import Orchestrator
from .quickstart import build_story_from_prompt, continue_request_to_words, next_planned_outline_card, outline_to_scene_request, quickstart_settings
from .runtime_settings import RuntimeSettingsStore
from .train_runtime import ensure_training_environment, inspect_training_environment, run_one_click_training
from .trained_runtime import list_trained_adapters, select_trained_adapter, start_trained_adapter_server, stop_trained_servers


def create_app(config: AppConfig) -> FastAPI:
    storage = Storage(config.workspace.database_path)
    runtime_store = RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend)
    jobs = JobManager(storage)
    training_jobs = JobManager(storage)
    initial_catalog = LocalModelCatalogOut()
    trained_servers = {}

    def save_settings(settings: RuntimeSettings) -> RuntimeSettings:
        return runtime_store.save(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        saved_settings, catalog, changed = auto_connect_settings(current_settings())
        if changed:
            save_settings(saved_settings)
        app.state.model_catalog = build_catalog_from_settings(current_settings(), catalog)
        try:
            yield
        finally:
            stop_trained_servers(trained_servers)
            jobs.shutdown()
            training_jobs.shutdown()

    app = FastAPI(title=config.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config
    app.state.storage = storage
    app.state.runtime_store = runtime_store
    app.state.jobs = jobs
    app.state.training_jobs = training_jobs
    app.state.model_catalog = initial_catalog
    app.state.trained_servers = trained_servers

    web_dir = Path(__file__).parent / "web"
    static_dir = web_dir / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def current_settings() -> RuntimeSettings:
        return runtime_store.load()

    def current_catalog(refresh: bool = False) -> LocalModelCatalogOut:
        if refresh:
            catalog = build_catalog_from_settings(current_settings())
            app.state.model_catalog = catalog
            return catalog
        cached = getattr(app.state, "model_catalog", None)
        if isinstance(cached, LocalModelCatalogOut):
            return build_catalog_from_settings(current_settings(), cached)
        catalog = build_catalog_from_settings(current_settings())
        app.state.model_catalog = catalog
        return catalog

    def generation_settings(story_id: str = "") -> tuple[RuntimeSettings, str]:
        adapters = list_trained_adapters(config, story_id=story_id) if story_id else []
        if not adapters:
            adapters = list_trained_adapters(config)
        if adapters:
            adapter = adapters[0]
            try:
                trained_settings, _log_path = start_trained_adapter_server(
                    config=config,
                    adapter=adapter,
                    current_settings=current_settings(),
                    registry=app.state.trained_servers,
                )
                saved = save_settings(trained_settings)
                app.state.model_catalog = build_catalog_from_settings(saved)
                fast_settings = saved.model_copy(update={"candidate_count": 1}) if saved.candidate_count > 1 else saved
                return fast_settings, f"학습 모델로 생성합니다: {saved.model}"
            except Exception as exc:
                fallback, detail = quickstart_settings(current_settings())
                return fallback, f"학습 모델 자동 연결 실패: {exc}. {detail}"
        return quickstart_settings(current_settings())

    def resolve_teacher_request(payload: OneClickTrainingRequest) -> Dict[str, Any]:
        request = payload.model_dump()
        requested_model = (request.get("teacher_model") or "google/gemma-4-E2B-it").strip()
        requested_base_url = (request.get("teacher_base_url") or "").strip()
        if requested_base_url:
            return request

        catalog = current_catalog(refresh=True)
        teacher_option = _pick_teacher_option(catalog.options, requested_model)
        if teacher_option is not None:
            request["teacher_base_url"] = teacher_option.base_url
            request["teacher_model"] = teacher_option.model
            return request

        settings = current_settings()
        if settings.provider == ProviderType.OPENAI_COMPATIBLE and _looks_like_teacher_model(settings.model):
            request["teacher_base_url"] = settings.base_url
            request["teacher_model"] = settings.model
        return request

    @contextmanager
    def orchestrator_context(settings: RuntimeSettings | None = None) -> Iterator[Orchestrator]:
        provider = build_provider(settings or current_settings())
        try:
            yield Orchestrator(storage=storage, provider=provider, config=config)
        finally:
            provider.close()

    def ensure_story(story_id: str):
        story = storage.get_story(story_id)
        if story is None:
            raise HTTPException(status_code=404, detail=f"Story not found: {story_id}")
        return story

    def build_story_bundle(story_id: str) -> Dict[str, Any]:
        story = ensure_story(story_id)
        bible = storage.get_bible(story_id)
        state = storage.get_latest_state(story_id)
        scenes = storage.list_scenes(story_id)
        recent = scenes[-config.orchestration.recent_scene_memory :]
        outline = storage.list_outline(story_id)
        return {
            "story": story.model_dump(),
            "bible": bible.model_dump(),
            "state": state.model_dump(),
            "recent_scenes": [scene.model_dump() for scene in recent],
            "outline": [card.model_dump() for card in outline],
        }

    def quickstart_bundle(story_id: str, detail: str) -> Dict[str, Any]:
        bundle = build_story_bundle(story_id)
        return QuickstartOut(
            story=ensure_story(story_id),
            bible=storage.get_bible(story_id),
            state=storage.get_latest_state(story_id),
            outline=storage.list_outline(story_id),
            recent_scenes=storage.list_scenes(story_id)[-config.orchestration.recent_scene_memory :],
            provider=current_settings().provider,
            detail=detail,
        ).model_dump()

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse((web_dir / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        settings = current_settings()
        with closing(build_provider(settings)) as provider:
            backend_ok, detail = provider.health()
        return HealthOut(
            status="ok" if backend_ok else "degraded",
            provider=settings.provider,
            model=settings.model,
            database_ok=True,
            backend_ok=backend_ok,
            detail=detail,
        ).model_dump()

    @app.get("/api/runtime-settings")
    def get_runtime_settings() -> Dict[str, Any]:
        return current_settings().model_dump()

    @app.put("/api/runtime-settings")
    def put_runtime_settings(payload: RuntimeSettings) -> Dict[str, Any]:
        saved = save_settings(payload)
        app.state.model_catalog = build_catalog_from_settings(saved)
        return saved.model_dump()

    @app.post("/api/runtime-settings/test")
    def test_runtime_settings(payload: RuntimeSettings | None = None) -> Dict[str, Any]:
        settings = payload or current_settings()
        with closing(build_provider(settings)) as provider:
            ok, detail = provider.health()
        return {"ok": ok, "detail": detail, "settings": settings.model_dump()}

    @app.post("/api/runtime-settings/auto-connect")
    def auto_connect_runtime_settings() -> Dict[str, Any]:
        result = detect_runtime_settings(current_settings())
        if result.found and result.settings is not None:
            saved = save_settings(result.settings)
            app.state.model_catalog = build_catalog_from_settings(saved)
            return AutoConnectOut(
                found=True,
                source=result.source,
                detail=result.detail,
                settings=saved,
                available_models=result.available_models,
            ).model_dump()
        return result.model_dump()

    @app.get("/api/runtime-settings/models")
    def list_runtime_models() -> Dict[str, Any]:
        return current_catalog(refresh=True).model_dump()

    @app.get("/api/training/environment")
    def get_training_environment() -> Dict[str, Any]:
        return TrainingEnvironmentOut.model_validate(inspect_training_environment(config)).model_dump()

    @app.get("/api/training/adapters")
    def get_trained_adapters(story_id: str = "") -> Dict[str, Any]:
        return {"items": [adapter.model_dump() for adapter in list_trained_adapters(config, story_id=story_id)]}

    @app.post("/api/training/adapters/use")
    def use_trained_adapter(payload: UseTrainedAdapterRequest) -> Dict[str, Any]:
        adapter = select_trained_adapter(config, story_id=payload.story_id, adapter_dir=payload.adapter_dir)
        trained_settings, log_path = start_trained_adapter_server(
            config=config,
            adapter=adapter,
            current_settings=current_settings(),
            registry=app.state.trained_servers,
            host=payload.host,
            port=payload.port,
        )
        saved = save_settings(trained_settings)
        app.state.model_catalog = build_catalog_from_settings(saved)
        return UseTrainedAdapterOut(
            ok=True,
            detail=f"학습 모델을 생성 모델로 연결했습니다: {saved.model}",
            settings=saved,
            adapter=adapter,
            log_path=log_path,
        ).model_dump()

    @app.put("/api/runtime-settings/select-model")
    def select_runtime_model(payload: ModelSelectRequest) -> Dict[str, Any]:
        current = current_settings()
        if payload.provider == ProviderType.MOCK:
            saved = save_settings(RuntimeSettings.model_validate({**current.model_dump(), "provider": ProviderType.MOCK}))
            app.state.model_catalog = build_catalog_from_settings(saved)
            return saved.model_dump()
        if not payload.base_url or not payload.model:
            raise HTTPException(status_code=422, detail="base_url and model are required for openai_compatible selection.")
        saved = save_settings(
            RuntimeSettings.model_validate(
                {
                    **current.model_dump(),
                    "provider": ProviderType.OPENAI_COMPATIBLE,
                    "base_url": payload.base_url,
                    "model": payload.model,
                }
            )
        )
        app.state.model_catalog = build_catalog_from_settings(saved)
        return saved.model_dump()

    @app.get("/api/stories")
    def list_stories() -> Dict[str, Any]:
        return {"items": [story.model_dump() for story in storage.list_stories()]}

    @app.post("/api/stories")
    def create_story(payload: StoryCreate) -> Dict[str, Any]:
        story = storage.create_story(payload)
        return story.model_dump()

    @app.post("/api/quickstart")
    def quickstart_story(payload: QuickstartRequest) -> Dict[str, Any]:
        story = storage.create_story(build_story_from_prompt(payload))
        settings, detail = generation_settings()
        with orchestrator_context(settings) as orchestrator:
            outline = orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=payload.scene_count))
            first_card = next_planned_outline_card(outline)
            if first_card is not None:
                orchestrator.run_scene(story.id, outline_to_scene_request(first_card, payload.desired_length_words))
        return quickstart_bundle(story.id, detail)

    @app.post("/api/quickstart/job")
    def quickstart_story_job(payload: QuickstartRequest) -> Dict[str, Any]:
        story = storage.create_story(build_story_from_prompt(payload))
        settings, detail = generation_settings()
        job_id = str(uuid4())

        def task(log):
            log("스토리 생성 시작", 0.03)
            with orchestrator_context(settings) as orchestrator:
                log("아웃라인 생성 중", 0.12)
                outline = orchestrator.generate_outline(story.id, OutlineGenerateRequest(scene_count=payload.scene_count))
                log(f"아웃라인 {len(outline)}개 생성 완료", 0.24)
                first_card = next_planned_outline_card(outline)
                if first_card is not None:
                    orchestrator.run_scene(
                        story.id,
                        outline_to_scene_request(first_card, payload.desired_length_words),
                        log=lambda message, progress: log(message, 0.24 + progress * 0.74),
                    )
            return quickstart_bundle(story.id, detail)

        jobs.enqueue(job_id=job_id, story_id=story.id, kind="quickstart_generation", fn=task)
        job = storage.get_job(job_id)
        assert job is not None
        return {"job": job.model_dump(), "story": story.model_dump()}

    @app.get("/api/stories/{story_id}")
    def get_story(story_id: str) -> Dict[str, Any]:
        bundle = build_story_bundle(story_id)
        return {
            "story": bundle["story"],
            "bible": bundle["bible"],
            "state": bundle["state"],
            "outline": bundle["outline"],
            "recent_scenes": bundle["recent_scenes"],
            "dataset_counts": storage.dataset_counts(story_id),
            "artifact_count": len(storage.list_artifacts(story_id)),
        }

    @app.patch("/api/stories/{story_id}")
    def update_story(story_id: str, payload: StoryUpdate) -> Dict[str, Any]:
        ensure_story(story_id)
        story = storage.update_story(story_id, payload)
        if story is None:
            raise HTTPException(status_code=404, detail="Story not found")
        return story.model_dump()

    @app.delete("/api/stories/{story_id}")
    def delete_story(story_id: str) -> Dict[str, Any]:
        story = ensure_story(story_id)
        deleted = storage.delete_story(story_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Story not found")
        shutil.rmtree(Path(config.workspace.exports_dir) / story_id, ignore_errors=True)
        return {"deleted": True, "story_id": story_id, "title": story.title}

    @app.post("/api/stories/{story_id}/training/setup")
    def setup_story_training(story_id: str, payload: TrainingSetupRequest | None = None) -> Dict[str, Any]:
        ensure_story(story_id)
        job_id = str(uuid4())

        def task(log):
            status = ensure_training_environment(config, log=log, force_reinstall=bool((payload or TrainingSetupRequest()).force_reinstall))
            return {"environment": TrainingEnvironmentOut.model_validate(status).model_dump()}

        training_jobs.enqueue(job_id=job_id, story_id=story_id, kind="training_setup", fn=task)
        job = storage.get_job(job_id)
        assert job is not None
        return job.model_dump()

    @app.post("/api/stories/{story_id}/training/auto")
    def auto_train_story(story_id: str, payload: OneClickTrainingRequest) -> Dict[str, Any]:
        ensure_story(story_id)
        job_id = str(uuid4())

        def task(log):
            request = resolve_teacher_request(payload)
            result = run_one_click_training(
                config=config,
                storage=storage,
                story_id=story_id,
                runtime_settings=current_settings(),
                request=request,
                log=log,
            )
            log("학습 완료. 최신 학습 모델을 생성 모델로 자동 연결합니다.", 0.985)
            adapter = select_trained_adapter(config, story_id=story_id)
            trained_settings, log_path = start_trained_adapter_server(
                config=config,
                adapter=adapter,
                current_settings=current_settings(),
                registry=app.state.trained_servers,
            )
            saved = save_settings(trained_settings)
            app.state.model_catalog = build_catalog_from_settings(saved)
            result["auto_connected_adapter"] = adapter.model_dump()
            result["auto_connected_settings"] = saved.model_dump()
            result["adapter_server_log_path"] = log_path
            log(f"학습 모델 자동 연결 완료: {saved.model}", 1.0)
            return result

        training_jobs.enqueue(job_id=job_id, story_id=story_id, kind="story_training", fn=task)
        job = storage.get_job(job_id)
        assert job is not None
        return job.model_dump()

    @app.get("/api/stories/{story_id}/bible")
    def get_bible(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.get_bible(story_id).model_dump()

    @app.put("/api/stories/{story_id}/bible")
    def put_bible(story_id: str, payload: BibleContent) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.save_bible(story_id, payload).model_dump()

    @app.get("/api/stories/{story_id}/state")
    def get_state(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.get_latest_state(story_id).model_dump()

    @app.get("/api/stories/{story_id}/outline")
    def get_outline(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": [card.model_dump() for card in storage.list_outline(story_id)]}

    @app.post("/api/stories/{story_id}/outline/generate")
    def generate_outline(story_id: str, payload: OutlineGenerateRequest) -> Dict[str, Any]:
        ensure_story(story_id)
        settings, _detail = generation_settings(story_id)
        with orchestrator_context(settings) as orchestrator:
            cards = orchestrator.generate_outline(story_id, payload)
        return {"items": [card.model_dump() for card in cards]}

    @app.get("/api/stories/{story_id}/scenes")
    def list_scenes(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": [scene.model_dump() for scene in storage.list_scenes(story_id)]}

    @app.post("/api/stories/{story_id}/scenes/generate")
    def generate_scene(story_id: str, payload: SceneRequest) -> Dict[str, Any]:
        ensure_story(story_id)
        settings, _detail = generation_settings(story_id)
        job_id = str(uuid4())

        def task(log):
            with orchestrator_context(settings) as orchestrator:
                result = orchestrator.run_scene(story_id, payload, log=log)
                return result.model_dump()

        jobs.enqueue(job_id=job_id, story_id=story_id, kind="scene_generation", fn=task)
        job = storage.get_job(job_id)
        assert job is not None
        return job.model_dump()

    @app.post("/api/stories/{story_id}/continue")
    def continue_story(story_id: str, payload: ContinueStoryRequest | None = None) -> Dict[str, Any]:
        ensure_story(story_id)
        outline = storage.list_outline(story_id)
        next_card = next_planned_outline_card(outline)
        if next_card is None:
            raise HTTPException(status_code=400, detail="No unused outline cards remain for this story.")
        settings, detail = generation_settings(story_id)
        with orchestrator_context(settings) as orchestrator:
            orchestrator.run_scene(story_id, outline_to_scene_request(next_card, continue_request_to_words(payload or ContinueStoryRequest())))
        return quickstart_bundle(story_id, detail)

    @app.post("/api/stories/{story_id}/continue/job")
    def continue_story_job(story_id: str, payload: ContinueStoryRequest | None = None) -> Dict[str, Any]:
        ensure_story(story_id)
        outline = storage.list_outline(story_id)
        next_card = next_planned_outline_card(outline)
        if next_card is None:
            raise HTTPException(status_code=400, detail="No unused outline cards remain for this story.")
        settings, detail = generation_settings(story_id)
        job_id = str(uuid4())

        def task(log):
            with orchestrator_context(settings) as orchestrator:
                orchestrator.run_scene(
                    story_id,
                    outline_to_scene_request(next_card, continue_request_to_words(payload or ContinueStoryRequest())),
                    log=log,
                )
            return quickstart_bundle(story_id, detail)

        jobs.enqueue(job_id=job_id, story_id=story_id, kind="continue_generation", fn=task)
        job = storage.get_job(job_id)
        assert job is not None
        return job.model_dump()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = storage.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job.model_dump()

    @app.get("/api/jobs/{job_id}/events")
    async def stream_job_events(job_id: str):
        async def event_stream():
            last_log_count = -1
            while True:
                job = storage.get_job(job_id)
                if job is None:
                    payload = {"error": f"Job not found: {job_id}"}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return
                log_count = len(job.logs)
                terminal = job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}
                if log_count != last_log_count or terminal:
                    yield f"data: {json.dumps(job.model_dump(mode='json'), ensure_ascii=False)}\n\n"
                    last_log_count = log_count
                if terminal:
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/stories/{story_id}/kg")
    def get_kg(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": storage.list_kg_edges(story_id)}

    @app.get("/api/stories/{story_id}/datasets")
    def get_datasets(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.dataset_counts(story_id)

    @app.get("/api/stories/{story_id}/artifacts")
    def list_artifacts(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": [artifact.model_dump() for artifact in storage.list_artifacts(story_id)]}

    @app.post("/api/stories/{story_id}/export")
    def export_story(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        with orchestrator_context() as orchestrator:
            bundle = orchestrator.export_story_markdown(story_id)
        story_dir = Path(config.workspace.exports_dir) / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        output_path = story_dir / bundle["filename"]
        output_path.write_text(bundle["content"], encoding="utf-8")
        artifact = storage.save_artifact(story_id, "manuscript_markdown", str(output_path), bundle["metadata"])
        return {"artifact": artifact.model_dump(), "preview": bundle["content"][:2000]}

    @app.post("/api/stories/{story_id}/evaluate")
    def evaluate_story(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        with orchestrator_context() as orchestrator:
            report = orchestrator.evaluate_story(story_id)
        story_dir = Path(config.workspace.exports_dir) / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        output_path = story_dir / f"{story_id}_evaluation.json"
        output_path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        artifact = storage.save_artifact(
            story_id,
            "evaluation_json",
            str(output_path),
            {"story_id": story_id, "scene_count": report.scene_count},
        )
        return {"report": report.model_dump(), "artifact": artifact.model_dump()}

    @app.get("/api/stories/{story_id}/artifacts/{artifact_id}/download")
    def download_artifact(story_id: str, artifact_id: int):
        ensure_story(story_id)
        artifact = storage.get_artifact(story_id, artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        path = Path(artifact.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact file missing on disk")
        return FileResponse(str(path), filename=path.name)

    return app


def _pick_teacher_option(options: list[Any], requested_model: str):
    requested = (requested_model or "").lower()
    exact_matches = [option for option in options if option.model.lower() == requested and _looks_like_teacher_model(option.model)]
    if exact_matches:
        return exact_matches[0]

    teachers = [option for option in options if _looks_like_teacher_model(option.model)]
    if not teachers:
        return None

    def score(option) -> tuple[int, int]:
        lowered = option.model.lower()
        preferred = 2 if "e4b" in lowered else 1 if "e2b" in lowered else 0
        requested_match = 1 if requested and requested in lowered else 0
        return (requested_match, preferred)

    return sorted(teachers, key=score, reverse=True)[0]


def _looks_like_teacher_model(model: str) -> bool:
    lowered = (model or "").lower()
    return "gemma" in lowered and ("e2b" in lowered or "e4b" in lowered or "4-e" in lowered)
