
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig
from .db import Storage
from .jobs import JobManager
from .llm import build_provider
from .hf_release import infer_base_model_slug, next_release_tag, suggest_repo_id
from .models import (
    BibleContent,
    GeneralistLoopRequest,
    HealthOut,
    HFPullRequest,
    HFPublishRequest,
    OneClickLoopRequest,
    OutlineGenerateRequest,
    RuntimeSettings,
    SceneRequest,
    StoryImportRequest,
    StoryCreate,
    StoryUpdate,
    TrainingRunRequest,
    UIPresetSaveRequest,
)
from .orchestrator import Orchestrator
from .runtime_settings import RuntimeSettingsStore
from .ui_presets import UIPresetStore
from .utils import extract_json_object


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(value: str, *, must_exist: bool = True, allow_directory: bool = True) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Path escapes repository root: {value}") from exc
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {value}")
    if resolved.exists() and not allow_directory and resolved.is_dir():
        raise HTTPException(status_code=400, detail=f"Expected a file path: {value}")
    return resolved


def _maybe_append_flag(command: List[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def build_one_click_command(payload: OneClickLoopRequest) -> List[str]:
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(REPO_ROOT / "scripts" / "one_click_loop.ps1"),
        "-Preset",
        payload.preset,
        "-Mode",
        payload.mode,
        "-TrainAction",
        payload.train_action,
        "-TrainPreset",
        payload.train_preset,
        "-StoryFile",
        str(_resolve_repo_path(payload.story_file, must_exist=True, allow_directory=False)),
        "-SceneFile",
        str(_resolve_repo_path(payload.scene_file, must_exist=True, allow_directory=False)),
    ]
    if payload.story_id:
        command += ["-StoryId", payload.story_id]
    if payload.scene_limit:
        command += ["-SceneLimit", str(payload.scene_limit)]
    _maybe_append_flag(command, payload.run_tests, "-RunTests")
    _maybe_append_flag(command, payload.install_training_deps, "-InstallTrainingDeps")
    return command


def build_generalist_command(payload: GeneralistLoopRequest) -> List[str]:
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(REPO_ROOT / "scripts" / "one_click_generalist.ps1"),
        "-Preset",
        payload.preset,
        "-Mode",
        payload.mode,
        "-TrainAction",
        payload.train_action,
        "-TrainPreset",
        payload.train_preset,
        "-StoryDir",
        str(_resolve_repo_path(payload.story_dir, must_exist=True, allow_directory=True)),
        "-CorpusOutputDir",
        str(_resolve_repo_path(payload.corpus_output_dir, must_exist=False, allow_directory=True)),
        "-ValidationStoryRatio",
        str(payload.validation_story_ratio),
    ]
    if payload.scene_file:
        command += ["-SceneFile", str(_resolve_repo_path(payload.scene_file, must_exist=True, allow_directory=False))]
    if payload.story_offset:
        command += ["-StoryOffset", str(payload.story_offset)]
    if payload.story_limit:
        command += ["-StoryLimit", str(payload.story_limit)]
    if payload.scene_limit:
        command += ["-SceneLimit", str(payload.scene_limit)]
    _maybe_append_flag(command, payload.resume, "-Resume")
    _maybe_append_flag(command, payload.run_tests, "-RunTests")
    _maybe_append_flag(command, payload.install_training_deps, "-InstallTrainingDeps")
    return command


def build_training_command(payload: TrainingRunRequest) -> List[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_qlora.py"),
        "--config",
        str(_resolve_repo_path(payload.config, must_exist=True, allow_directory=False)),
    ]
    if payload.train_file:
        command += ["--train-file", str(_resolve_repo_path(payload.train_file, must_exist=True, allow_directory=False))]
    if payload.eval_file:
        command += ["--eval-file", str(_resolve_repo_path(payload.eval_file, must_exist=True, allow_directory=False))]
    if payload.output_dir:
        command += ["--output-dir", str(_resolve_repo_path(payload.output_dir, must_exist=False, allow_directory=True))]
    if payload.model_name_or_path:
        candidate = Path(payload.model_name_or_path)
        if candidate.exists() or (not candidate.is_absolute() and (REPO_ROOT / candidate).exists()):
            command += ["--model-name-or-path", str(_resolve_repo_path(payload.model_name_or_path, must_exist=True, allow_directory=True))]
        else:
            command += ["--model-name-or-path", payload.model_name_or_path]
    _maybe_append_flag(command, payload.dry_run, "--dry-run")
    _maybe_append_flag(command, payload.print_config, "--print-config")
    return command


def build_hf_publish_command(payload: HFPublishRequest) -> List[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_to_hf.py"),
        "publish",
        "--source-dir",
        str(_resolve_repo_path(payload.source_dir, must_exist=True, allow_directory=True)),
        "--repo-type",
        payload.repo_type,
    ]
    if payload.repo_id:
        command += ["--repo-id", payload.repo_id]
    if payload.namespace:
        command += ["--namespace", payload.namespace]
    if payload.project:
        command += ["--project", payload.project]
    if payload.role:
        command += ["--role", payload.role]
    if payload.base_model:
        command += ["--base-model", payload.base_model]
    if payload.stage:
        command += ["--stage", payload.stage]
    if payload.path_in_repo:
        command += ["--path-in-repo", payload.path_in_repo]
    if payload.revision:
        command += ["--revision", payload.revision]
    if payload.commit_message:
        command += ["--commit-message", payload.commit_message]
    if payload.release_tag:
        command += ["--release-tag", payload.release_tag]
    if payload.release_prefix:
        command += ["--release-prefix", payload.release_prefix]
    if payload.bump:
        command += ["--bump", payload.bump]
    if payload.tag_message:
        command += ["--tag-message", payload.tag_message]
    for pattern in payload.ignore_patterns:
        if pattern:
            command += ["--ignore-pattern", pattern]
    _maybe_append_flag(command, payload.private, "--private")
    _maybe_append_flag(command, payload.exclude_checkpoints, "--exclude-checkpoints")
    _maybe_append_flag(command, payload.auto_tag, "--auto-tag")
    return command


def build_hf_pull_command(payload: HFPullRequest) -> List[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_to_hf.py"),
        "pull",
        "--repo-id",
        payload.repo_id,
        "--repo-type",
        payload.repo_type,
        "--local-dir",
        str(_resolve_repo_path(payload.local_dir, must_exist=False, allow_directory=True)),
    ]
    if payload.revision:
        command += ["--revision", payload.revision]
    for pattern in payload.allow_patterns:
        if pattern:
            command += ["--allow-pattern", pattern]
    for pattern in payload.ignore_patterns:
        if pattern:
            command += ["--ignore-pattern", pattern]
    return command


def run_process_job(command: List[str], emit, *, cwd: Path = REPO_ROOT) -> Dict[str, Any]:
    emit(f"Running command: {' '.join(command)}", 0.05)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: List[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        output_lines.append(line)
        emit(line[:300] or "(blank line)", 0.1)
    returncode = process.wait()
    joined = "\n".join(output_lines).strip()
    parsed_json: Optional[Dict[str, Any] | List[Any]] = None
    if joined:
        try:
            parsed_json = extract_json_object(joined)
        except Exception:
            parsed_json = None
    if returncode != 0:
        raise RuntimeError(joined or f"Command failed with exit code {returncode}")
    return {
        "command": command,
        "returncode": returncode,
        "output_tail": output_lines[-80:],
        "parsed_json": parsed_json,
    }


def search_hf_hub(repo_type: str, search: str = "", author: str = "", limit: int = 12) -> Dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            f"Missing Hugging Face Hub dependency: {type(exc).__name__}: {exc}. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    normalized_limit = max(1, min(int(limit or 12), 50))
    normalized_repo_type = repo_type if repo_type in {"model", "dataset"} else "model"
    kwargs = {
        "search": search or None,
        "author": author or None,
        "limit": normalized_limit,
    }
    iterator = api.list_models(**kwargs) if normalized_repo_type == "model" else api.list_datasets(**kwargs)
    items = []
    for item in iterator:
        repo_id = getattr(item, "id", None) or getattr(item, "modelId", None) or getattr(item, "repoId", None) or ""
        items.append(
            {
                "repo_id": str(repo_id),
                "repo_type": normalized_repo_type,
                "author": str(getattr(item, "author", "") or ""),
                "downloads": int(getattr(item, "downloads", 0) or 0),
                "likes": int(getattr(item, "likes", 0) or 0),
                "private": bool(getattr(item, "private", False)),
                "last_modified": str(getattr(item, "lastModified", "") or ""),
                "sha": str(getattr(item, "sha", "") or ""),
                "tags": list(getattr(item, "tags", []) or []),
            }
        )
    return {
        "items": items,
        "repo_type": normalized_repo_type,
        "search": search,
        "author": author,
        "limit": normalized_limit,
    }


def suggest_hf_release(
    *,
    namespace: str,
    repo_type: str = "model",
    project: str = "conarrative",
    role: str = "",
    base_model: str = "",
    stage: str = "",
    release_prefix: str = "v",
) -> Dict[str, Any]:
    normalized_repo_type = repo_type if repo_type in {"model", "dataset"} else "model"
    repo_id = suggest_repo_id(
        namespace,
        repo_type=normalized_repo_type,
        project=project,
        role=role,
        base_model=base_model,
        stage=stage,
    )
    existing_tags: list[str] = []
    try:
        from huggingface_hub import HfApi

        refs = HfApi().list_repo_refs(repo_id=repo_id, repo_type=None if normalized_repo_type == "model" else normalized_repo_type)
        existing_tags = [str(getattr(tag, "name", "") or "") for tag in getattr(refs, "tags", [])]
    except Exception:
        existing_tags = []
    return {
        "repo_id": repo_id,
        "repo_type": normalized_repo_type,
        "project": project,
        "role": role,
        "base_model": base_model,
        "base_model_slug": infer_base_model_slug(base_model),
        "stage": stage,
        "release_prefix": release_prefix,
        "existing_tags": existing_tags,
        "suggested_tag": next_release_tag(existing_tags, prefix=release_prefix, bump="patch"),
    }


def create_app(config: AppConfig) -> FastAPI:
    storage = Storage(config.workspace.database_path)
    runtime_store = RuntimeSettingsStore(config.workspace.runtime_settings_path, config.backend)
    ui_preset_store = UIPresetStore(Path(config.workspace.root) / "ui_presets.json")
    jobs = JobManager()

    app = FastAPI(title=config.app_name)
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
    app.state.ui_preset_store = ui_preset_store
    app.state.jobs = jobs

    web_dir = Path(__file__).parent / "web"
    static_dir = web_dir / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def current_settings() -> RuntimeSettings:
        return runtime_store.load()

    def make_orchestrator() -> Orchestrator:
        provider = build_provider(current_settings())
        return Orchestrator(storage=storage, provider=provider, config=config)

    def ensure_story(story_id: str):
        story = storage.get_story(story_id)
        if story is None:
            raise HTTPException(status_code=404, detail=f"Story not found: {story_id}")
        return story

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse((web_dir / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        provider = build_provider(current_settings())
        backend_ok, detail = provider.health()
        return HealthOut(
            status="ok" if backend_ok else "degraded",
            provider=current_settings().provider,
            model=current_settings().model,
            database_ok=True,
            backend_ok=backend_ok,
            detail=detail,
        ).model_dump()

    @app.get("/api/runtime-settings")
    def get_runtime_settings() -> Dict[str, Any]:
        return current_settings().model_dump()

    @app.put("/api/runtime-settings")
    def put_runtime_settings(payload: RuntimeSettings) -> Dict[str, Any]:
        return runtime_store.save(payload).model_dump()

    @app.post("/api/runtime-settings/test")
    def test_runtime_settings(payload: Optional[RuntimeSettings] = None) -> Dict[str, Any]:
        settings = payload or current_settings()
        provider = build_provider(settings)
        ok, detail = provider.health()
        return {"ok": ok, "detail": detail, "settings": settings.model_dump()}

    @app.get("/api/stories")
    def list_stories() -> Dict[str, Any]:
        return {"items": [story.model_dump() for story in storage.list_stories()]}

    @app.post("/api/stories")
    def create_story(payload: StoryCreate) -> Dict[str, Any]:
        story = storage.create_story(payload)
        return story.model_dump()

    @app.post("/api/stories/import")
    def import_story(payload: StoryImportRequest) -> Dict[str, Any]:
        raw = yaml.safe_load(payload.yaml_text) or {}
        try:
            story = storage.create_story(StoryCreate(**raw))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid story YAML: {exc}") from exc
        return story.model_dump()

    @app.get("/api/stories/{story_id}")
    def get_story(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        bundle = make_orchestrator().build_memory_bundle(story_id)
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

    @app.get("/api/stories/{story_id}/bible")
    def get_bible(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.get_bible(story_id).model_dump()

    @app.put("/api/stories/{story_id}/bible")
    def put_bible(story_id: str, payload: BibleContent) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.save_bible(story_id, payload).model_dump()

    @app.get("/api/stories/{story_id}/outline")
    def get_outline(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": [card.model_dump() for card in storage.list_outline(story_id)]}

    @app.post("/api/stories/{story_id}/outline/generate")
    def generate_outline(story_id: str, payload: OutlineGenerateRequest) -> Dict[str, Any]:
        ensure_story(story_id)
        cards = make_orchestrator().generate_outline(story_id, payload)
        return {"items": [card.model_dump() for card in cards]}

    @app.get("/api/stories/{story_id}/scenes")
    def list_scenes(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": storage.list_scenes(story_id)}

    @app.get("/api/stories/{story_id}/scenes/{scene_id}")
    def get_scene(story_id: str, scene_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        scene = storage.get_scene(scene_id)
        if scene is None or scene["story_id"] != story_id:
            raise HTTPException(status_code=404, detail="Scene not found")
        return scene

    @app.get("/api/stories/{story_id}/state")
    def get_state(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return storage.get_latest_state(story_id).model_dump()

    @app.get("/api/stories/{story_id}/snapshots")
    def list_snapshots(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": storage.list_state_snapshots(story_id)}

    @app.get("/api/stories/{story_id}/kg")
    def list_kg(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": storage.list_kg_edges(story_id)}

    @app.get("/api/stories/{story_id}/datasets")
    def list_dataset_records(story_id: str, pool_type: Optional[str] = Query(default=None), limit: int = 100) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": storage.list_dataset_records(story_id, pool_type=pool_type, limit=limit), "counts": storage.dataset_counts(story_id)}

    @app.get("/api/stories/{story_id}/artifacts")
    def list_artifacts(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return {"items": [item.model_dump() for item in storage.list_artifacts(story_id)]}

    @app.get("/api/artifacts/download")
    def download_artifact(path: str) -> FileResponse:
        file_path = Path(path).resolve()
        workspace_root = Path(config.workspace.root).resolve()
        try:
            file_path.relative_to(workspace_root)
        except Exception as exc:  # pragma: no cover - safety path
            raise HTTPException(status_code=403, detail="Artifact path is outside workspace") from exc
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(str(file_path), filename=file_path.name)

    @app.post("/api/stories/{story_id}/jobs/run-scene")
    def submit_scene_job(story_id: str, payload: SceneRequest) -> Dict[str, Any]:
        ensure_story(story_id)

        def runner(emit):
            result = make_orchestrator().run_scene(story_id, payload, log=emit)
            return result.model_dump(mode="json")

        job = jobs.submit("run_scene", story_id, runner)
        return job.model_dump()

    @app.post("/api/stories/{story_id}/jobs/auto-novel")
    def submit_auto_novel(story_id: str, scene_limit: Optional[int] = None) -> Dict[str, Any]:
        ensure_story(story_id)

        def runner(emit):
            return make_orchestrator().auto_write_novel(story_id, scene_limit=scene_limit, log=emit)

        job = jobs.submit("auto_novel", story_id, runner)
        return job.model_dump()

    @app.post("/api/stories/{story_id}/export")
    def export_story(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return make_orchestrator().write_export_files(story_id)

    @app.post("/api/stories/{story_id}/evaluate")
    def evaluate_story(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return make_orchestrator().write_evaluation_file(story_id)

    @app.post("/api/stories/{story_id}/export-datasets")
    def export_datasets(story_id: str) -> Dict[str, Any]:
        ensure_story(story_id)
        return make_orchestrator().write_training_bundle(story_id)

    @app.get("/api/jobs")
    def list_jobs(story_id: Optional[str] = None) -> Dict[str, Any]:
        return {"items": [job.model_dump() for job in jobs.list(story_id)]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.model_dump()

    @app.get("/api/ui-presets")
    def list_ui_presets() -> Dict[str, Any]:
        return {"items": ui_preset_store.list_all()}

    @app.get("/api/hf/repos")
    def browse_hf_repos(
        repo_type: str = Query(default="model"),
        search: str = Query(default=""),
        author: str = Query(default=""),
        limit: int = Query(default=12),
    ) -> Dict[str, Any]:
        try:
            return search_hf_hub(repo_type=repo_type, search=search, author=author, limit=limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/hf/suggest-release")
    def hf_suggest_release(
        namespace: str = Query(...),
        repo_type: str = Query(default="model"),
        project: str = Query(default="conarrative"),
        role: str = Query(default=""),
        base_model: str = Query(default=""),
        stage: str = Query(default=""),
        release_prefix: str = Query(default="v"),
    ) -> Dict[str, Any]:
        return suggest_hf_release(
            namespace=namespace,
            repo_type=repo_type,
            project=project,
            role=role,
            base_model=base_model,
            stage=stage,
            release_prefix=release_prefix,
        )

    @app.post("/api/ui-presets")
    def save_ui_preset(payload: UIPresetSaveRequest) -> Dict[str, Any]:
        try:
            record = ui_preset_store.save(payload.kind, payload.name, payload.payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @app.delete("/api/ui-presets/{kind}/{name}")
    def delete_ui_preset(kind: str, name: str) -> Dict[str, Any]:
        deleted = ui_preset_store.delete(kind, name)
        if not deleted:
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"ok": True, "kind": kind, "name": name}

    @app.post("/api/system/jobs/one-click")
    def submit_one_click_job(payload: OneClickLoopRequest) -> Dict[str, Any]:
        command = build_one_click_command(payload)

        def runner(emit):
            return run_process_job(command, emit)

        job = jobs.submit("one_click_loop", None, runner)
        return job.model_dump()

    @app.post("/api/system/jobs/generalist")
    def submit_generalist_job(payload: GeneralistLoopRequest) -> Dict[str, Any]:
        command = build_generalist_command(payload)

        def runner(emit):
            return run_process_job(command, emit)

        job = jobs.submit("generalist_loop", None, runner)
        return job.model_dump()

    @app.post("/api/system/jobs/train")
    def submit_training_job(payload: TrainingRunRequest) -> Dict[str, Any]:
        command = build_training_command(payload)

        def runner(emit):
            return run_process_job(command, emit)

        job = jobs.submit("training_run", None, runner)
        return job.model_dump()

    @app.post("/api/system/jobs/hf-publish")
    def submit_hf_publish_job(payload: HFPublishRequest) -> Dict[str, Any]:
        command = build_hf_publish_command(payload)

        def runner(emit):
            return run_process_job(command, emit)

        job = jobs.submit("hf_publish", None, runner)
        return job.model_dump()

    @app.post("/api/system/jobs/hf-pull")
    def submit_hf_pull_job(payload: HFPullRequest) -> Dict[str, Any]:
        command = build_hf_pull_command(payload)

        def runner(emit):
            return run_process_job(command, emit)

        job = jobs.submit("hf_pull", None, runner)
        return job.model_dump()

    return app
