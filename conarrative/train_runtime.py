from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import httpx

from .config import AppConfig
from .db import Storage
from .models import ProviderType, RuntimeSettings, utcnow_iso
from .training import export_training_corpus

LogFn = Callable[[str, float], None]

DEFAULT_TRAINING_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TEACHER_MODEL = "google/gemma-4-E2B-it"


def training_root(config: AppConfig) -> Path:
    root = Path(config.workspace.root) / "training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def training_dataset_dir(config: AppConfig, story_id: str) -> Path:
    path = training_root(config) / "datasets" / story_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_runs_dir(config: AppConfig, story_id: str) -> Path:
    path = training_root(config) / "runs" / story_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_venv_dir(config: AppConfig) -> Path:
    return training_root(config) / ".venv-py312"


def training_python_path(config: AppConfig) -> Path:
    venv = training_venv_dir(config)
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def inspect_training_environment(config: AppConfig) -> Dict[str, Any]:
    selected_python = find_supported_training_python()
    env_python = training_python_path(config)
    gpu = detect_gpu()
    venv_exists = env_python.exists()
    env_info = {
        "python_version": "",
        "torch_installed": False,
        "torch_version": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_total_gib": 0.0,
        "cuda_free_gib": 0.0,
        "bitsandbytes_ok": False,
        "transformers_ok": False,
        "peft_ok": False,
        "detail": "",
    }
    if venv_exists:
        try:
            env_info = inspect_training_python(env_python)
        except Exception as exc:
            env_info["detail"] = str(exc)
    ready = bool(
        venv_exists
        and env_info["torch_installed"]
        and env_info["transformers_ok"]
        and env_info["peft_ok"]
        and (env_info["cuda_available"] if gpu["available"] else True)
    )
    detail_parts = []
    if selected_python["path"]:
        detail_parts.append(f"Preferred training Python: {selected_python['version']} @ {selected_python['path']}")
    else:
        detail_parts.append("No supported Python 3.12/3.10 installation was found.")
    if gpu["available"]:
        detail_parts.append(f"GPU detected: {gpu['name']}")
    else:
        detail_parts.append("No NVIDIA GPU was detected via nvidia-smi.")
    if venv_exists:
        detail_parts.append(f"Training env: {env_python}")
        if env_info["cuda_total_gib"]:
            detail_parts.append(
                f"CUDA memory: {env_info['cuda_free_gib']:.1f} GiB free / {env_info['cuda_total_gib']:.1f} GiB total."
            )
        if env_info["detail"]:
            detail_parts.append(env_info["detail"])
    else:
        detail_parts.append("Training env has not been created yet.")
    profile_name = training_profile_name(env_info)
    if not venv_exists and gpu["available"]:
        profile_name = "pending-cuda"
    return {
        "preferred_python_version": selected_python["version"],
        "preferred_python_path": selected_python["path"],
        "training_env_dir": str(training_venv_dir(config)),
        "training_python_path": str(env_python),
        "training_env_exists": venv_exists,
        "gpu_available": gpu["available"],
        "gpu_name": gpu["name"],
        "python_version": env_info["python_version"],
        "torch_installed": env_info["torch_installed"],
        "torch_version": env_info["torch_version"],
        "cuda_available": env_info["cuda_available"],
        "cuda_device_count": env_info["cuda_device_count"],
        "cuda_total_gib": env_info["cuda_total_gib"],
        "cuda_free_gib": env_info["cuda_free_gib"],
        "bitsandbytes_ok": env_info["bitsandbytes_ok"],
        "transformers_ok": env_info["transformers_ok"],
        "peft_ok": env_info["peft_ok"],
        "ready": ready,
        "training_profile": profile_name,
        "detail": " ".join(detail_parts),
    }


def ensure_training_environment(config: AppConfig, log: Optional[LogFn] = None, force_reinstall: bool = False) -> Dict[str, Any]:
    def emit(message: str, progress: float) -> None:
        if log is not None:
            log(message, progress)

    status = inspect_training_environment(config)
    if status["ready"] and not force_reinstall:
        emit("Training environment is already ready", 1.0)
        return status

    selection = find_supported_training_python()
    if not selection["path"] or not selection["launcher"]:
        raise RuntimeError(
            "Python 3.12 or 3.10 was not found. Install one of them to prepare the training environment."
        )

    venv_dir = training_venv_dir(config)
    env_python = training_python_path(config)
    if force_reinstall and venv_dir.exists():
        emit("Removing existing training environment", 0.05)
        import shutil

        shutil.rmtree(venv_dir, ignore_errors=True)

    if not env_python.exists():
        emit(f"Creating training environment with Python {selection['version']}", 0.1)
        create_command = (
            [selection["launcher"], f"-{selection['version']}", "-m", "venv", str(venv_dir)]
            if selection["launcher"] == "py"
            else [selection["launcher"], "-m", "venv", str(venv_dir)]
        )
        run_command_live(
            create_command,
            cwd=Path.cwd(),
            emit=emit,
            progress=(0.1, 0.2),
        )

    emit("Upgrading pip tooling", 0.25)
    run_command_live(
        [str(env_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=Path.cwd(),
        emit=emit,
        progress=(0.25, 0.35),
    )

    gpu = detect_gpu()
    torch_index = "https://download.pytorch.org/whl/cu124" if gpu["available"] else "https://download.pytorch.org/whl/cpu"
    emit("Installing PyTorch for the training environment", 0.4)
    run_command_live(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            torch_index,
        ],
        cwd=Path.cwd(),
        emit=emit,
        progress=(0.4, 0.6),
    )

    emit("Installing training dependencies", 0.7)
    run_command_live(
        [str(env_python), "-m", "pip", "install", "--upgrade", "-r", "requirements-train.txt"],
        cwd=Path.cwd(),
        emit=emit,
        progress=(0.7, 0.9),
    )

    final_status = inspect_training_environment(config)
    if not final_status["torch_installed"]:
        raise RuntimeError("Training environment setup completed, but torch is still unavailable.")
    if gpu["available"] and not final_status["cuda_available"]:
        raise RuntimeError(
            "Training environment was created but CUDA is still unavailable in the training Python. "
            "Check the installed PyTorch build or GPU driver state."
        )
    emit("Training environment is ready", 1.0)
    return final_status


def export_story_training_dataset(config: AppConfig, storage: Storage, story_id: str) -> Dict[str, Any]:
    records = storage.list_dataset_records(story_id=story_id)
    if not records:
        raise RuntimeError("No dataset records are available for this story yet. Generate at least one scene first.")
    out_dir = training_dataset_dir(config, story_id)
    manifest = export_training_corpus(records, out_dir)
    manifest_path = Path(manifest["manifest"])
    storage.save_artifact(
        story_id,
        "training_dataset_manifest",
        str(manifest_path),
        {"counts": manifest["counts"], "dataset_dir": str(out_dir)},
    )
    return manifest


def training_profile_name(env_status: Dict[str, Any]) -> str:
    if not env_status.get("cuda_available"):
        return "cpu-debug"
    total_gib = float(env_status.get("cuda_total_gib") or 0.0)
    if total_gib and total_gib <= 8.5:
        return "low-vram-8gb"
    return "balanced-cuda"


def training_profile_args(env_status: Dict[str, Any], request: Dict[str, Any]) -> list[str]:
    profile = training_profile_name(env_status)
    max_seq_length = int(request.get("max_seq_length") or (2048 if profile == "low-vram-8gb" else 4096))
    lora_r = int(request.get("lora_r") or (8 if profile == "low-vram-8gb" else 16))
    lora_alpha = int(request.get("lora_alpha") or (16 if profile == "low-vram-8gb" else 32))
    script_profile = "low-vram" if profile == "low-vram-8gb" else "auto"
    return [
        "--profile",
        script_profile,
        "--max-seq-length",
        str(max_seq_length),
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
    ]


def distill_prompt_only_dataset(
    *,
    input_file: str | Path,
    output_file: str | Path,
    base_url: str,
    model: str,
    api_key: str,
    log: Optional[LogFn] = None,
    timeout_seconds: int = 600,
    resume: bool = True,
) -> Dict[str, Any]:
    def emit(message: str, progress: float) -> None:
        if log is not None:
            log(message, progress)

    source_rows = load_jsonl(input_file)
    if not source_rows:
        raise RuntimeError("No prompt-only rows were found for distillation.")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(load_jsonl(target)) if resume and target.exists() else set()
    mode = "a" if done else "w"
    completed = 0
    with httpx.Client(timeout=timeout_seconds) as client, target.open(mode, encoding="utf-8") as handle:
        total = len(source_rows)
        for index, row in enumerate(source_rows, start=1):
            key = row_key(row)
            if key in done:
                completed += 1
                continue
            teacher_text = call_teacher_model(
                client=client,
                base_url=base_url,
                model=model,
                api_key=api_key,
                messages=row.get("messages") or [],
            )
            distilled = {
                "messages": [*(row.get("messages") or []), {"role": "assistant", "content": teacher_text}],
                "metadata": {
                    **(row.get("metadata") or {}),
                    "teacher_model": model,
                    "teacher_base_url": base_url,
                    "source": "distilled_prompt_only",
                },
            }
            handle.write(json.dumps(distilled, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            emit(f"Distilled {completed}/{total} prompt-only rows", 0.2 + (0.8 * completed / max(total, 1)))
    return {"output_file": str(target), "count": completed}


def distill_teacher_coaching_dataset(
    *,
    input_file: str | Path,
    output_file: str | Path,
    base_url: str,
    model: str,
    api_key: str,
    variants_per_prompt: int = 1,
    log: Optional[LogFn] = None,
    timeout_seconds: int = 900,
    resume: bool = True,
) -> Dict[str, Any]:
    def emit(message: str, progress: float) -> None:
        if log is not None:
            log(message, progress)

    source_rows = load_jsonl(input_file)
    if not source_rows:
        raise RuntimeError("No prompt-only rows were found for teacher coaching.")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(load_jsonl(target)) if resume and target.exists() else set()
    mode = "a" if done else "w"
    completed = 0
    variant_count = max(0, int(variants_per_prompt))
    total = max(len(source_rows) * max(variant_count, 1), 1)
    with httpx.Client(timeout=timeout_seconds) as client, target.open(mode, encoding="utf-8") as handle:
        for row_index, row in enumerate(source_rows, start=1):
            for variant_index in range(1, variant_count + 1):
                key = f"{row_key(row)}::coach::{variant_index}"
                if key in done:
                    completed += 1
                    continue
                teacher_text = call_teacher_coaching_model(
                    client=client,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    messages=row.get("messages") or [],
                    variant_index=variant_index,
                )
                coached = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "당신은 한국어 장편소설 작가 겸 비평 코치다. "
                                "좋은 예시 장면과 그 평가를 JSON으로 작성한다."
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_teacher_coaching_user_prompt(row.get("messages") or [], variant_index),
                        },
                        {"role": "assistant", "content": teacher_text},
                    ],
                    "metadata": {
                        **(row.get("metadata") or {}),
                        "teacher_model": model,
                        "teacher_base_url": base_url,
                        "source": "teacher_coached_example",
                        "variant_index": variant_index,
                        "row_index": row_index,
                    },
                }
                handle.write(json.dumps(coached, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                emit(f"Teacher coached {completed}/{total} examples", 0.2 + (0.8 * completed / total))
    return {"output_file": str(target), "count": completed}


def run_one_click_training(
    *,
    config: AppConfig,
    storage: Storage,
    story_id: str,
    runtime_settings: RuntimeSettings,
    request: Dict[str, Any],
    log: Optional[LogFn] = None,
) -> Dict[str, Any]:
    def emit(message: str, progress: float) -> None:
        if log is not None:
            log(message, progress)

    emit("Preparing training environment", 0.02)
    env_status = ensure_training_environment(config, log=log)

    emit("Exporting training dataset", 0.15)
    manifest = export_story_training_dataset(config, storage, story_id)
    dataset_dir = training_dataset_dir(config, story_id)

    train_files = [str(dataset_dir / "accepted_sft.jsonl")]
    multi_target_file = dataset_dir / "multi_target_sft.jsonl"
    if multi_target_file.exists() and multi_target_file.stat().st_size > 0:
        # Narrative-MTP: supervise prose plus predicted future/state targets.
        train_files.append(str(multi_target_file))
    distilled_output = dataset_dir / "distilled_sft.jsonl"
    teacher_coached_output = dataset_dir / "teacher_coached_sft.jsonl"
    distillation_used = False
    distillation_detail = "Distillation skipped."
    teacher_coaching_used = False
    teacher_coaching_detail = "Teacher coaching skipped."
    teacher_base_url = (request.get("teacher_base_url") or "").strip()
    teacher_model = (request.get("teacher_model") or DEFAULT_TEACHER_MODEL).strip()
    teacher_api_key = request.get("teacher_api_key") or ""
    use_distillation = bool(request.get("use_distillation", True))
    use_teacher_coaching = bool(request.get("teacher_coaching", True))
    teacher_variants = int(request.get("teacher_variants_per_prompt") or 1)

    prompt_only_manifest = dataset_dir / "prompt_only_teacher.jsonl"
    if (
        use_distillation
        and teacher_base_url
        and teacher_model
        and prompt_only_manifest.exists()
        and prompt_only_manifest.stat().st_size > 0
    ):
        emit(f"Distilling prompt-only dataset with teacher {teacher_model}", 0.25)
        distill_result = distill_prompt_only_dataset(
            input_file=prompt_only_manifest,
            output_file=distilled_output,
            base_url=teacher_base_url,
            model=teacher_model,
            api_key=teacher_api_key,
            log=lambda message, progress: emit(message, 0.25 + progress * 0.25),
            resume=True,
        )
        if distilled_output.exists() and distilled_output.stat().st_size > 0:
            train_files.insert(0, str(distilled_output))
            distillation_used = True
            distillation_detail = f"Used teacher {teacher_model} via {teacher_base_url}."
            storage.save_artifact(
                story_id,
                "distilled_training_dataset",
                str(distilled_output),
                {"teacher_model": teacher_model, "teacher_base_url": teacher_base_url, "count": distill_result["count"]},
            )
    else:
        if not prompt_only_manifest.exists() or prompt_only_manifest.stat().st_size == 0:
            distillation_detail = "Distillation skipped because there were no prompt-only rows to distill."
        else:
            distillation_detail = "Distillation skipped because teacher_base_url or teacher_model was not set."

    if (
        use_teacher_coaching
        and teacher_variants > 0
        and teacher_base_url
        and teacher_model
        and prompt_only_manifest.exists()
        and prompt_only_manifest.stat().st_size > 0
    ):
        emit(f"Generating teacher coached examples with {teacher_model}", 0.42)
        coaching_result = distill_teacher_coaching_dataset(
            input_file=prompt_only_manifest,
            output_file=teacher_coached_output,
            base_url=teacher_base_url,
            model=teacher_model,
            api_key=teacher_api_key,
            variants_per_prompt=teacher_variants,
            log=lambda message, progress: emit(message, 0.42 + progress * 0.1),
            resume=True,
        )
        if teacher_coached_output.exists() and teacher_coached_output.stat().st_size > 0:
            train_files.append(str(teacher_coached_output))
            teacher_coaching_used = True
            teacher_coaching_detail = f"Used teacher coaching from {teacher_model} via {teacher_base_url}."
            storage.save_artifact(
                story_id,
                "teacher_coached_training_dataset",
                str(teacher_coached_output),
                {"teacher_model": teacher_model, "teacher_base_url": teacher_base_url, "count": coaching_result["count"]},
            )
    elif use_teacher_coaching:
        if not prompt_only_manifest.exists() or prompt_only_manifest.stat().st_size == 0:
            teacher_coaching_detail = "Teacher coaching skipped because there were no prompt-only rows."
        else:
            teacher_coaching_detail = "Teacher coaching skipped because teacher_base_url or teacher_model was not set."

    run_id = utcnow_iso().replace(":", "-")
    output_dir = training_runs_dir(config, story_id) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    base_model = request.get("base_model") or DEFAULT_TRAINING_MODEL
    hf_token = request.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    profile_name = training_profile_name(env_status)
    profile_args = training_profile_args(env_status, request)
    if profile_name == "low-vram-8gb":
        emit("Using low-VRAM training profile: max_seq_length=2048, LoRA r=8, alpha=16", 0.5)
    if env_status.get("cuda_free_gib") and float(env_status["cuda_free_gib"]) < 4.0:
        emit("GPU free memory is low. Close LM Studio/Ollama/other GPU apps before training if loading fails.", 0.51)
    if base_model.startswith("google/gemma") and not hf_token:
        emit("No Hugging Face token was provided. Gemma downloads can fail or be rate-limited without one.", 0.52)
    command = [
        str(training_python_path(config)),
        "scripts/train_qlora.py",
        "--train-file",
        *train_files,
        "--model-name",
        base_model,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(request.get("epochs", 1.0)),
        "--per-device-batch-size",
        str(request.get("per_device_batch_size", 1)),
        "--gradient-accumulation-steps",
        str(request.get("gradient_accumulation_steps", 16)),
        *profile_args,
    ]
    if not env_status.get("bitsandbytes_ok") or not env_status.get("cuda_available"):
        command.append("--no-4bit")

    emit("Starting LoRA/QLoRA training", 0.55)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    run_command_live(command, cwd=Path.cwd(), env=env, emit=emit, progress=(0.55, 0.98))

    metadata = {
        "base_model": base_model,
        "train_files": train_files,
        "distillation_used": distillation_used,
        "distillation_detail": distillation_detail,
        "teacher_coaching_used": teacher_coaching_used,
        "teacher_coaching_detail": teacher_coaching_detail,
        "teacher_model": teacher_model,
        "teacher_base_url": teacher_base_url,
        "training_profile": profile_name,
        "training_args": profile_args,
        "output_dir": str(output_dir),
        "final_adapter_dir": str(output_dir / "final_adapter"),
        "env_status": env_status,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    storage.save_artifact(story_id, "training_run_metadata", str(metadata_path), metadata)
    emit("Training completed", 1.0)
    return {
        "dataset_manifest": manifest,
        "training_metadata": metadata,
    }


def detect_gpu() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        return {"available": False, "name": ""}
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"available": bool(names), "name": names[0] if names else ""}


def find_supported_training_python() -> Dict[str, str]:
    if os.name == "nt":
        for version in ("3.12", "3.10"):
            try:
                result = subprocess.run(
                    ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
            except Exception:
                continue
            executable = result.stdout.strip()
            if executable:
                return {"version": version, "path": executable, "launcher": "py"}
        return {"version": "", "path": "", "launcher": ""}

    for candidate in ("python3.12", "python3.10"):
        path = shutil_which(candidate)
        if path:
            return {"version": candidate.replace("python", ""), "path": path, "launcher": path}
    return {"version": "", "path": "", "launcher": ""}


def inspect_training_python(python_path: Path) -> Dict[str, Any]:
    code = """
import importlib.util
import json
import platform

payload = {
    "python_version": platform.python_version(),
    "torch_installed": False,
    "torch_version": "",
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_total_gib": 0.0,
    "cuda_free_gib": 0.0,
    "bitsandbytes_ok": importlib.util.find_spec("bitsandbytes") is not None,
    "transformers_ok": importlib.util.find_spec("transformers") is not None,
    "peft_ok": importlib.util.find_spec("peft") is not None,
    "detail": "",
}
try:
    import torch
    payload["torch_installed"] = True
    payload["torch_version"] = torch.__version__
    payload["cuda_available"] = torch.cuda.is_available()
    payload["cuda_device_count"] = torch.cuda.device_count()
    if payload["cuda_available"]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        payload["cuda_total_gib"] = round(total_bytes / (1024 ** 3), 2)
        payload["cuda_free_gib"] = round(free_bytes / (1024 ** 3), 2)
except Exception as exc:
    payload["detail"] = str(exc)
print(json.dumps(payload))
"""
    result = subprocess.run(
        [str(python_path), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout.strip())


def run_command_live(
    command: list[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    emit: Optional[LogFn] = None,
    progress: tuple[float, float] = (0.0, 1.0),
) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    start, end = progress
    lines_seen = 0
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if emit is not None and line:
            lines_seen += 1
            scaled = min(end, start + min(0.95, lines_seen / 200.0) * max(end - start, 0.0))
            emit(line, scaled)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_keys(rows: Iterable[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        key = row_key(row)
        metadata = row.get("metadata") or {}
        if metadata.get("source") == "teacher_coached_example":
            key = f"{key}::coach::{metadata.get('variant_index', 1)}"
        keys.add(key)
    return keys


def row_key(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return f"{metadata.get('story_id', '')}::{metadata.get('scene_index', '')}"


def call_teacher_model(
    *,
    client: httpx.Client,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[Dict[str, str]],
) -> str:
    if not messages:
        raise RuntimeError("Distillation row is missing messages.")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 한국어 장면 소설을 쓰는 상위 교사 모델이다. "
                    "자연스럽고 개연성 있는 한국어 장면 본문만 작성하라. "
                    "영어 설명, 제목, 메타 발화 없이 결과만 출력하라."
                ),
            },
            *messages,
        ],
        "temperature": 0.7,
        "max_tokens": 2600,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected teacher response format: {json.dumps(data)[:400]}") from exc


def call_teacher_coaching_model(
    *,
    client: httpx.Client,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[Dict[str, str]],
    variant_index: int,
) -> str:
    if not messages:
        raise RuntimeError("Teacher coaching row is missing messages.")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 한국어 장편소설을 지도하는 상위 교사 모델이다. "
                    "학생 모델이 과적합하지 않도록 같은 요청에서 다른 좋은 예시를 만들고, "
                    "왜 좋은지 짧게 평가한다. 반드시 JSON만 출력한다."
                ),
            },
            {
                "role": "user",
                "content": build_teacher_coaching_user_prompt(messages, variant_index),
            },
        ],
        "temperature": 0.85,
        "max_tokens": 3200,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected teacher coaching response format: {json.dumps(data)[:400]}") from exc


def build_teacher_coaching_user_prompt(messages: list[Dict[str, str]], variant_index: int) -> str:
    source_prompt = "\n\n".join(
        f"[{message.get('role', 'user')}]\n{message.get('content', '')}".strip()
        for message in messages
        if message.get("content")
    )
    return (
        "아래 장면 생성 요청을 보고 학습용 좋은 예시를 하나 만드세요.\n"
        f"변형 번호: {variant_index}\n\n"
        "출력 JSON 스키마:\n"
        "{\n"
        '  "draft": "좋은 예시 장면 본문",\n'
        '  "why_it_works": ["개연성", "감정선", "복선/회수", "문체 관점의 장점"],\n'
        '  "rubric_scores": {"coherence": 0.0, "character": 0.0, "payoff": 0.0, "language": 0.0},\n'
        '  "avoid_overfitting_notes": ["학생 모델이 외우지 말고 일반화해야 할 원칙"]\n'
        "}\n\n"
        "조건:\n"
        "- draft는 자연스러운 한국어 소설 장면이어야 합니다.\n"
        "- 기존 accepted 장면을 그대로 반복하지 말고 다른 전개/표현을 사용하세요.\n"
        "- 평가와 원칙은 짧고 구체적으로 작성하세요.\n"
        "- JSON 외의 설명은 출력하지 마세요.\n\n"
        f"원본 요청:\n{source_prompt}"
    )


def shutil_which(name: str) -> str:
    from shutil import which

    return which(name) or ""
