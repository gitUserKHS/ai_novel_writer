
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .models import JobRecord, JobStatus, utcnow_iso


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="conarrative-job")
        self.lock = threading.Lock()
        self.jobs: Dict[str, JobRecord] = {}

    def _append_message(self, job_id: str, message: str, progress: float) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.messages.append({"time": utcnow_iso(), "message": message, "progress": progress})
            job.progress = max(job.progress, progress)
            job.updated_at = utcnow_iso()

    def submit(self, job_type: str, story_id: Optional[str], runner: Callable[[Callable[[str, float], None]], Dict[str, Any]]) -> JobRecord:
        job = JobRecord(id=uuid.uuid4().hex, job_type=job_type, story_id=story_id)
        with self.lock:
            self.jobs[job.id] = job

        def emit(message: str, progress: float) -> None:
            self._append_message(job.id, message, progress)

        def task() -> None:
            with self.lock:
                current = self.jobs[job.id]
                current.status = JobStatus.RUNNING
                current.updated_at = utcnow_iso()
            try:
                result = runner(emit)
                with self.lock:
                    current = self.jobs[job.id]
                    current.status = JobStatus.SUCCEEDED
                    current.progress = 1.0
                    current.result = result
                    current.updated_at = utcnow_iso()
            except Exception as exc:  # pragma: no cover - exercised via API integration if failures happen
                with self.lock:
                    current = self.jobs[job.id]
                    current.status = JobStatus.FAILED
                    current.error = f"{type(exc).__name__}: {exc}"
                    current.updated_at = utcnow_iso()
                    current.messages.append({"time": utcnow_iso(), "message": current.error, "progress": current.progress})

        self.executor.submit(task)
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def list(self, story_id: Optional[str] = None) -> List[JobRecord]:
        with self.lock:
            items = list(self.jobs.values())
        if story_id:
            items = [job for job in items if job.story_id == story_id]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return [item.model_copy(deep=True) for item in items]
