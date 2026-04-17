from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .db import Storage
from .models import JobStatus


@dataclass
class JobTask:
    job_id: str
    story_id: str
    kind: str
    fn: Callable[..., Dict[str, Any]]
    kwargs: Dict[str, Any]


class JobManager:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.storage.recover_incomplete_jobs()
        self.queue: "queue.Queue[Optional[JobTask]]" = queue.Queue()
        self._shutdown = False
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def enqueue(self, job_id: str, story_id: str, kind: str, fn: Callable[..., Dict[str, Any]], **kwargs) -> None:
        if self._shutdown:
            raise RuntimeError("Job manager is shut down")
        self.storage.create_job(job_id, story_id, kind, message="Queued")
        self.queue.put(JobTask(job_id=job_id, story_id=story_id, kind=kind, fn=fn, kwargs=kwargs))

    def _worker_loop(self) -> None:
        while True:
            task = self.queue.get()
            if task is None:
                return
            try:
                self.storage.append_job_log(task.job_id, "Job started", progress=0.01, status=JobStatus.RUNNING)

                def emit(message: str, progress: float) -> None:
                    self.storage.append_job_log(task.job_id, message, progress=progress, status=JobStatus.RUNNING)

                result = task.fn(log=emit, **task.kwargs)
                self.storage.finish_job(task.job_id, result=result, message="Completed")
                self.storage.append_job_log(task.job_id, "Job completed", progress=1.0, status=JobStatus.SUCCEEDED)
            except Exception as exc:
                tb = traceback.format_exc()
                self.storage.append_job_log(task.job_id, f"Error: {exc}", progress=1.0, status=JobStatus.FAILED)
                self.storage.fail_job(task.job_id, error_text=tb)
            finally:
                self.queue.task_done()

    def shutdown(self, wait: bool = False, timeout: float = 1.0) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.queue.put(None)
        if wait and threading.current_thread() is not self.worker:
            self.worker.join(timeout=timeout)
