from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    idempotency_key: str
    request_hash: str
    cancel: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    active: bool = True

    def cancelled(self) -> bool:
        return self.cancel.is_set()


class JobRegistry:
    def __init__(self, *, max_jobs: int = 1000) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._max_jobs = max_jobs

    def begin(self, idempotency_key: str, request_hash: str) -> tuple[Job, dict[str, Any] | None]:
        with self._lock:
            existing = self._jobs.get(idempotency_key)
            if existing:
                if existing.request_hash != request_hash:
                    raise ValueError("idempotency key is bound to a different request")
                if existing.result is not None:
                    return existing, existing.result
                if existing.active:
                    raise ValueError("job already active")
            job = Job(idempotency_key=idempotency_key, request_hash=request_hash)
            self._jobs[idempotency_key] = job
            self._prune_locked()
            return job, None

    def complete(self, job: Job, result: dict[str, Any]) -> None:
        with self._lock:
            job.result = result
            job.active = False
            self._prune_locked()

    def fail(self, job: Job) -> None:
        with self._lock:
            job.active = False

    def cancel(self, idempotency_key: str) -> bool:
        with self._lock:
            job = self._jobs.get(idempotency_key)
            if job is None or not job.active:
                return False
            job.cancel.set()
            return True

    def get(self, idempotency_key: str) -> Job | None:
        with self._lock:
            return self._jobs.get(idempotency_key)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _prune_locked(self) -> None:
        overflow = len(self._jobs) - self._max_jobs
        if overflow <= 0:
            return
        removable = [key for key, job in self._jobs.items() if not job.active]
        for key in removable[:overflow]:
            del self._jobs[key]


registry = JobRegistry()
