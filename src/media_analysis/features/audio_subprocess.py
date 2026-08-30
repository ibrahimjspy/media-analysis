"""Bounded subprocess runner with cancellation and timeout for audio ffmpeg paths."""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable

from media_analysis.errors import TIMEOUT, AnalyzeError

_POLL_INTERVAL_SEC = 0.05


def run_bounded_subprocess(
    args: list[str],
    *,
    timeout_sec: float | None = None,
    cancel_check: Callable[[], None] | None = None,
    text_mode: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run a subprocess with cancel/timeout while draining pipes to avoid deadlocks."""
    if cancel_check:
        cancel_check()

    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text_mode,
    )

    comm_result: dict[str, tuple] = {}
    comm_error: dict[str, BaseException] = {}

    def _reader() -> None:
        try:
            comm_result["data"] = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            comm_error["timeout"] = exc
            proc.kill()
            proc.communicate()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    try:
        while reader.is_alive():
            if cancel_check:
                cancel_check()
            if deadline is not None and time.monotonic() >= deadline:
                proc.kill()
                reader.join(timeout=1.0)
                raise AnalyzeError(TIMEOUT, "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC")
            reader.join(timeout=_POLL_INTERVAL_SEC)

        if "timeout" in comm_error:
            raise AnalyzeError(
                TIMEOUT,
                "Analysis exceeded MEDIA_ANALYSIS_JOB_TIMEOUT_SEC",
            ) from comm_error["timeout"]

        if cancel_check:
            cancel_check()

        stdout, stderr = comm_result["data"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
        )
    except AnalyzeError:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        raise
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        raise
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        reader.join(timeout=0.1)
