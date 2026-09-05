from __future__ import annotations

import logging
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from media_analysis import __version__
from media_analysis.analyze import run_analyze, validate_analyze_request
from media_analysis.config import Settings, get_settings
from media_analysis.errors import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    UNAUTHORIZED,
    AnalyzeError,
    error_response,
)
from media_analysis.jobs import registry
from media_analysis.models_manifest import ManifestState
from media_analysis.request_hash import request_hash
from media_analysis.runtime import RuntimeState, load_runtime
from media_analysis.schemas import AnalyzeRequest
from media_analysis.telemetry import stage_latency_metrics

_manifest: ManifestState | None = None
_runtime: RuntimeState | None = None
logger = logging.getLogger(__name__)


def current_manifest(settings: Settings | None = None) -> ManifestState:
    return current_runtime(settings).manifest


def current_runtime(settings: Settings | None = None) -> RuntimeState:
    global _manifest, _runtime
    cfg = settings or get_settings()
    if _runtime is None:
        _runtime = load_runtime(cfg)
        _manifest = _runtime.manifest
    return _runtime


def reset_manifest() -> None:
    global _manifest, _runtime
    _manifest = None
    _runtime = None


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is not None:
        get_settings.cache_clear()
    cfg = settings or get_settings()
    analysis_gate = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state = current_runtime(cfg)
        if not state.ready:
            problems = "; ".join((*state.manifest.errors, *state.errors))
            raise RuntimeError(f"model runtime failed to initialize: {problems}")
        yield

    app = FastAPI(title="media-analysis", version=__version__, lifespan=lifespan)

    @app.exception_handler(AnalyzeError)
    async def analyze_error_handler(_request: Request, exc: AnalyzeError) -> JSONResponse:
        return error_response(exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(INVALID_REQUEST, "Invalid request")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        state = current_runtime(cfg)
        return {
            "ready": state.ready,
            "productionInferenceReady": state.production_inference_ready,
            "productionBlockers": list(state.production_blockers),
            "referenceMode": state.reference_mode,
            "version": __version__,
            "loadedModels": list(state.loaded_models),
            "warmupComplete": state.warmup_complete,
            "workerRole": cfg.worker_role,
            "executionProviders": list(state.execution_providers),
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        """Rolling per-stage p50/p95 latency signals for worker autoscaling."""
        return {"stageLatencyPercentiles": stage_latency_metrics.snapshot()}

    def _require_key(x_media_analysis_key: str | None) -> None:
        if (
            not cfg.media_analysis_key
            or x_media_analysis_key is None
            or not secrets.compare_digest(x_media_analysis_key, cfg.media_analysis_key)
        ):
            raise AnalyzeError(UNAUTHORIZED, "Missing or invalid X-Media-Analysis-Key")

    @app.post("/analyze")
    def analyze(
        payload: AnalyzeRequest,
        x_media_analysis_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_key(x_media_analysis_key)
        state = current_runtime(cfg)
        validate_analyze_request(payload, cfg)
        digest = request_hash(payload.model_dump())
        try:
            job, cached = registry.begin(payload.idempotencyKey, digest)
        except ValueError as exc:
            raise AnalyzeError(INVALID_REQUEST, str(exc)) from exc
        if cached is not None:
            return cached
        try:
            with analysis_gate:
                result = run_analyze(payload, settings=cfg, runtime=state, job=job)
        except AnalyzeError:
            registry.fail(job)
            raise
        except Exception as exc:
            registry.fail(job)
            logger.exception(
                "analysis request failed",
                extra={"idempotency_key": payload.idempotencyKey},
            )
            raise AnalyzeError(INTERNAL_ERROR, "Internal error") from exc
        registry.complete(job, result)
        return result

    @app.post("/analyze/{idempotency_key}/cancel")
    def cancel(
        idempotency_key: str,
        x_media_analysis_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_key(x_media_analysis_key)
        found = registry.cancel(idempotency_key)
        return {"cancelled": found, "idempotencyKey": idempotency_key}

    return app


app = create_app()
