"""Stable error codes. Never return HTML."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException
from fastapi.responses import JSONResponse

UNAUTHORIZED = "UNAUTHORIZED"
INVALID_REQUEST = "INVALID_REQUEST"
SOURCE_EXPIRED = "SOURCE_EXPIRED"
SOURCE_FETCH_FAILED = "SOURCE_FETCH_FAILED"
DECODE_FAILED = "DECODE_FAILED"
FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
MODEL_NOT_READY = "MODEL_NOT_READY"
TIMEOUT = "TIMEOUT"
CANCELLED = "CANCELLED"
UPLOAD_FAILED = "UPLOAD_FAILED"
CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
INTERNAL_ERROR = "INTERNAL_ERROR"

STABLE_CODES = frozenset(
    {
        UNAUTHORIZED,
        INVALID_REQUEST,
        SOURCE_EXPIRED,
        SOURCE_FETCH_FAILED,
        DECODE_FAILED,
        FEATURE_UNAVAILABLE,
        MODEL_NOT_READY,
        TIMEOUT,
        CANCELLED,
        UPLOAD_FAILED,
        CHECKSUM_MISMATCH,
        LIMIT_EXCEEDED,
        INTERNAL_ERROR,
    }
)

HTTP_STATUS = {
    UNAUTHORIZED: 401,
    INVALID_REQUEST: 400,
    SOURCE_EXPIRED: 400,
    SOURCE_FETCH_FAILED: 502,
    DECODE_FAILED: 422,
    FEATURE_UNAVAILABLE: 400,
    MODEL_NOT_READY: 503,
    TIMEOUT: 504,
    CANCELLED: 409,
    UPLOAD_FAILED: 502,
    CHECKSUM_MISMATCH: 422,
    LIMIT_EXCEEDED: 413,
    INTERNAL_ERROR: 500,
}


class AnalyzeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        if code not in STABLE_CODES:
            raise ValueError(f"unstable error code: {code}")
        self.code = code
        self.message = message
        super().__init__(message)


def error_body(code: str, message: str) -> dict[str, str]:
    return {"error": message, "code": code}


def error_response(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=HTTP_STATUS[code], content=error_body(code, message))


def raise_http(code: str, message: str) -> NoReturn:
    raise HTTPException(status_code=HTTP_STATUS[code], detail=error_body(code, message))
