"""Bounded content-addressed cache for source media, canonical proxies, and results."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheEntry:
    path: Path
    byte_count: int


class LocalMediaCache:
    """Bounded local cache with atomic publication and least-recently-used pruning."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        ttl_sec: float,
        enabled: bool = True,
    ) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.ttl_sec = ttl_sec
        self.enabled = enabled
        self._lock = threading.RLock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def stable_key(*parts: str) -> str:
        material = "\0".join(parts).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _path(self, namespace: str, key: str, suffix: str) -> Path:
        safe_namespace = "".join(ch for ch in namespace if ch.isalnum() or ch in "-_")
        safe_key = "".join(ch for ch in key.lower() if ch in "0123456789abcdef")
        if not safe_namespace or len(safe_key) != 64:
            raise ValueError("cache namespace/key is invalid")
        return self.root / safe_namespace / f"{safe_key}{suffix}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # All cache instances/processes sharing this directory coordinate writes,
        # eviction and checkout. Job copies survive deletion of the cache name.
        with self._lock, (self.root / ".lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def get(
        self,
        namespace: str,
        key: str,
        *,
        suffix: str,
        dest: Path | None = None,
    ) -> CacheEntry | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key, suffix)
        with self._locked():
            try:
                stat = path.stat()
            except FileNotFoundError:
                return None
            if self.ttl_sec and time.time() - stat.st_mtime > self.ttl_sec:
                path.unlink(missing_ok=True)
                return None
            os.utime(path, None)
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(path, dest)
                except OSError:
                    shutil.copyfile(path, dest)
                path = dest
            return CacheEntry(path=path, byte_count=stat.st_size)

    def put_file(
        self,
        namespace: str,
        key: str,
        source: Path,
        *,
        suffix: str,
    ) -> CacheEntry:
        size = source.stat().st_size
        if not self.enabled or size > self.max_bytes:
            return CacheEntry(path=source, byte_count=size)
        path = self._path(namespace, key, suffix)
        with self._locked():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            os.utime(path, None)
            self._prune_locked(protect=path)
            return CacheEntry(path=path, byte_count=path.stat().st_size)

    def get_json(self, namespace: str, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key, ".json")
        with self._locked():
            try:
                if self.ttl_sec and time.time() - path.stat().st_mtime > self.ttl_sec:
                    path.unlink(missing_ok=True)
                    return None
                value = json.loads(path.read_text(encoding="utf-8"))
                os.utime(path, None)
            except (OSError, ValueError):
                path.unlink(missing_ok=True)
                return None
            return value if isinstance(value, dict) else None

    def put_json(self, namespace: str, key: str, value: dict[str, Any]) -> CacheEntry | None:
        if not self.enabled:
            return None
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(encoded) > self.max_bytes:
            return None
        path = self._path(namespace, key, ".json")
        with self._locked():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(encoded)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            self._prune_locked(protect=path)
            return CacheEntry(path=path, byte_count=path.stat().st_size)

    def _prune_locked(self, *, protect: Path) -> None:
        files = [
            path
            for path in self.root.glob("*/*")
            if path.is_file() and not path.name.startswith(".")
        ]
        records: list[tuple[float, int, Path]] = []
        total = 0
        now = time.time()
        for path in files:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if self.ttl_sec and now - stat.st_mtime > self.ttl_sec:
                path.unlink(missing_ok=True)
                continue
            total += stat.st_size
            records.append((stat.st_mtime, stat.st_size, path))
        for _mtime, size, path in sorted(records):
            if total <= self.max_bytes:
                break
            if path == protect:
                continue
            path.unlink(missing_ok=True)
            total -= size


_cache_lock = threading.Lock()
_caches: dict[tuple[str, int, float, bool], LocalMediaCache] = {}


def get_local_media_cache(
    root: Path,
    *,
    max_bytes: int,
    ttl_sec: float,
    enabled: bool,
) -> LocalMediaCache:
    identity = (str(root.resolve()), max_bytes, ttl_sec, enabled)
    with _cache_lock:
        cache = _caches.get(identity)
        if cache is None:
            cache = LocalMediaCache(
                root,
                max_bytes=max_bytes,
                ttl_sec=ttl_sec,
                enabled=enabled,
            )
            _caches[identity] = cache
        return cache
