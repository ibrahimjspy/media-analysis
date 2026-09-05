from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from media_analysis.cache import LocalMediaCache


@pytest.mark.unit
def test_cache_publishes_files_and_json_by_stable_digest(tmp_path: Path) -> None:
    cache = LocalMediaCache(tmp_path / "cache", max_bytes=1024, ttl_sec=60)
    key = LocalMediaCache.stable_key("media", "v1")
    source = tmp_path / "source.bin"
    source.write_bytes(b"video")

    stored = cache.put_file("source", key, source, suffix=".media")
    assert stored.path != source
    assert cache.get("source", key, suffix=".media").path.read_bytes() == b"video"

    cache.put_json("result", key, {"status": "completed"})
    assert cache.get_json("result", key) == {"status": "completed"}


@pytest.mark.unit
def test_cache_prunes_expired_entries(tmp_path: Path) -> None:
    cache = LocalMediaCache(tmp_path / "cache", max_bytes=1024, ttl_sec=0.01)
    key = LocalMediaCache.stable_key("old")
    source = tmp_path / "source.bin"
    source.write_bytes(b"video")
    entry = cache.put_file("source", key, source, suffix=".media")
    old = time.time() - 1
    os.utime(entry.path, (old, old))
    assert cache.get("source", key, suffix=".media") is None


@pytest.mark.unit
def test_cache_prunes_lru_to_byte_budget(tmp_path: Path) -> None:
    cache = LocalMediaCache(tmp_path / "cache", max_bytes=6, ttl_sec=60)
    first_key = LocalMediaCache.stable_key("first")
    second_key = LocalMediaCache.stable_key("second")
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"1111")
    second.write_bytes(b"2222")
    cache.put_file("source", first_key, first, suffix=".media")
    cache.put_file("source", second_key, second, suffix=".media")
    assert cache.get("source", first_key, suffix=".media") is None
    assert cache.get("source", second_key, suffix=".media") is not None


@pytest.mark.unit
def test_checked_out_media_survives_eviction_by_another_instance(tmp_path: Path) -> None:
    cache = LocalMediaCache(tmp_path / "cache", max_bytes=80, ttl_sec=60)
    other = LocalMediaCache(tmp_path / "cache", max_bytes=80, ttl_sec=60)
    key = LocalMediaCache.stable_key("video")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x" * 70)
    cache.put_file("canonical", key, source, suffix=".mp4")
    active = cache.get("canonical", key, suffix=".mp4", dest=tmp_path / "job.mp4")
    other.put_json("result", LocalMediaCache.stable_key("result"), {"overallStatus": "completed"})
    assert cache.get("canonical", key, suffix=".mp4") is None
    assert active.path.read_bytes() == source.read_bytes()


@pytest.mark.unit
def test_oversized_entries_bypass_cache_without_failing(tmp_path: Path) -> None:
    cache = LocalMediaCache(tmp_path / "cache", max_bytes=8, ttl_sec=60)
    key = LocalMediaCache.stable_key("large")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x" * 16)
    assert cache.put_file("source", key, source, suffix=".media").path == source
    assert cache.put_json("result", key, {"too": "large"}) is None
    assert cache.get("source", key, suffix=".media") is None


@pytest.mark.unit
def test_disabled_cache_is_a_noop(tmp_path: Path) -> None:
    cache = LocalMediaCache(tmp_path / "disabled", max_bytes=8, ttl_sec=60, enabled=False)
    key = LocalMediaCache.stable_key("disabled")
    assert cache.put_json("result", key, {"status": "completed"}) is None
    assert cache.get_json("result", key) is None
    assert not cache.root.exists()
