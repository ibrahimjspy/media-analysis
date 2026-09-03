import pytest

from media_analysis.build_provenance import (
    _configuration_line,
    _first_token_after,
    collect_ffmpeg_build,
    collect_machine_fingerprint,
    collect_runtime_build,
)


@pytest.mark.unit
def test_ffmpeg_version_parser_reads_standard_banner() -> None:
    banner = (
        "ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers\n"
        "built with Apple clang version 16.0.0\n"
        "configuration: --prefix=/opt/ffmpeg --enable-gpl --enable-libx264\n"
    )
    assert _first_token_after(banner, "ffmpeg version") == "7.1.1"
    assert "--enable-libx264" in _configuration_line(banner)


@pytest.mark.unit
def test_runtime_build_reports_installed_libraries() -> None:
    build = collect_runtime_build()
    assert build["pythonVersion"]
    assert build["onnxruntimeVersion"]
    assert build["opencvVersion"]
    assert build["numpyVersion"]


@pytest.mark.unit
def test_machine_fingerprint_names_the_host() -> None:
    fingerprint = collect_machine_fingerprint()
    assert fingerprint["platform"]
    assert fingerprint["machine"]
    assert fingerprint["cpuCount"]


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_ffmpeg_build_captures_local_binary() -> None:
    build = collect_ffmpeg_build()
    assert build["ffmpegVersion"] != "unavailable"
    assert build["ffprobeVersion"] != "unavailable"
    assert "enable" in build["ffmpegConfiguration"] or build["ffmpegConfiguration"]
