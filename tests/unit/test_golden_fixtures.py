import pytest
from tests.fixtures.generate import (
    write_faces_circles_mp4,
    write_ocr_text_mp4,
    write_people_shapes_mp4,
    write_rotated_mp4,
    write_speech_tone_mp4,
    write_vfr_mp4,
)

from media_analysis.decode import canonicalize, probe


@pytest.mark.unit
@pytest.mark.ffmpeg
def test_generated_goldens_cover_required_categories(tmp_path) -> None:
    people = write_people_shapes_mp4(tmp_path / "people.mp4")
    faces = write_faces_circles_mp4(tmp_path / "faces.mp4")
    ocr = write_ocr_text_mp4(tmp_path / "ocr.mp4")
    speech = write_speech_tone_mp4(tmp_path / "speech.mp4")
    vfr = write_vfr_mp4(tmp_path / "vfr.mp4")
    rotated = write_rotated_mp4(tmp_path / "rotated.mp4")

    for path in (people, faces, ocr, rotated):
        media = probe(path)
        assert media.width == 320
        assert media.height == 240
        assert media.frame_count >= 15
        assert media.has_audio is False

    spoken = probe(speech)
    assert spoken.has_audio is True
    assert spoken.audio_sample_rate == 48000

    source = probe(vfr)
    canonical = canonicalize(vfr, tmp_path / "vfr-cfr.mp4", preserve_audio=False)
    assert canonical.fps.as_dict() == {"numerator": 30, "denominator": 1}
    assert canonical.frame_count != source.frame_count or source.fps.as_dict() != {
        "numerator": 30,
        "denominator": 1,
    }
