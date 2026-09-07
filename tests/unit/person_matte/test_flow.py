import cv2
import numpy as np
import pytest

from media_analysis.person_matte.pipeline import propagate_alpha_with_flow
from media_analysis.person_matte.temporal import TemporalMatteState


@pytest.mark.unit
def test_nonuniform_flow_uses_current_to_previous_coordinates(monkeypatch):
    height, width = 16, 64
    previous = np.zeros((height, width), np.uint8)
    previous[:, 12:16] = 255
    previous_frame = np.zeros((height, width, 3), np.uint8)
    current_frame = np.full((height, width, 3), 100, np.uint8)

    def flow(current_gray, previous_gray, *args):
        assert np.all(current_gray == 100)
        assert np.all(previous_gray == 0)
        # Exact backward mapping for horizontal scale 2: previous_x = current_x / 2.
        displacement = np.zeros((height, width, 2), np.float32)
        displacement[:, :, 0] = -np.arange(width, dtype=np.float32) / 2
        return displacement

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", flow)
    actual = propagate_alpha_with_flow(previous_frame, current_frame, previous)
    assert np.all(actual[:, 24:31] == 255)
    assert np.all(actual[:, :23] == 0)
    assert np.all(actual[:, 32:] == 0)


@pytest.mark.unit
def test_motion_aligned_ema_does_not_leave_foreground_at_old_position():
    state = TemporalMatteState(())
    previous = np.zeros((10, 30), np.uint8)
    previous[:, 5:10] = 255
    current = np.zeros((10, 30), np.uint8)
    current[:, 15:20] = 255
    state.apply(previous, source_frame=0)
    actual = state.apply(current, source_frame=1, aligned_previous=current)
    np.testing.assert_array_equal(actual, current)


@pytest.mark.unit
def test_low_resolution_flow_restores_displacement_in_output_pixels(monkeypatch):
    # Uneven dimensions exercise independent x/y scale factors after rounding.
    height, width = 301, 603
    alpha = np.zeros((height, width), np.uint8)
    alpha[80:120, 100:140] = 255
    frame = np.zeros((height, width, 3), np.uint8)

    def flow(current_gray, previous_gray, *args):
        work_h, work_w = current_gray.shape
        assert max(work_h, work_w) == 256
        assert previous_gray.shape == current_gray.shape
        displacement = np.zeros((work_h, work_w, 2), np.float32)
        displacement[:, :, 0] = -12 * work_w / width
        displacement[:, :, 1] = -8 * work_h / height
        return displacement

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", flow)
    actual = propagate_alpha_with_flow(frame, frame, alpha)
    assert actual.shape == alpha.shape
    np.testing.assert_array_equal(actual[88:128, 112:152], 255)
    assert np.count_nonzero(actual) == 40 * 40


def test_flow_rejects_invalid_working_resolution():
    frame = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        propagate_alpha_with_flow(frame, frame, frame[:, :, 0], max_dimension=1)
