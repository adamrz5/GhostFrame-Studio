from __future__ import annotations

import numpy as np

from core.video_processor import censure_roi_inplace, apply_censure


def _frame() -> np.ndarray:
    frame = np.full((20, 20, 3), 128, dtype=np.uint8)
    frame[5:15, 5:15] = 220
    return frame


def test_apply_censure_does_not_mutate_input_frame():
    frame = _frame()
    original = frame.copy()

    apply_censure(frame, [5, 5, 15, 15], "blur", 5, 0.0)

    assert np.array_equal(frame, original)


def test_apply_censure_output_shape_equals_input_shape():
    frame = _frame()

    out = apply_censure(frame, [5, 5, 15, 15], "pixelate", 5, 0.1)

    assert out.shape == frame.shape


def test_apply_censure_blackbox_zeroes_roi_center():
    frame = _frame()

    out = apply_censure(frame, [5, 5, 15, 15], "blackbox", 5, 0.0)

    assert np.array_equal(out[10, 10], np.zeros(3, dtype=np.uint8))


def test_censure_roi_inplace_modifies_frame_in_place():
    frame = _frame()
    original = frame.copy()

    censure_roi_inplace(frame, [5, 5, 15, 15], "blackbox", 5, 0.0)

    assert not np.array_equal(frame, original)
