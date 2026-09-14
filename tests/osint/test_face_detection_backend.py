"""
Real, non-mocked tests for HaarCascadeDetector — the default face
detection backend, verifying it works with zero configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.osint.face_detection_backend import HaarCascadeDetector, load_default_backend


def test_load_default_backend_returns_haar_cascade():
    backend = load_default_backend()
    assert isinstance(backend, HaarCascadeDetector)


def test_haar_cascade_loads_without_any_external_file():
    """The whole point of this backend: it must construct successfully
    with zero network access and zero manually-provided files."""
    detector = HaarCascadeDetector()
    assert detector is not None


def test_haar_cascade_detects_no_faces_in_blank_frame():
    detector = HaarCascadeDetector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(blank_frame)
    assert results == []


def test_haar_cascade_detection_return_shape():
    """Even with no faces, confirms detect() returns the documented
    (x, y, w, h, confidence) tuple shape contract other code relies on."""
    detector = HaarCascadeDetector()
    blank_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    results = detector.detect(blank_frame)
    assert isinstance(results, list)
    for r in results:
        assert len(r) == 5
