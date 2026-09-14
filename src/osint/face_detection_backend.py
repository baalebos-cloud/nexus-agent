"""
Pluggable face-detection backends for VideoProcessor.

The original design required a separately-downloaded YuNet ONNX model,
which turned out to be undownloadable from this project's build
environment (Git LFS binary storage isn't reachable through the allowed
network). Rather than leave the OSINT pipeline permanently blocked on a
manual step, the default backend now uses OpenCV's Haar cascade
classifier, which ships bundled inside the `opencv-python-headless`
package itself — zero network access required, works immediately after
`pip install`.

YuNet remains available and is used automatically if you do go through the
manual download (see face_detector_loader.py) — it's meaningfully more
accurate, especially on non-frontal faces — but it's an upgrade, not a
requirement to get the pipeline running.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetectionBackend(Protocol):
    """
    Common interface both detector implementations satisfy, so
    VideoProcessor doesn't need to know which one it's using.
    """

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Returns a list of (x, y, w, h, confidence) for each detected face."""
        ...


class HaarCascadeDetector:
    """
    Default detector — bundled with opencv-python-headless, no download
    needed. Less accurate than YuNet on profile/angled faces, but gets the
    pipeline running immediately with zero external dependencies.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError(
                f"Failed to load bundled Haar cascade from {cascade_path!r} — "
                f"check your opencv-python-headless installation."
            )
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        # Haar cascades don't produce a real confidence score — report a
        # fixed nominal value so downstream code (which expects one) still
        # works. This is honestly reflected in the confidence value: it's
        # not a calibrated probability, just "detected."
        return [(int(x), int(y), int(w), int(h), 0.99) for (x, y, w, h) in boxes]


class YuNetDetector:
    """
    Higher-accuracy detector, used automatically once the YuNet ONNX model
    is present on disk — see face_detector_loader.py for the download step.
    """

    def __init__(self, cv2_face_detector_yn: cv2.FaceDetectorYN) -> None:
        self._detector = cv2_face_detector_yn

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [
            (int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces
        ]


def load_default_backend() -> FaceDetectionBackend:
    """The zero-configuration default: always available, no download."""
    return HaarCascadeDetector()
