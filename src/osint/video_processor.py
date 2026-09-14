"""
Video frame extraction and face-crop detection.

This module produces face crops and hands them to FaceRecognizer for
embedding — it never computes embeddings or does identity search itself,
keeping the scope-gated identity logic in exactly one place
(face_recognizer.py).

Detection is delegated to a FaceDetectionBackend (see
face_detection_backend.py) so the concrete detector (Haar cascade by
default, YuNet if available) is swappable without touching this module.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.osint.face_detection_backend import FaceDetectionBackend, load_default_backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceCrop:
    """A single detected face, cropped from one frame, on disk as a temp file."""

    image_path: str
    frame_timestamp_ms: int
    detection_confidence: float


class VideoProcessingError(Exception):
    """Raised for frame extraction / face detection failures."""


class VideoProcessor:
    """
    Extracts frames from a video at a fixed sampling interval and detects
    faces in each sampled frame.

    Deliberately does NOT do identity matching — see module docstring.
    """

    def __init__(
        self,
        sample_interval_ms: int = 1000,
        min_detection_confidence: float = 0.6,
        face_detector: FaceDetectionBackend | None = None,
    ) -> None:
        self._sample_interval_ms = sample_interval_ms
        self._min_detection_confidence = min_detection_confidence
        # Defaults to the bundled, zero-download Haar cascade backend.
        # Pass a YuNetDetector (face_detection_backend.py) for higher
        # accuracy once you've downloaded that model — see
        # face_detector_loader.py.
        self._backend: FaceDetectionBackend = face_detector or load_default_backend()

    def _process_sync(self, video_path: str, output_dir: str) -> list[FaceCrop]:
        """Blocking OpenCV work — always invoked via asyncio.to_thread."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video: {video_path!r}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, round(fps * (self._sample_interval_ms / 1000.0)))

        crops: list[FaceCrop] = []
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    timestamp_ms = int((frame_index / fps) * 1000)
                    detections = self._backend.detect(frame)

                    for x, y, fw, fh, confidence in detections:
                        if confidence < self._min_detection_confidence:
                            continue

                        x, y = max(0, x), max(0, y)
                        crop = frame[y : y + fh, x : x + fw]
                        if crop.size == 0:
                            continue

                        out_path = str(
                            Path(output_dir) / f"face_{timestamp_ms}_{len(crops)}.jpg"
                        )
                        cv2.imwrite(out_path, crop)
                        crops.append(
                            FaceCrop(
                                image_path=out_path,
                                frame_timestamp_ms=timestamp_ms,
                                detection_confidence=confidence,
                            )
                        )

                frame_index += 1
        finally:
            cap.release()

        return crops

    async def extract_face_crops(self, video_path: str) -> list[FaceCrop]:
        """
        Sample frames from `video_path` and return detected face crops.

        Crops are written to a temp directory that the caller (typically
        an orchestrator tool node) is responsible for cleaning up after
        embeddings have been extracted from them.
        """
        output_dir = tempfile.mkdtemp(prefix="nexus_osint_")
        try:
            crops = await asyncio.to_thread(self._process_sync, video_path, output_dir)
        except VideoProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VideoProcessingError(f"Frame extraction failed: {exc}") from exc

        logger.info(
            "Extracted %d face crop(s) from %s (sample_interval_ms=%d)",
            len(crops),
            video_path,
            self._sample_interval_ms,
        )
        return crops
