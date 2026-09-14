from __future__ import annotations

from src.osint.face_detection_backend import HaarCascadeDetector
from src.osint.video_processor import VideoProcessor


def test_video_processor_defaults_to_haar_cascade_with_zero_config():
    """
    This is the core fix: VideoProcessor must be constructible and usable
    with no face_detector argument at all — the original design required
    a manually-downloaded YuNet model, which made the OSINT pipeline
    permanently blocked without a manual step. It no longer does.
    """
    vp = VideoProcessor()
    assert isinstance(vp._backend, HaarCascadeDetector)
