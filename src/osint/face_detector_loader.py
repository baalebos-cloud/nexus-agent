"""
Loads the (optional) YuNet face-detection model into a cv2.FaceDetectorYN
instance.

NOT required to run the OSINT pipeline — VideoProcessor defaults to a
bundled Haar cascade detector (see face_detection_backend.py) that needs
zero downloads and works immediately. This loader exists for anyone who
wants YuNet's higher accuracy (especially on non-frontal/angled faces) as
an upgrade.

The model file (~233KB ONNX) is distributed via Git LFS on the OpenCV Zoo
GitHub repo, and this project's sandboxed build/dev environment could not
reach GitHub's LFS media storage (only api.github.com /
raw.githubusercontent.com metadata endpoints were reachable, not
media.githubusercontent.com) — hence it isn't bundled here. Download it
yourself, from a machine with normal internet access, if you want it:

    curl -L -o models/face_detection_yunet_2023mar.onnx \\
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

(If that raw/ redirect 404s or returns a tiny ~130 byte file due to LFS,
use `git lfs pull` after cloning https://github.com/opencv/opencv_zoo
instead, then copy the file out.)

Place it at the path configured by Settings.face_detector_model_path
(defaults to `models/face_detection_yunet_2023mar.onnx` relative to the
process working directory) and server.py will pick it up automatically on
next startup — no code changes needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class FaceDetectorModelMissingError(Exception):
    """Raised when the configured YuNet model file isn't present on disk."""


def load_face_detector(
    model_path: str,
    input_size: tuple[int, int] = (320, 320),
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> cv2.FaceDetectorYN:
    """
    Construct a cv2.FaceDetectorYN from a model file on disk.

    `input_size` is a placeholder; VideoProcessor calls `setInputSize` per
    frame with the actual frame dimensions before each `detect()` call, so
    this initial value doesn't need to match your video resolution.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FaceDetectorModelMissingError(
            f"YuNet model not found at {model_path!r}. See the module docstring "
            f"in src/osint/face_detector_loader.py for how to download it — it "
            f"cannot be fetched automatically inside this build environment."
        )

    detector = cv2.FaceDetectorYN.create(
        model=str(path),
        config="",
        input_size=input_size,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )
    logger.info("Loaded YuNet face detector from %s", model_path)
    return detector
