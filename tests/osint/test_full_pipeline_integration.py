"""
Real, end-to-end OSINT pipeline integration test — zero mocks.

Unlike test_face_recognizer.py (which mocks Qdrant to test FaceRecognizer's
logic in isolation) and test_video_processor.py (which tests detection
only), this exercises the ENTIRE chain with real components: a real
generated video, real Haar-cascade face detection, real DeepFace embedding
extraction (downloads real model weights on first run), and a real Qdrant
instance (in-memory mode — qdrant-client's `location=":memory:"`, not a
mock) for indexing and vector search.

This is slow (downloads ~95MB of model weights on first run, runs real
CPU inference) and network-dependent, so it's opt-in like the live_chat
tests rather than part of the default fast suite.

Run explicitly with:

    RUN_LIVE_TESTS=1 pytest tests/osint/test_full_pipeline_integration.py -v

This test is what caught a real bug during development: FaceRecognizer
called qdrant_client.search(), which doesn't exist in current
qdrant-client versions (renamed to query_points()) — a bug that a fully
mocked test suite had completely missed, because an unspec'd AsyncMock()
silently accepts calls to methods that don't exist on the real object.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="Slow, network-dependent (downloads real model weights). Set RUN_LIVE_TESTS=1 to run.",
)


@pytest.fixture(scope="module")
def real_test_video() -> str:
    """
    Builds a real, short test video from a real face image, using ffmpeg.
    Requires network access (fetches a standard OpenCV sample image) and
    ffmpeg installed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "test_face.jpg"
        video_path = Path(tmpdir) / "test_video.mp4"

        subprocess.run(
            [
                "curl",
                "-sL",
                "-o",
                str(image_path),
                "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
            ],
            check=True,
            timeout=30,
        )
        assert image_path.stat().st_size > 10_000, "Test image download failed or returned a stub."

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-t",
                "2",
                "-vf",
                "fps=10",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        yield str(video_path)


@pytest.mark.asyncio
async def test_full_osint_pipeline_real_video_to_confident_match(real_test_video: str):
    """
    The complete chain, nothing mocked: video -> face detection -> DeepFace
    embedding -> Qdrant indexing -> Qdrant search -> confident match against
    itself.
    """
    from qdrant_client import AsyncQdrantClient

    from src.osint.face_recognizer import FaceRecognizer
    from src.osint.models import AnalysisPurpose, AnalysisScope
    from src.osint.video_processor import VideoProcessor

    qdrant = AsyncQdrantClient(location=":memory:")
    recognizer = FaceRecognizer(qdrant_client=qdrant)
    await recognizer.ensure_collection()

    vp = VideoProcessor(sample_interval_ms=1000)
    crops = await vp.extract_face_crops(real_test_video)
    assert len(crops) > 0, "Expected at least one real face detected in the test video."

    scope = AnalysisScope(
        purpose=AnalysisPurpose.OWNED_FOOTAGE_REVIEW,
        requested_by="integration-test",
        justification="automated end-to-end pipeline verification",
    )

    embedding = await recognizer.extract_embedding(
        scope=scope,
        image_path=crops[0].image_path,
        source_video_id=real_test_video,
        source_frame_timestamp_ms=crops[0].frame_timestamp_ms,
        detection_confidence=crops[0].detection_confidence,
    )
    assert len(embedding.vector) == 512  # Facenet512's real output dimensionality

    await recognizer.index_embedding(scope, embedding, identity_id="test-person")

    result = await recognizer.search_identity(scope, embedding)
    assert result.is_confident_match is True
    assert "test-person" in result.to_agent_summary()


@pytest.mark.asyncio
async def test_full_osint_pipeline_rejects_dissimilar_vector(real_test_video: str):
    """
    Confirms the pipeline correctly reports NO confident match for a
    genuinely different (random) embedding, not just that it can find a
    match for identical input — both directions matter for a system that
    must not hallucinate identity.
    """
    import random

    from qdrant_client import AsyncQdrantClient

    from src.osint.face_recognizer import FaceRecognizer
    from src.osint.models import AnalysisPurpose, AnalysisScope
    from src.osint.video_processor import VideoProcessor

    qdrant = AsyncQdrantClient(location=":memory:")
    recognizer = FaceRecognizer(qdrant_client=qdrant)
    await recognizer.ensure_collection()

    vp = VideoProcessor(sample_interval_ms=1000)
    crops = await vp.extract_face_crops(real_test_video)

    scope = AnalysisScope(
        purpose=AnalysisPurpose.OWNED_FOOTAGE_REVIEW,
        requested_by="integration-test",
        justification="automated end-to-end pipeline verification",
    )

    embedding = await recognizer.extract_embedding(
        scope=scope,
        image_path=crops[0].image_path,
        source_video_id=real_test_video,
        source_frame_timestamp_ms=crops[0].frame_timestamp_ms,
        detection_confidence=crops[0].detection_confidence,
    )
    await recognizer.index_embedding(scope, embedding, identity_id="test-person")

    random_embedding = embedding.model_copy(
        update={
            "vector": [random.random() for _ in range(512)],
            "embedding_id": "unrelated-random-vector",
        }
    )
    result = await recognizer.search_identity(scope, random_embedding)
    assert result.is_confident_match is False
    assert "UNCERTAIN" in result.to_agent_summary()
