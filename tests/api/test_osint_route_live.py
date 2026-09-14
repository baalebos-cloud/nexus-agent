"""
Real, end-to-end test of the /osint/analyze-video route — patches only the
Qdrant *transport* (in-memory instead of a networked server, since no
standalone Qdrant binary/Docker is available in this environment) while
exercising the actual FastAPI route, real request validation, real
dependency wiring from app.state, real video processing, and real DeepFace
embedding extraction. Nothing about the OSINT logic itself is mocked.

This is the one layer the other OSINT tests didn't cover: proof that the
HTTP route itself (not just the underlying FaceRecognizer/VideoProcessor
classes) works when the pipeline IS configured, not just when it's
unconfigured (which test_server_integration.py already covers via 503
tests).

Opt-in like the other slow/network-dependent tests:

    RUN_LIVE_TESTS=1 pytest tests/api/test_osint_route_live.py -v
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="Slow, network-dependent (downloads real model weights). Set RUN_LIVE_TESTS=1 to run.",
)


@pytest.fixture(scope="module")
def real_test_video() -> str:
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
        assert image_path.stat().st_size > 10_000

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


@pytest.fixture
def client_with_real_in_memory_qdrant():
    """
    Patches AsyncQdrantClient construction inside server.py's lifespan to
    use in-memory mode, so the route gets a genuine, working Qdrant client
    without needing a networked server. Everything downstream (routes.py,
    FaceRecognizer, VideoProcessor) runs completely unmodified/unmocked.
    """
    with patch("src.api.server.AsyncQdrantClient") as mock_ctor:
        mock_ctor.side_effect = lambda **kwargs: AsyncQdrantClient(location=":memory:")
        from src.api.server import app

        with TestClient(app) as c:
            yield c


def test_osint_analyze_video_route_end_to_end(
    client_with_real_in_memory_qdrant, real_test_video: str
):
    """
    Hits the real HTTP route with a real video file path. If this returns
    200 with a well-formed summary, the entire chain — request validation,
    AnalysisScope construction, app.state wiring, real face detection,
    real DeepFace embedding, real Qdrant search — works through the
    actual API surface a client would call, not just the underlying
    Python classes in isolation.

    Note: this deliberately does NOT expect a confident match. The
    analyze_video_for_faces tool (src/orchestrator/osint_tools.py)
    extracts embeddings and SEARCHES against whatever is already indexed
    — it never indexes the video's own faces as a side effect. Against a
    freshly created, empty Qdrant collection, "No candidate matches
    found." for every detected face is the correct, expected response —
    not a failure. An earlier version of this test wrongly asserted a
    self-match here, which was the test's incorrect assumption, not a
    bug in the route.
    """
    resp = client_with_real_in_memory_qdrant.post(
        "/osint/analyze-video",
        json={
            "video_path": real_test_video,
            "purpose": "owned_footage_review",
            "requested_by": "integration-test",
            "justification": "automated end-to-end route verification",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scope_id"]
    # Correct behavior against an empty index: every detected face frame
    # reports no match, with the uncertainty framing intact (never a
    # fabricated identity claim).
    assert "No candidate matches found." in body["summary"]
    assert "frame" in body["summary"]  # confirms real per-frame detection ran


def test_osint_analyze_video_route_finds_match_after_pre_indexing(
    client_with_real_in_memory_qdrant, real_test_video: str
):
    """
    Completes the round trip the previous test deliberately didn't: index
    a face from the SAME video directly via app.state.face_recognizer
    (simulating a prior "index this known person" step), then hit the
    real HTTP route and confirm it reports a confident match against the
    now-populated index. This proves the full chain works for the
    genuinely intended use case, not just the empty-index case.
    """
    import asyncio

    from src.osint.models import AnalysisPurpose, AnalysisScope
    from src.osint.video_processor import VideoProcessor

    app = client_with_real_in_memory_qdrant.app
    face_recognizer = app.state.face_recognizer

    scope = AnalysisScope(
        purpose=AnalysisPurpose.OWNED_FOOTAGE_REVIEW,
        requested_by="integration-test-preindex",
        justification="pre-indexing a known face before route-level search test",
    )

    async def _pre_index() -> None:
        vp = VideoProcessor(sample_interval_ms=1000)
        crops = await vp.extract_face_crops(real_test_video)
        embedding = await face_recognizer.extract_embedding(
            scope=scope,
            image_path=crops[0].image_path,
            source_video_id=real_test_video,
            source_frame_timestamp_ms=crops[0].frame_timestamp_ms,
            detection_confidence=crops[0].detection_confidence,
        )
        await face_recognizer.index_embedding(scope, embedding, identity_id="known-person")

    asyncio.run(_pre_index())

    resp = client_with_real_in_memory_qdrant.post(
        "/osint/analyze-video",
        json={
            "video_path": real_test_video,
            "purpose": "owned_footage_review",
            "requested_by": "integration-test",
            "justification": "automated end-to-end route verification, post pre-indexing",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Likely match" in body["summary"]
    assert "known-person" in body["summary"]


def test_osint_analyze_video_route_rejects_invalid_scope(client_with_real_in_memory_qdrant):
    """Confirms input validation still happens correctly even when the
    pipeline IS fully configured (not just in the 503-unconfigured case
    already covered elsewhere)."""
    resp = client_with_real_in_memory_qdrant.post(
        "/osint/analyze-video",
        json={
            "video_path": "irrelevant.mp4",
            "purpose": "law_enforcement_request",
            "requested_by": "someone",
            "justification": "case work requested by department",
            "case_reference": None,
        },
    )
    assert resp.status_code == 400
