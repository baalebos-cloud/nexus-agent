from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import QueryResponse, ScoredPoint

from src.osint.access_control import ScopeError
from src.osint.face_recognizer import FaceRecognizer
from src.osint.models import AnalysisPurpose, AnalysisScope, FaceEmbedding


def _make_scoped_hit(point_id: str, score: float, identity_id: str) -> ScoredPoint:
    """
    A real ScoredPoint, not a MagicMock — QueryResponse validates `.points`
    strictly as ScoredPoint instances (confirmed: a plain MagicMock fails
    Pydantic validation here), so a loose mock can't stand in for it.
    """
    return ScoredPoint(id=point_id, version=1, score=score, payload={"identity_id": identity_id})


def _make_scope(**overrides) -> AnalysisScope:
    defaults = dict(
        purpose=AnalysisPurpose.OWNED_FOOTAGE_REVIEW,
        requested_by="alice",
        justification="reviewing my own security footage from last night",
    )
    defaults.update(overrides)
    return AnalysisScope(**defaults)


def _make_embedding(**overrides) -> FaceEmbedding:
    defaults = dict(
        vector=[0.1] * 512,
        source_frame_timestamp_ms=1000,
        source_video_id="vid-1",
        detection_confidence=0.9,
    )
    defaults.update(overrides)
    return FaceEmbedding(**defaults)


def _fake_qdrant_client() -> AsyncMock:
    """
    spec=AsyncQdrantClient is deliberate, not decorative: a real bug
    (calling the removed `.search()` method instead of the current
    `.query_points()`) previously went completely undetected by this test
    suite, because an unspec'd AsyncMock() silently accepts calls to
    methods that don't exist on the real client at all. With spec= set,
    calling a nonexistent method raises AttributeError immediately, the
    same way the real client would.
    """
    return AsyncMock(spec=AsyncQdrantClient)


@pytest.mark.asyncio
async def test_search_identity_rejects_none_scope():
    recognizer = FaceRecognizer(qdrant_client=_fake_qdrant_client())
    with pytest.raises(ScopeError):
        await recognizer.search_identity(None, _make_embedding())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_index_embedding_rejects_none_scope():
    recognizer = FaceRecognizer(qdrant_client=_fake_qdrant_client())
    with pytest.raises(ScopeError):
        await recognizer.index_embedding(None, _make_embedding())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_search_identity_returns_uncertain_below_threshold():
    fake_hit = _make_scoped_hit("pt1", score=0.55, identity_id="person-1")
    qdrant = _fake_qdrant_client()
    qdrant.query_points = AsyncMock(return_value=QueryResponse(points=[fake_hit]))

    recognizer = FaceRecognizer(qdrant_client=qdrant)
    scope = _make_scope()
    result = await recognizer.search_identity(scope, _make_embedding())

    assert result.is_confident_match is False
    assert "UNCERTAIN" in result.to_agent_summary()


@pytest.mark.asyncio
async def test_search_identity_returns_confident_above_threshold():
    fake_hit = _make_scoped_hit("pt2", score=0.91, identity_id="person-2")
    qdrant = _fake_qdrant_client()
    qdrant.query_points = AsyncMock(return_value=QueryResponse(points=[fake_hit]))

    recognizer = FaceRecognizer(qdrant_client=qdrant)
    scope = _make_scope()
    result = await recognizer.search_identity(scope, _make_embedding())

    assert result.is_confident_match is True
    assert "person-2" in result.to_agent_summary()
    assert "not a verified identity" in result.to_agent_summary()


@pytest.mark.asyncio
async def test_search_identity_rejects_expired_scope():
    from datetime import datetime, timedelta, timezone

    scope = _make_scope(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    recognizer = FaceRecognizer(qdrant_client=_fake_qdrant_client())
    with pytest.raises(ScopeError):
        await recognizer.search_identity(scope, _make_embedding())


@pytest.mark.asyncio
async def test_law_enforcement_scope_without_case_reference_fails_at_construction():
    with pytest.raises(ValueError):
        _make_scope(purpose=AnalysisPurpose.LAW_ENFORCEMENT_REQUEST, case_reference=None)


@pytest.mark.asyncio
async def test_index_embedding_does_not_require_identity_id():
    """Indexing an unknown face (no identity_id) should succeed — identity
    attachment happens later through an independent confirmation step."""
    qdrant = _fake_qdrant_client()
    qdrant.upsert = AsyncMock()

    recognizer = FaceRecognizer(qdrant_client=qdrant)
    scope = _make_scope()
    await recognizer.index_embedding(scope, _make_embedding())  # no identity_id passed

    qdrant.upsert.assert_awaited_once()
    _, kwargs = qdrant.upsert.call_args
    payload = kwargs["points"][0].payload
    assert "identity_id" not in payload
