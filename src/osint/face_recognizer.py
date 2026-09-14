"""
Facial embedding extraction and identity-matching against Qdrant.

Every public entrypoint here takes an `AnalysisScope` and routes it through
`require_valid_scope` before touching DeepFace or Qdrant. There is no
scope-free code path — see src/osint/access_control.py.

Results are always returned as `IdentityMatchResult`, never raw distances
or raw DeepFace output, so the 0.7-threshold uncertainty framing
(models.py: `IdentityMatchResult.to_agent_summary`) can't be bypassed by a
caller reaching for a "lower-level" function.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from src.osint.access_control import IdentityVerifier, require_valid_scope
from src.osint.models import (
    AnalysisScope,
    FaceEmbedding,
    IdentityMatchCandidate,
    IdentityMatchResult,
)

logger = logging.getLogger(__name__)

QDRANT_COLLECTION_NAME = "face_embeddings"
_TOP_K_CANDIDATES = 5


class FaceRecognitionError(Exception):
    """Raised for embedding/indexing/search failures not related to scope."""


class FaceRecognizer:
    """
    Wraps DeepFace embedding extraction and Qdrant vector search behind the
    AnalysisScope gate.

    DeepFace itself is imported lazily inside methods rather than at module
    level — it pulls in TensorFlow/torch backends that are heavy to import
    and unnecessary for any code path that only reads/searches existing
    embeddings (e.g. re-running a search against previously indexed video).
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        identity_verifier: IdentityVerifier | None = None,
        embedding_model: str = "Facenet512",
    ) -> None:
        self._qdrant = qdrant_client
        self._verifier = identity_verifier
        self._embedding_model = embedding_model

    async def ensure_collection(self, vector_size: int = 512) -> None:
        """Idempotently create the Qdrant collection. Call once at startup."""
        collections = await self._qdrant.get_collections()
        existing = {c.name for c in collections.collections}
        if QDRANT_COLLECTION_NAME in existing:
            return
        await self._qdrant.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size, distance=qdrant_models.Distance.COSINE
            ),
        )
        logger.info("Created Qdrant collection %r (size=%d)", QDRANT_COLLECTION_NAME, vector_size)

    def _extract_embedding_sync(self, image_path: str) -> list[float]:
        """Blocking DeepFace call — always invoked via asyncio.to_thread."""
        from deepface import DeepFace  # lazy import, see class docstring

        result: list[dict[str, Any]] = DeepFace.represent(
            img_path=image_path,
            model_name=self._embedding_model,
            enforce_detection=True,
        )
        if not result:
            raise FaceRecognitionError(f"No face detected in {image_path!r}.")
        # DeepFace.represent returns one entry per detected face; the
        # calling video_processor.py is responsible for one-face-per-crop
        # framing upstream, so we take the first/best here.
        return result[0]["embedding"]

    async def extract_embedding(
        self,
        scope: AnalysisScope,
        image_path: str,
        source_video_id: str,
        source_frame_timestamp_ms: int,
        detection_confidence: float,
    ) -> FaceEmbedding:
        """Extract a facial embedding from a single face-crop image."""
        require_valid_scope(scope, self._verifier)

        import asyncio

        try:
            vector = await asyncio.to_thread(self._extract_embedding_sync, image_path)
        except FaceRecognitionError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize DeepFace/backend errors
            raise FaceRecognitionError(f"Embedding extraction failed: {exc}") from exc

        return FaceEmbedding(
            vector=vector,
            source_frame_timestamp_ms=source_frame_timestamp_ms,
            source_video_id=source_video_id,
            detection_confidence=detection_confidence,
        )

    async def index_embedding(
        self, scope: AnalysisScope, embedding: FaceEmbedding, identity_id: str | None = None
    ) -> None:
        """
        Store an embedding in Qdrant for future search.

        `identity_id` is optional at index time — you can index an unknown
        face (e.g. "person seen in frame X") and only attach an identity
        label later once one is confirmed through an independent source,
        consistent with the no-hallucinated-identity requirement.
        """
        require_valid_scope(scope, self._verifier)

        payload: dict[str, Any] = {
            "source_video_id": embedding.source_video_id,
            "source_frame_timestamp_ms": embedding.source_frame_timestamp_ms,
            "detection_confidence": embedding.detection_confidence,
            "scope_id": scope.scope_id,
        }
        if identity_id is not None:
            payload["identity_id"] = identity_id

        await self._qdrant.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=embedding.embedding_id,
                    vector=embedding.vector,
                    payload=payload,
                )
            ],
        )

    async def search_identity(
        self, scope: AnalysisScope, query_embedding: FaceEmbedding, top_k: int = _TOP_K_CANDIDATES
    ) -> IdentityMatchResult:
        """
        Search for candidate identity matches for a query embedding.

        Returns ALL top-k candidates regardless of confidence — filtering
        for "is this actually a match" happens in IdentityMatchResult's
        derived properties (models.py), not here, so there's exactly one
        place that logic lives.
        """
        require_valid_scope(scope, self._verifier)

        # `.search()` is removed in current qdrant-client — confirmed via a
        # real (non-mocked) end-to-end run, not an assumption. The current
        # replacement is `.query_points()`, which returns a QueryResponse
        # wrapping the hit list in `.points` rather than returning the
        # list directly.
        response = await self._qdrant.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding.vector,
            limit=top_k,
        )
        hits = response.points

        candidates = [
            IdentityMatchCandidate(
                candidate_identity_id=str(hit.payload.get("identity_id", hit.id)),
                cosine_similarity=hit.score,
            )
            for hit in hits
        ]

        result = IdentityMatchResult(
            query_embedding_id=query_embedding.embedding_id,
            scope_id=scope.scope_id,
            top_candidates=candidates,
        )
        logger.info(
            "Identity search scope_id=%s query_embedding_id=%s confident_match=%s",
            scope.scope_id,
            query_embedding.embedding_id,
            result.is_confident_match,
        )
        return result
