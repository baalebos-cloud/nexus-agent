"""
Data models for the Video OSINT Pipeline.

This module is deliberately built before any matching/inference logic
(face_recognizer.py, video_processor.py). Per PROJECT_CONTEXT.md Rule 3 and
the scope note established at the start of this build: identity-matching
capability is high-risk for enabling surveillance or stalking if the
guardrails aren't structural. So the enforcement objects live here, in the
same module boundary as the data shapes, not bolted on as an afterthought
in whatever calls this code.

Two guardrails are load-bearing and non-optional:
  1. `AnalysisScope` — every OSINT job must declare what it's for and who
     authorized it. There is no "just match this face" entrypoint without
     a scope attached.
  2. Similarity confidence gating — per Rule 3, any match below the 0.7
     cosine-similarity threshold MUST be surfaced as explicit uncertainty,
     never presented as a confirmed identity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

IDENTITY_MATCH_CONFIDENCE_THRESHOLD = 0.7


class AnalysisPurpose(str, Enum):
    """
    Declared, auditable reason for running identity-matching on a video.

    This is not a free-text field on purpose — an enumerated, reviewable
    set of purposes is what makes downstream access-control and audit
    logging meaningful. If a real use case doesn't fit one of these,
    it needs its own reviewed category, not a string the caller invents.
    """

    MISSING_PERSON_CASE = "missing_person_case"
    CONSENTED_RESEARCH = "consented_research"  # subjects have given informed consent
    OWNED_FOOTAGE_REVIEW = "owned_footage_review"  # e.g. reviewing your own security footage
    LAW_ENFORCEMENT_REQUEST = "law_enforcement_request"  # requires case/warrant reference


class AnalysisScope(BaseModel):
    """
    Authorization envelope required for any OSINT identity-matching job.

    `face_recognizer.py` and `video_processor.py` should refuse to run
    without a valid, non-expired AnalysisScope attached to the request —
    enforced in policy.py, not left to caller discipline.
    """

    scope_id: str = Field(default_factory=lambda: str(uuid4()))
    purpose: AnalysisPurpose
    requested_by: str = Field(..., min_length=1, description="Identity of the requester, for audit trail.")
    case_reference: str | None = Field(
        default=None,
        description="Case/ticket/warrant number. Required for LAW_ENFORCEMENT_REQUEST.",
    )
    justification: str = Field(..., min_length=10, description="Human-readable reason, logged for audit.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = Field(
        default=None, description="Optional expiry; unscoped/open-ended jobs are discouraged."
    )

    @model_validator(mode="after")
    def _validate_case_reference(self) -> "AnalysisScope":
        if self.purpose == AnalysisPurpose.LAW_ENFORCEMENT_REQUEST and not self.case_reference:
            raise ValueError(
                "AnalysisScope.case_reference is required when purpose=LAW_ENFORCEMENT_REQUEST."
            )
        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or datetime.now(timezone.utc)) > self.expires_at


class FaceEmbedding(BaseModel):
    """A single facial embedding extracted from one frame."""

    embedding_id: str = Field(default_factory=lambda: str(uuid4()))
    vector: list[float] = Field(..., description="Embedding vector, dimensionality set by the model backend.")
    source_frame_timestamp_ms: int = Field(..., ge=0)
    source_video_id: str
    detection_confidence: float = Field(..., ge=0.0, le=1.0, description="Face detector's confidence, not identity-match confidence.")


class IdentityMatchCandidate(BaseModel):
    """One candidate match returned from a vector similarity search."""

    candidate_identity_id: str
    cosine_similarity: float = Field(..., ge=-1.0, le=1.0)

    @property
    def meets_confidence_threshold(self) -> bool:
        return self.cosine_similarity >= IDENTITY_MATCH_CONFIDENCE_THRESHOLD


class IdentityMatchResult(BaseModel):
    """
    Normalized output of an identity-matching query.

    Per Rule 3: the agent MUST NOT hallucinate identity details. This shape
    makes an unconfident result structurally impossible to present as a
    confirmed identity — `is_confident_match` is derived, not caller-set,
    and `top_candidates` below threshold are retained (not discarded) so
    the caller can render "possible matches, unconfirmed" rather than
    silence, without ever promoting them to a claim.
    """

    query_embedding_id: str
    scope_id: str
    top_candidates: list[IdentityMatchCandidate] = Field(default_factory=list)
    searched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def best_candidate(self) -> IdentityMatchCandidate | None:
        if not self.top_candidates:
            return None
        return max(self.top_candidates, key=lambda c: c.cosine_similarity)

    @property
    def is_confident_match(self) -> bool:
        best = self.best_candidate
        return best is not None and best.meets_confidence_threshold

    def to_agent_summary(self) -> str:
        """
        The ONLY sanctioned way this result should be turned into text an
        LLM sees or a user reads. Callers should not hand-roll their own
        formatting of match results elsewhere, so this uncertainty framing
        can't be silently dropped at some other call site.
        """
        best = self.best_candidate
        if best is None:
            return "No candidate matches found."
        if best.meets_confidence_threshold:
            return (
                f"Likely match (similarity={best.cosine_similarity:.2f}): "
                f"identity_id={best.candidate_identity_id}. "
                f"Above the {IDENTITY_MATCH_CONFIDENCE_THRESHOLD} confidence threshold, "
                f"but this is a vector-similarity estimate, not a verified identity — "
                f"confirm through an independent source before treating it as fact."
            )
        return (
            f"UNCERTAIN — best candidate similarity={best.cosine_similarity:.2f} is below the "
            f"{IDENTITY_MATCH_CONFIDENCE_THRESHOLD} confidence threshold. Do not present this as "
            f"an identified person; report only that no confident match was found."
        )
