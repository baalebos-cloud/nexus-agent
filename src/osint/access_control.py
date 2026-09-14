"""
Authorization gate for OSINT identity-matching jobs.

`video_processor.py` and `face_recognizer.py` MUST call `require_valid_scope`
before running any identity-matching query. This is the structural
enforcement point referenced in osint/models.py — not caller discipline.
"""

from __future__ import annotations

import logging
from typing import Protocol

from src.osint.models import AnalysisScope

logger = logging.getLogger(__name__)


class ScopeError(Exception):
    """Raised when an OSINT job is attempted without a valid AnalysisScope."""


class IdentityVerifier(Protocol):
    """
    Verifies that `requested_by` on an AnalysisScope corresponds to a real,
    authenticated requester.

    Single-operator deployments can use `StaticIdentityVerifier` (accepts
    any non-empty string — matches today's threat model where the operator
    running this system is already trusted). Multi-user deployments should
    supply a verifier backed by the real auth system (session lookup, SSO
    claim, etc.) at `E2BSandboxManager`-style construction time in
    src/api/server.py — swapping the implementation doesn't touch this file
    or models.py.
    """

    def verify(self, requested_by: str) -> bool: ...


class StaticIdentityVerifier:
    """Default verifier: accepts any non-empty identity string. Appropriate
    only for single-operator/trusted-environment deployments."""

    def verify(self, requested_by: str) -> bool:
        return bool(requested_by and requested_by.strip())


_default_verifier: IdentityVerifier = StaticIdentityVerifier()


def require_valid_scope(
    scope: AnalysisScope | None, verifier: IdentityVerifier | None = None
) -> AnalysisScope:
    """
    Validate an AnalysisScope before any identity-matching work proceeds.

    Raises ScopeError (not a silent no-op, not a warning log) if scope is
    missing, expired, or its requester fails verification, so there is no
    code path where identity-matching runs unscoped or unattributed.
    """
    if scope is None:
        raise ScopeError(
            "No AnalysisScope provided. Identity-matching requires an authorized, "
            "audited AnalysisScope — see src/osint/models.py:AnalysisScope."
        )
    if scope.is_expired():
        raise ScopeError(
            f"AnalysisScope {scope.scope_id!r} expired at {scope.expires_at}. "
            "Request a new scope to continue."
        )

    verifier = verifier or _default_verifier
    if not verifier.verify(scope.requested_by):
        raise ScopeError(
            f"Requester {scope.requested_by!r} failed identity verification."
        )

    # Audit trail — every authorized OSINT job gets a structured log line.
    # This should be sent to durable, queryable storage in production, not
    # just process logs (open item for the API-integration pass).
    logger.info(
        "OSINT scope authorized: scope_id=%s purpose=%s requested_by=%s case_reference=%s",
        scope.scope_id,
        scope.purpose.value,
        scope.requested_by,
        scope.case_reference,
    )
    return scope

