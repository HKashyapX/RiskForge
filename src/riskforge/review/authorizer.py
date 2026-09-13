"""Reviewer authorization policies.

RiskForge's authorization posture is deny-by-default at two layers:

1. The API boundary checks the authenticated principal's roles against the
   deployment's configured reviewer roles (``riskforge.api.dependencies``).
2. The review service re-checks every command through the injected
   ``ReviewerAuthorizer``.  The composer injects an explicit subject
   allow-list; when no allow-list is configured, every review is denied.

The legacy ``AllowAllReviewerAuthorizer`` is retained strictly for
development/test fixtures and must never be composed in a deployed runtime.
"""

from __future__ import annotations

from riskforge.persistence.models import ReviewAction


class SubjectAllowlistReviewerAuthorizer:
    """Permit review actions only for explicitly allow-listed reviewer ids.

    Constructed by the composer from ``RISKFORGE_REVIEWER_SUBJECTS``
    (comma-separated subject ids).  An empty/unset allow-list denies every
    review — failing closed rather than trusting an implicit default.
    """

    def __init__(self, allowed_subjects: set[str] | None = None) -> None:
        self._allowed_subjects = allowed_subjects or set()

    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        return reviewer_id in self._allowed_subjects


class DenyAllReviewerAuthorizer:
    """Deny every review action unconditionally (fail-closed default)."""

    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        return False


class AllowAllReviewerAuthorizer:
    """DEVELOPMENT/TEST ONLY.  Permit any authenticated reviewer.

    Never compose this in a deployed runtime; it exists so existing
    integration fixtures can exercise the review workflow without an
    identity provider.
    """

    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        return True
