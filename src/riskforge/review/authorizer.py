"""Default reviewer authorization policy."""

from __future__ import annotations

from riskforge.persistence.models import ReviewAction


class AllowAllReviewerAuthorizer:
    """Permit any authenticated reviewer to perform any review action.

    This is the default authorization policy for RiskForge.  It trusts that
    the authentication boundary has already verified the caller's identity
    and that only authenticated principals reach the review workflow.

    Production deployments may replace this with a role-based or
    permission-scoped implementation by injecting a different
    ``ReviewerAuthorizer`` into the ``ReviewService``.
    """

    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        """Return ``True`` for every authenticated reviewer."""
        return True
