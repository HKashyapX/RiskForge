"""Explicit deployment modes: demo, pilot, production.

RiskForge has exactly three supported deployment modes.  The mode is chosen
once, through ``RISKFORGE_DEPLOYMENT_MODE``, and every safety-relevant default
(authentication, persistence, scoring engine, exposure) derives from it.
There is no "production-like but actually demo" ambiguity: each mode's
requirements are validated at configuration time and composition fails closed
when a requirement is unmet.

Mode contracts
--------------
demo
    Public.  Synthetic data only.  No operational uploads, no credentials,
    no model artifacts.  Operational routes are not exposed at all.
pilot
    Authenticated.  Operational routes enabled.  A heuristic/rule scorer is
    permitted but must be explicitly selected and is visibly labelled
    ``heuristic`` everywhere (readiness, API metadata).  A configured model
    artifact must load; a missing artifact is a hard error, never a silent
    fallback.
production
    Authenticated.  PostgreSQL required.  A validated ONNX model artifact,
    manifest, and record encoder are required.  Startup fails closed when
    authentication, database, artifact, manifest, or encoder setup is
    missing or invalid.  No heuristic fallback exists in this mode.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, ConfigDict


class DeploymentMode(str, Enum):
    DEMO = "demo"
    PILOT = "pilot"
    PRODUCTION = "production"


class ScoringMode(str, Enum):
    """What actually produced a score.  Never conflated in metadata."""

    ONNX_MODEL = "onnx-model"
    HEURISTIC = "heuristic"


class DeploymentConfigError(RuntimeError):
    """Raised when the requested deployment mode is impossible or unsafe."""


class DeploymentConfig(BaseModel):
    """Validated, fail-closed deployment configuration.

    Construct exclusively through :meth:`from_env`; direct construction is
    allowed for tests but ``from_env`` is the only path that reads the
    process environment.
    """

    model_config = ConfigDict(frozen=True)

    mode: DeploymentMode
    require_auth: bool
    allow_heuristic_engine: bool
    require_model_artifact: bool
    require_postgres: bool
    deny_by_default_reviews: bool
    expose_operational_routes: bool
    public_metrics: bool

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> DeploymentConfig:
        env = os.environ if environ is None else environ
        raw = env.get("RISKFORGE_DEPLOYMENT_MODE", "").strip().lower()
        if not raw:
            raise DeploymentConfigError(
                "RISKFORGE_DEPLOYMENT_MODE is required and must be one of: "
                "demo, pilot, production (refusing to guess a deployment mode)"
            )
        try:
            mode = DeploymentMode(raw)
        except ValueError as error:
            raise DeploymentConfigError(
                f"unsupported RISKFORGE_DEPLOYMENT_MODE: {raw!r} "
                "(expected demo, pilot, or production)"
            ) from error
        return cls.for_mode(mode)

    @classmethod
    def for_mode(cls, mode: DeploymentMode) -> DeploymentConfig:
        if mode is DeploymentMode.DEMO:
            # Public by design; operational capability is off, not merely
            # unauthenticated.  Heuristic scoring is the only engine (there
            # is no artifact to trust) and is labelled synthetic/heuristic.
            return cls(
                mode=mode,
                require_auth=False,
                allow_heuristic_engine=True,
                require_model_artifact=False,
                require_postgres=False,
                deny_by_default_reviews=True,
                expose_operational_routes=False,
                public_metrics=False,
            )
        if mode is DeploymentMode.PILOT:
            return cls(
                mode=mode,
                require_auth=True,
                allow_heuristic_engine=True,
                require_model_artifact=False,
                require_postgres=False,
                deny_by_default_reviews=True,
                expose_operational_routes=True,
                public_metrics=False,
            )
        return cls(
            mode=mode,
            require_auth=True,
            allow_heuristic_engine=False,
            require_model_artifact=True,
            require_postgres=True,
            deny_by_default_reviews=True,
            expose_operational_routes=True,
            public_metrics=False,
        )

    @property
    def scoring_label(self) -> str:
        """Human-readable scoring label surfaced via API and readiness."""
        if self.mode is DeploymentMode.DEMO:
            return "synthetic-heuristic"
        if self.allow_heuristic_engine and not self.require_model_artifact:
            return "heuristic-or-model"
        return "model"
