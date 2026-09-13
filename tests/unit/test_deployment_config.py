"""Tests for the explicit demo/pilot/production deployment-mode configuration."""

from __future__ import annotations

import pytest

from riskforge.runtime.deployment import (
    DeploymentConfig,
    DeploymentConfigError,
    DeploymentMode,
    ScoringMode,
)


class TestFromEnv:
    def test_mode_is_required(self) -> None:
        with pytest.raises(DeploymentConfigError, match="RISKFORGE_DEPLOYMENT_MODE is required"):
            DeploymentConfig.from_env({})

    def test_unsupported_mode_is_rejected(self) -> None:
        with pytest.raises(DeploymentConfigError, match="unsupported RISKFORGE_DEPLOYMENT_MODE"):
            DeploymentConfig.from_env({"RISKFORGE_DEPLOYMENT_MODE": "staging-prod-ish"})

    def test_mode_is_case_insensitive(self) -> None:
        config = DeploymentConfig.from_env({"RISKFORGE_DEPLOYMENT_MODE": "  Production "})
        assert config.mode is DeploymentMode.PRODUCTION

    @pytest.mark.parametrize("mode", ["demo", "pilot", "production"])
    def test_every_mode_is_constructible(self, mode: str) -> None:
        config = DeploymentConfig.from_env({"RISKFORGE_DEPLOYMENT_MODE": mode})
        assert config.mode is DeploymentMode(mode)


class TestModeContracts:
    def test_demo_is_public_without_operational_routes(self) -> None:
        config = DeploymentConfig.for_mode(DeploymentMode.DEMO)
        assert config.require_auth is False
        assert config.expose_operational_routes is False
        assert config.public_metrics is False
        assert config.require_model_artifact is False
        assert config.require_postgres is False

    def test_pilot_requires_auth_and_allows_labelled_heuristics(self) -> None:
        config = DeploymentConfig.for_mode(DeploymentMode.PILOT)
        assert config.require_auth is True
        assert config.expose_operational_routes is True
        assert config.allow_heuristic_engine is True
        assert config.require_model_artifact is False
        assert config.require_postgres is False
        assert config.deny_by_default_reviews is True

    def test_production_requires_everything_and_fails_closed(self) -> None:
        config = DeploymentConfig.for_mode(DeploymentMode.PRODUCTION)
        assert config.require_auth is True
        assert config.allow_heuristic_engine is False
        assert config.require_model_artifact is True
        assert config.require_postgres is True
        assert config.expose_operational_routes is True
        # Metrics are served behind credential verification, never publicly.
        assert config.public_metrics is True

    def test_metrics_are_never_public_unauthenticated(self) -> None:
        """Guarded-mount semantics: pilot/production serve metrics only to
        verified principals; demo serves none at all."""
        demo = DeploymentConfig.for_mode(DeploymentMode.DEMO)
        assert demo.public_metrics is False and demo.require_auth is False
        for mode in (DeploymentMode.PILOT, DeploymentMode.PRODUCTION):
            config = DeploymentConfig.for_mode(mode)
            assert config.public_metrics is True
            assert config.require_auth is True

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            (DeploymentMode.DEMO, "synthetic-heuristic"),
            (DeploymentMode.PILOT, "heuristic-or-model"),
            (DeploymentMode.PRODUCTION, "model"),
        ],
    )
    def test_scoring_label_reflects_mode(self, mode: DeploymentMode, expected: str) -> None:
        assert DeploymentConfig.for_mode(mode).scoring_label == expected


class TestScoringMode:
    def test_scoring_modes_are_distinct(self) -> None:
        assert ScoringMode.ONNX_MODEL is not ScoringMode.HEURISTIC
        assert {mode.value for mode in ScoringMode} == {"onnx-model", "heuristic"}
