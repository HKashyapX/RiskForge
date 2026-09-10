"""Portable configuration shared by offline modeling commands."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

MODELING_DATA_DIR_ENV = "RISKFORGE_MODELING_DATA_DIR"
DEFAULT_MODELING_DATA_DIR = Path("data") / "processed"


def modeling_data_dir(
    explicit: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the modeling data directory without platform-specific paths.

    An explicit value has highest priority, followed by
    ``RISKFORGE_MODELING_DATA_DIR`` and finally ``data/processed`` relative to
    the current working directory. The path is not required to exist so that
    callers can create output directories when appropriate.
    """

    env = os.environ if environment is None else environment
    configured = explicit if explicit is not None else env.get(MODELING_DATA_DIR_ENV)
    candidate = Path(configured) if configured else DEFAULT_MODELING_DATA_DIR
    return candidate.expanduser().resolve(strict=False)
