"""Architectural invariant tests.

These tests verify structural properties of the RiskForge codebase that are
not captured by individual unit or integration tests. They guard against
architectural drift by asserting negative invariants (things that must NOT
exist in the production source tree).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "riskforge"

# ---------------------------------------------------------------------------
# Invariant 1: No outbound HTTP client libraries in production code
# ---------------------------------------------------------------------------

# These imports indicate outbound HTTP communication. They are forbidden in
# production source code (dev/test dependencies like httpx are acceptable in
# tests but not in src/).
_FORBIDDEN_HTTP_IMPORTS = frozenset({
    "httpx",
    "requests",
    "ahttphttp",
    "aiohttp",
    "urllib3",
    "grpc",
    "grpcio",
    "websocket",
    "websockets",
    "http.client",
    "urllib.request",
})


def _collect_python_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Collect all .py files under *root*, excluding __pycache__."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in str(p))


def _extract_imports(filepath: pathlib.Path) -> list[str]:
    """Return all top-level import names from a Python file."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return imports


@pytest.mark.parametrize(
    "filepath",
    _collect_python_files(_SRC_ROOT),
    ids=lambda p: str(p.relative_to(_SRC_ROOT)),
)
def test_no_outbound_http_imports(filepath: pathlib.Path) -> None:
    """Production source code must not import outbound HTTP client libraries."""
    imports = _extract_imports(filepath)
    violations = sorted(set(imports) & _FORBIDDEN_HTTP_IMPORTS)
    assert not violations, (
        f"{filepath.relative_to(_SRC_ROOT)} imports forbidden HTTP client "
        f"libraries: {violations}. RiskForge has no outbound HTTP calls; "
        f"see authentication/CONTEXT.md for architectural rationale."
    )


# ---------------------------------------------------------------------------
# Invariant 2: No external service URL configuration in production code
# ---------------------------------------------------------------------------

# Patterns that suggest external service endpoint configuration.
_EXTERNAL_URL_PATTERNS = (
    "http://",
    "https://",
    "grpc://",
    "amqp://",
    "redis://",
    "kafka://",
    "mongodb://",
    "amqps://",
)


@pytest.mark.parametrize(
    "filepath",
    _collect_python_files(_SRC_ROOT),
    ids=lambda p: str(p.relative_to(_SRC_ROOT)),
)
def test_no_external_service_urls(filepath: pathlib.Path) -> None:
    """Production source code must not contain hardcoded external service URLs."""
    source = filepath.read_text(encoding="utf-8")
    violations: list[str] = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # Skip comments and docstrings (rough heuristic: lines starting with # or
        # inside triple-quoted strings are ok for documentation purposes)
        if stripped.startswith(("#", '"', "'")):
            continue
        for pattern in _EXTERNAL_URL_PATTERNS:
            if pattern in stripped:
                violations.append(f"  line {i}: {stripped[:120]}")
                break
    assert not violations, (
        f"{filepath.relative_to(_SRC_ROOT)} contains external service URL patterns:\n"
        + "\n".join(violations)
        + "\nRiskForge communicates only via inbound HTTP and direct database "
        "connections. See authentication/CONTEXT.md."
    )


# ---------------------------------------------------------------------------
# Invariant 3: No message queue / event bus dependencies in production code
# ---------------------------------------------------------------------------

_FORBIDDEN_MQ_IMPORTS = frozenset({
    "kombu",
    "celery",
    "rq",
    " dramatiq",
    "bull",
    "pika",
    "aiokafka",
    "confluent_kafka",
    "nats",
    "nats.aio",
})


@pytest.mark.parametrize(
    "filepath",
    _collect_python_files(_SRC_ROOT),
    ids=lambda p: str(p.relative_to(_SRC_ROOT)),
)
def test_no_message_queue_dependencies(filepath: pathlib.Path) -> None:
    """Production source code must not import message queue client libraries."""
    imports = _extract_imports(filepath)
    violations = sorted(set(imports) & _FORBIDDEN_MQ_IMPORTS)
    assert not violations, (
        f"{filepath.relative_to(_SRC_ROOT)} imports message queue libraries: "
        f"{violations}. RiskForge uses synchronous in-process communication."
    )
