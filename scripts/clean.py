"""Remove common Python tooling caches without assuming a Unix shell."""

from __future__ import annotations

import shutil
from pathlib import Path


CACHE_DIRECTORIES = ("__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache")
SKIP_DIRECTORIES = {".git", ".venv"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    removed = 0
    for path in root.rglob("*"):
        if not path.is_dir() or path.name not in CACHE_DIRECTORIES:
            continue
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        shutil.rmtree(path)
        removed += 1
        print(f"Removed {path.relative_to(root)}")
    print(f"Removed {removed} cache directories.")


if __name__ == "__main__":
    main()
