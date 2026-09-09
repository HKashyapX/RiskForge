from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_gitignore_is_utf8_text_without_nul_bytes() -> None:
    content = (REPOSITORY_ROOT / ".gitignore").read_bytes()

    assert b"\x00" not in content
    content.decode("utf-8")


def test_gitignore_does_not_ignore_all_files() -> None:
    rules = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "*" not in rules
