.PHONY: check test clean

check:
	python -m ruff check src/ tests/
	python -m mypy --strict src/ tests/

test:
	python -m pytest -v --tb=short tests/

clean:
	python scripts/clean.py
