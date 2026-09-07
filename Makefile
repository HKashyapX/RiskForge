.PHONY: check test contracts-diff clean

check:
	ruff check src/ tests/
	mypy --strict src/ tests/

test:
	pytest -v --tb=short tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
