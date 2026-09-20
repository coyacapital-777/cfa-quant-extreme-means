.PHONY: help install test lint demo clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies with uv"
	@echo "  make test     - Run pytest with coverage"
	@echo "  make lint     - Run ruff linting"
	@echo "  make demo     - Run demo script"
	@echo "  make clean    - Remove cache files"

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/ scripts/

demo:
	uv run python scripts/demo.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
