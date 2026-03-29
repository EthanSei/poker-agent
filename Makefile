.PHONY: dev install test lint format typecheck check play run

dev:
	pip install -e ".[dev]"

install:
	pip install -e .

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy poker/

check: lint typecheck test

play:
	python -m poker.cli

run:
	uvicorn poker.api.app:app --reload
