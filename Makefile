.PHONY: install lint format typecheck test train run docker-build docker-run audit

install:
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app src tests

test:
	uv run pytest

train:
	uv run python -m src.training.train --input data/raw/creditcard.csv

run:
	uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

docker-build:
	docker build -t sentryguard:latest .

docker-run:
	docker run --rm -p 8000:8000 sentryguard:latest

audit:
	uv run bandit -r app src
	uv run pip-audit

