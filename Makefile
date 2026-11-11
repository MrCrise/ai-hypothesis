.PHONY: sync lint test up down license-check

sync:
	uv sync

lint:
	uv run ruff check .
	uv run mypy packages services

test:
	uv run pytest

license-check:
	uv run pip-licenses --summary
	uv run pip-licenses --fail-on="GPL;AGPL" > /dev/null
	uv run pip-licenses --format=markdown --output-file=THIRD_PARTY_LICENSES.md

up:
	docker compose up --build -d

down:
	docker compose down
