.PHONY: install lint fmt format-check typecheck docs test up down logs seed reset

install:
	uv sync

lint:
	uv run ruff check

fmt:
	uv run ruff format

format-check:
	uv run ruff format --check

typecheck:
	uv run ty check src/

docs:
	uv run zensical build

test:
	uv run pytest

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

seed:
	uv run python scripts/seed.py

reset:
	uv run python scripts/reset.py
