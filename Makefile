.PHONY: install lint fmt typecheck test up down logs seed reset

install:
	uv sync --all-extras

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy

test:
	pytest

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

seed:
	python scripts/seed.py

reset:
	python scripts/reset.py
