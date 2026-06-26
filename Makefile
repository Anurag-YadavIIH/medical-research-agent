.PHONY: install dev test lint format type up down logs migrate

install:        ## Install runtime + dev deps with uv (fallback: pip)
	uv sync --extra dev || pip install -e ".[dev]"

dev:            ## Run API with autoreload
	uvicorn medical_research_agent.api.main:app --reload --app-dir src

test:           ## Run tests with coverage
	pytest

lint:           ## Lint with ruff
	ruff check src tests

format:         ## Format with black + ruff
	black src tests && ruff check --fix src tests

type:           ## Static type-check
	mypy src

up:             ## Start the full stack
	docker compose up --build

down:           ## Stop the stack
	docker compose down -v

logs:           ## Tail backend logs
	docker compose logs -f backend

migrate:        ## Apply DB migrations
	alembic upgrade head
