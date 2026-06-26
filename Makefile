.PHONY: install dev frontend test lint format type up down logs migrate

install:        ## Install runtime + dev + frontend deps with uv (fallback: pip)
	uv sync --extra dev --extra frontend || pip install -e ".[dev,frontend]"

dev:            ## Run API with autoreload
	uvicorn medical_research_agent.api.main:app --reload --app-dir src

frontend:       ## Run the Streamlit UI locally
	streamlit run frontend/app.py

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
