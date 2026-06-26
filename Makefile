.PHONY: install dev frontend test lint format type up down logs migrate eval

install:        ## Install runtime + dev + frontend deps with uv (fallback: pip)
	uv sync --extra dev --extra frontend || pip install -e ".[dev,frontend]"

dev:            ## Run API with autoreload
	uvicorn medical_research_agent.api.main:app --reload --app-dir src

frontend:       ## Run the Streamlit UI locally
	streamlit run frontend/app.py

test:           ## Run tests with coverage
	pytest

lint:           ## Lint with ruff
	ruff check src tests evaluations

format:         ## Format with black + ruff
	black src tests evaluations && ruff check --fix src tests evaluations

type:           ## Static type-check
	mypy src evaluations

up:             ## Start the full stack
	docker compose up --build

down:           ## Stop the stack
	docker compose down -v

logs:           ## Tail backend logs
	docker compose logs -f backend

migrate:        ## Apply DB migrations
	alembic upgrade head

eval:           ## Run the evaluation harness against fixtures (add ARGS="--live" for a real run)
	python -m evaluations.cli $(ARGS)
