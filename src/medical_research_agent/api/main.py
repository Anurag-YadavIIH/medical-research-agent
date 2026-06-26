"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medical_research_agent import __version__
from medical_research_agent.api.routes import health, research
from medical_research_agent.api.schemas import DISCLAIMER
from medical_research_agent.database.session import dispose_engine
from medical_research_agent.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Medical Research Agent",
        version=__version__,
        description=(
            "Multi-agent biomedical evidence synthesis.\n\n" f"**Disclaimer:** {DISCLAIMER}"
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(research.router, tags=["research"])
    return app


app = create_app()
