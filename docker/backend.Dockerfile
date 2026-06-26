FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install uv for fast, reproducible installs.
RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

EXPOSE 8000
CMD ["uvicorn", "medical_research_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
