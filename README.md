# 🔬 Medical Research Agent

A multi-agent **evidence synthesis** system for clinicians and researchers. Ask a
clinical question and receive an evidence-based, **fully cited** summary generated
from biomedical literature.

> **⚠️ For research and educational purposes only. Clinical decisions should rely
> on professional judgment and full-text evidence review.** This system is an
> evidence-synthesis assistant, **not** a diagnosis or treatment engine. It never
> fabricates citations and explicitly states uncertainty when evidence is thin.

---

## Problem statement

Clinicians and researchers cannot keep pace with the volume of published
literature. Answering even a focused question ("What is the evidence for corneal
cross-linking in pediatric keratoconus?") requires searching PubMed, screening
abstracts, appraising study design and bias, reconciling conflicting findings,
and assembling references. This project automates that workflow with a transparent,
auditable pipeline of specialised agents — surfacing **levels of evidence**,
**study comparisons**, and **Vancouver-formatted references** rather than a single
opaque answer.

## Architecture

```mermaid
flowchart TD
    START([Clinical question]) --> QU[Query Understanding Agent]
    QU --> PS[PubMed Search Agent]
    PS --> CR[CrossRef Enrichment Agent]
    CR --> PR[Paper Reader Agent]
    PR --> EE[Evidence Evaluator Agent]
    EE --> SC[Study Comparator Agent]
    SC --> SU[Summary Agent]
    SU --> END([Report + machine JSON])

    PS -. NCBI E-utilities .-> EXT1[(PubMed)]
    CR -. REST .-> EXT2[(CrossRef)]
    subgraph Infra
      API[FastAPI] --- DB[(PostgreSQL)]
      API --- CACHE[(Redis)]
      UI[Streamlit] --- API
    end
```

**Stack:** Python 3.11 · FastAPI · LangGraph · OpenAI/Groq (provider abstraction)
· Streamlit · PostgreSQL · Redis · Docker Compose · pytest · ruff/black/mypy ·
optional LangSmith tracing.

## Agents

| Agent | Responsibility | Output |
|---|---|---|
| Query Understanding | Parse question into PICO + search strategy | `QueryUnderstanding` |
| PubMed Search | Retrieve studies via NCBI E-utilities | `list[Study]` |
| CrossRef Enrichment | Add DOI, citations, publisher, URL (best-effort) | enriched `Study` |
| Paper Reader | Extract design, sample, findings, limitations from abstracts | `ExtractedStudy` |
| Evidence Evaluator | Assign level-of-evidence + bias appraisal | `EvidenceAssessment` |
| Study Comparator | Reconcile agreement/conflict; identify strongest evidence | `StudyComparison` |
| Summary | Synthesise final report + Vancouver references + caveats | `EvidenceReport` |

## Installation

```bash
# 1. Clone and enter
git clone <repo-url> && cd medical-research-agent

# 2. Configure
cp .env.example .env        # fill in OPENAI_API_KEY / GROQ_API_KEY and NCBI_EMAIL

# 3a. Local dev (uv preferred)
uv sync --extra dev --extra frontend   # or: pip install -e ".[dev,frontend]"
make dev                               # API at http://localhost:8000/docs
make frontend                          # Streamlit at http://localhost:8501 (separate shell)

# 3b. Full stack
docker compose up --build   # API :8000 · Streamlit :8501 · Postgres · Redis
```

## Environment variables

See [`.env.example`](.env.example). Key ones: `DEFAULT_LLM_PROVIDER`,
`OPENAI_API_KEY`/`GROQ_API_KEY`, `NCBI_EMAIL` (required by NCBI) and `NCBI_API_KEY`
(higher rate limit), `CROSSREF_MAILTO`, `DATABASE_URL`, `REDIS_URL`, and the
`LANGCHAIN_*` LangSmith tracing toggles.

## Docker setup

```bash
# 1. Configure (once)
cp .env.example .env        # fill in OPENAI_API_KEY / GROQ_API_KEY and NCBI_EMAIL

# 2. Bring the stack up
docker compose up --build

# 3. Tear down (add -v to also drop the Postgres volume)
docker compose down [-v]
```

`docker compose up --build` runs five services:

| Service | Role |
|---|---|
| `postgres` | Database. Has a healthcheck (`pg_isready`). |
| `redis` | Cache. Has a healthcheck (`redis-cli ping`). |
| `migrate` | One-shot: runs `alembic upgrade head` once Postgres is healthy, then exits. Re-running the stack re-runs this — it's a no-op if the schema is already current. |
| `backend` | FastAPI app. Only starts once `migrate` has exited successfully (`depends_on: migrate: condition: service_completed_successfully`) and `postgres`/`redis` are healthy. Has its own healthcheck against `GET /health`. |
| `frontend` | Streamlit UI. Only starts once `backend` is healthy. Reaches the API via `BACKEND_URL=http://backend:8000`. |

Tables are created exclusively through this `migrate` step (`alembic upgrade
head`) — the containerized path never relies on `Base.metadata.create_all`.
Migrations are not baked into the image at build time; `docker/backend.Dockerfile`
only copies `alembic.ini` and `alembic/` into the image, and the actual
`alembic upgrade head` only runs at container start, as the `migrate` service's
command.

Once up: API at `http://localhost:8000` (`/docs` for Swagger), Streamlit at
`http://localhost:8501`. `GET /health` reports `"redis": "ok"` once the stack
is running (it shows `"unavailable"` if you run the API standalone with `make
dev` and no local Redis). `POST /research` returns a `503` with an actionable
message if no LLM provider key is set in `.env`.

To generate a new migration after changing the SQLAlchemy models:

```bash
docker compose up -d postgres   # or point DATABASE_URL at any reachable Postgres
DATABASE_URL=postgresql+asyncpg://mra:mra@localhost:5432/mra \
  uv run alembic revision --autogenerate -m "describe the change"
DATABASE_URL=postgresql+asyncpg://mra:mra@localhost:5432/mra \
  uv run alembic upgrade head   # verify it applies cleanly
```

## API

| Method | Path | Description |
|---|---|---|
| POST | `/research` | Run the synthesis pipeline for a question |
| GET | `/studies/{query_id}` | Retrieve persisted studies for a prior query |
| GET | `/health` | Liveness + dependency checks |
| GET | `/docs` | OpenAPI / Swagger UI |

## Screenshots

_Placeholder — add screenshots of the Streamlit tabs (Summary, Comparison,
Details, References, Raw JSON) here._

## Evaluation methodology

The evaluation module (`evaluations/`) measures citation completeness, extraction
accuracy against a labelled gold set, evidence-grade consistency, and hallucination
(every cited PMID/DOI must exist in the retrieved set). Reports are emitted as JSON
and Markdown. See [`evaluations/README.md`](evaluations/README.md).

## Future improvements

- Full-text retrieval (PMC/Unpaywall) beyond abstracts.
- GRADE-based grading and risk-of-bias tooling (RoB 2 / ROBINS-I cues).
- Parallel fan-out for per-study extraction with LangGraph map-reduce.
- Human-in-the-loop review checkpoints and exportable PRISMA-style flow.

## Build phases

1. ✅ Scaffolding · 2. PubMed + CrossRef · 3. LangGraph agents · 4. FastAPI ·
5. Streamlit · 6. Docker · 7. Evaluation · 8. Testing + docs.

## License

MIT.
