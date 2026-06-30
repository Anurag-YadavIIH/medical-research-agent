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

> **Note:** the screenshots below predate a frontend UX refresh (light theme,
> reading links per study, an at-a-glance evidence summary, the Raw JSON tab
> removed, free-text search keywords). They still reflect real pipeline output
> shape; new screenshots are pending a retake against the refreshed UI.

These are real renders of the live Streamlit UI, driven against a genuine run
of the LangGraph pipeline (real PubMed/CrossRef data for 3 actual randomized
trials on corneal cross-linking, recorded as fixtures — see
[`evaluations/README.md`](evaluations/README.md)). The LLM steps in this
particular run used the evaluation harness's deterministic stub rather than a
live model call, since no LLM key is provisioned in the environment that
generated these — see [Verification status](#verification-status) below for
exactly what that does and doesn't prove.

| Evidence Summary | Study Comparison |
|---|---|
| ![Evidence Summary](docs/screenshots/01_evidence_summary.png) | ![Study Comparison](docs/screenshots/02_study_comparison.png) |

| Study Details | References |
|---|---|
| ![Study Details](docs/screenshots/03_study_details.png) | ![References](docs/screenshots/04_references.png) |

## Evaluation methodology

`evaluations/` (Phase 7) implements four metrics as pure functions over a
run's `machine_json`, each against either real recorded fixtures or crafted
unit-test inputs — never live calls by default:

- **citation_completeness** — every PMID/DOI cited in the narrative or listed
  as a reference must exist in the retrieved set. A fabricated citation is a
  named hard failure; scoring 1.0 while citing one is structurally impossible.
- **extraction_accuracy** — extracted fields vs. a labelled gold set: exact
  match for `study_design`/`sample_size`, fuzzy (`difflib` ratio ≥ 0.6) for
  free text.
- **evidence_consistency** — the assigned evidence level must agree with the
  real `deterministic_level()` mapping (reused, not re-implemented) from
  `publication_types`. An unclassifiable publication type is surfaced as a
  gap, not silently passed.
- **hallucination_check** — every PMID/DOI anywhere in the output, plus every
  sample-size/p-value-shaped numeric claim in the narrative, must trace back
  to retrieved data.

```bash
make eval                      # fixtures only — no network, no LLM key needed
make eval ARGS="--live"        # real NCBI/CrossRef + a real configured LLM key
```

Reports land in `evaluations/reports/<timestamp>.{json,md}`. Full detail,
matching rules, and the fixture/gold-case layout: see
[`evaluations/README.md`](evaluations/README.md).

## Verification status

What's actually been exercised, versus what's still pending a real LLM key:

| Claim | Status | How it's verified |
|---|---|---|
| Pipeline wiring (7 agents → graph → API → DB) | ✅ Verified | 120 automated tests, real `build_research_graph()` run against fixtures (`tests/test_evaluation_runner.py`, `evaluations/`) |
| Anti-fabrication: no citation survives outside the retrieved set | ✅ Verified | Enforced in code (`summary.py`'s `_strip_fabricated_citations`, `study_comparator.py`'s PMID filtering), tested directly with adversarial inputs |
| Deterministic evidence-level mapping | ✅ Verified | Reused (not duplicated) across the agent and the evaluator; parametrized tests covering every level + edge cases (bare "Review", empty/unmapped types) |
| Persistence transaction + rollback on failure | ✅ Verified | Direct repository-level test forces a commit failure and asserts rollback + no partial data survives |
| 503 pre-flight on missing LLM key, 500 on pipeline/DB failure, 200-with-warnings on partial success | ✅ Verified | API-level tests against all four states |
| Docker Compose stack (4 services, migration-on-start, healthchecks) | ✅ Verified | Brought up for real this session; `/health`, `/docs`, the 503 path, and frontend↔backend connectivity all checked against running containers |
| Real PubMed/CrossRef HTTP integration | ✅ Verified | `services/pubmed.py` and `services/crossref.py` tested against respx-mocked *and* genuinely recorded real responses (`evaluations/fixtures/`) |
| **End-to-end run with a real LLM** (query understanding, abstract extraction, evidence evaluation, study comparison, narrative summary) | ✅ Verified | Two live `POST /research` calls on Groq/llama-3.3-70b-versatile across 2 real clinical questions, 20 total real PubMed studies retrieved and extracted. 0 fabricated citations (every cited PMID traced to the retrieved set), 0 hallucinated numeric claims, `warnings: []` on both runs. `sample_size` schema fix confirmed on 3 separate review-type studies (correctly left `sample_size` null with verbatim text in `sample_size_description`) with zero Groq 400s. This proves Groq narrative generation works end-to-end — it does not exercise OpenAI (held in reserve, not run live) or Gemini (requires a billing-enabled project; free tier returns `limit: 0`, confirmed on two separate keys). |
| Live NCBI rate-limit behavior under sustained real traffic | ⏳ **Pending sustained live use** | The pacing logic is implemented and unit-tested in isolation; it hasn't been observed under real multi-request load. |

## Security notes

A self-audit (secret scanning across full git history, dependency CVE scan,
API/Docker hardening review) was run before publishing. Conscious, documented
trade-offs rather than oversights:

- **CORS is open by design** (`allow_origins=["*"]` in `api/main.py`). There is
  no cookie/session/`Authorization`-header auth anywhere in the codebase, so
  `allow_credentials` is left at its default `False` and a wildcard origin
  carries no cross-origin credential risk. This is appropriate for a
  public, read-mostly demo API — it is not a stand-in for a real auth layer.
- **`POST /research` has a minimal in-process rate limiter**
  (`api/rate_limit.py`): 10 requests/minute per client IP, returning 429 once
  exceeded. It's a local, in-memory sliding window — state resets on process
  restart and is **not** shared across replicas. It's enough to stop a single
  client from trivially burning LLM/NCBI quota against this single-instance
  deployment; it is **not** a substitute for real infra-level rate limiting
  (an API gateway or a Redis-backed limiter) in a multi-replica production
  deployment.
- **Postgres/Redis host port mappings in `docker-compose.yml`**
  (`5432:5432`, `6380:6379`) are a local-dev convenience only — neither
  `backend` nor `migrate` need them; both talk to `postgres:5432`/`redis:6379`
  over the internal Compose network. Drop these `ports:` entries in any real
  deployment.
- **Base-image CVEs**: Docker Scout flags a handful of HIGH/CRITICAL CVEs
  (`perl`, `wheel`, `quinn-proto`) that come from the `python:3.11-slim`
  (Debian) base image and its build toolchain, not from this project's own
  dependencies (`pip-audit` against the resolved `uv.lock` set reports zero
  known vulnerabilities). These are tracked for resolution on the next
  `python:3.11-slim` base image bump rather than worked around here.
- **No real secret has ever been committed** to this repo, on any branch, at
  any point in its history (verified by a full `git log --all -p` history
  scan plus a tree-walk for any `.env` file ever existing). `.env.example`
  ships only blank placeholders for every provider key.

## Future improvements

- Full-text retrieval (PMC/Unpaywall) beyond abstracts.
- GRADE-based grading and risk-of-bias tooling (RoB 2 / ROBINS-I cues).
- Parallel fan-out for per-study extraction with LangGraph map-reduce.
- Human-in-the-loop review checkpoints and exportable PRISMA-style flow.

## Build phases

✅ 1. Scaffolding · ✅ 2. PubMed + CrossRef · ✅ 3. LangGraph agents ·
✅ 4. FastAPI · ✅ 5. Streamlit · ✅ 6. Docker · ✅ 7. Evaluation ·
✅ 8. Testing + docs.

## License

MIT.
