"""Project endpoints: named workspaces with scoped search history, uploaded
PDFs, and a NotebookLM-style chat grounded in the project's accumulated papers.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from medical_research_agent.api.rate_limit import enforce_rate_limit
from medical_research_agent.api.routes.research import run_research
from medical_research_agent.api.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    CreateProjectRequest,
    ProjectDetailResponse,
    ProjectDocumentResponse,
    ProjectHistoryItem,
    ProjectResponse,
    ResearchRequest,
    ResearchResponse,
)
from medical_research_agent.database.models import ProjectRecord
from medical_research_agent.database.project_repository import ProjectRepository
from medical_research_agent.database.session import get_session
from medical_research_agent.logging_config import get_logger
from medical_research_agent.models.report import EvidenceReport
from medical_research_agent.services.embeddings import chunk_text, embed_texts
from medical_research_agent.services.pdf_extraction import PdfExtractionError, extract_text
from medical_research_agent.services.project_chat import answer_project_question

router = APIRouter(prefix="/projects")
logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB — bounds embedding cost/time per upload
MAX_TITLE_LENGTH = 255  # matches typical filesystem filename limits; stored, never used as a path
# A sane ceiling on a project's RAG corpus — bounds cumulative embedding cost
# from repeated uploads/searches into the same project (a single upload is
# already bounded by MAX_UPLOAD_BYTES/pdf_extraction's page+char caps, but
# nothing previously stopped uploading many PDFs into one project).
MAX_DOCUMENTS_PER_PROJECT = 200
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024  # 1 MB


def _to_response(project: ProjectRecord) -> ProjectResponse:
    return ProjectResponse(id=project.id, name=project.name, created_at=project.created_at)


async def _get_project_or_404(repo: ProjectRepository, project_id: str) -> ProjectRecord:
    project = await repo.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No project found for project_id '{project_id}'.",
        )
    return project


@router.post("", response_model=ProjectResponse, dependencies=[Depends(enforce_rate_limit)])
async def create_project(
    request: CreateProjectRequest, session: AsyncSession = Depends(get_session)
) -> ProjectResponse:
    repo = ProjectRepository(session)
    project = await repo.create_project(request.name)
    return _to_response(project)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(session: AsyncSession = Depends(get_session)) -> list[ProjectResponse]:
    repo = ProjectRepository(session)
    return [_to_response(p) for p in await repo.list_projects()]


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: str, session: AsyncSession = Depends(get_session)
) -> ProjectDetailResponse:
    repo = ProjectRepository(session)
    project = await _get_project_or_404(repo, project_id)

    history = await repo.list_project_history(project_id)
    documents = await repo.list_documents(project_id)

    return ProjectDetailResponse(
        project=_to_response(project),
        history=[
            ProjectHistoryItem(
                query_id=q.id,
                question=q.question,
                created_at=q.created_at,
                studies=[s.payload for s in q.studies],
                report_markdown=q.summary.markdown if q.summary else "",
            )
            for q in history
        ],
        documents=[
            ProjectDocumentResponse(
                id=d.id, source=d.source, pmid=d.pmid, title=d.title, created_at=d.created_at
            )
            for d in documents
        ],
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, session: AsyncSession = Depends(get_session)) -> None:
    repo = ProjectRepository(session)
    deleted = await repo.delete_project(project_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No project found for project_id '{project_id}'.",
        )


@router.post(
    "/{project_id}/research",
    response_model=ResearchResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def research_in_project(
    project_id: str,
    request: ResearchRequest,
    session: AsyncSession = Depends(get_session),
) -> ResearchResponse:
    """Run the same evidence-synthesis pipeline as POST /research, scoped to a
    project: results are tagged with project_id, and newly-seen papers are
    best-effort embedded into the project's chat corpus.
    """
    repo = ProjectRepository(session)
    await _get_project_or_404(repo, project_id)

    response, report = await run_research(request, session, project_id=project_id)

    try:
        cap_warning = await _embed_new_studies(repo, project_id, report)
        if cap_warning:
            response.warnings = [*response.warnings, cap_warning]
    except Exception as exc:  # noqa: BLE001 - embedding failure must not fail the search
        logger.error(
            "projects.embedding_failed",
            project_id=project_id,
            error=str(exc),
            exc_type=type(exc).__name__,
            exc_info=True,
        )
        response.warnings = [
            *response.warnings,
            "Could not index these papers for project chat (embedding failed). "
            "Search results are unaffected, but chat may not see these papers yet.",
        ]

    return response


async def _embed_new_studies(
    repo: ProjectRepository, project_id: str, report: EvidenceReport
) -> str | None:
    """Embed newly-seen studies into the project's RAG corpus, capped at
    ``MAX_DOCUMENTS_PER_PROJECT``. Returns a warning if the cap meant some
    newly-retrieved papers couldn't be indexed, else ``None``.
    """
    existing_pmids = await repo.get_existing_pmids(project_id)
    extracted_by_pmid = {item.pmid: item for item in report.extracted}

    new_studies = [s for s in report.studies if s.pmid not in existing_pmids]
    if not new_studies:
        return None

    current_count = await repo.count_documents(project_id)
    remaining_budget = max(0, MAX_DOCUMENTS_PER_PROJECT - current_count)
    skipped_count = max(0, len(new_studies) - remaining_budget)
    new_studies = new_studies[:remaining_budget]

    warning = None
    if skipped_count:
        warning = (
            f"This project is at its {MAX_DOCUMENTS_PER_PROJECT}-paper cap — "
            f"{skipped_count} newly-retrieved paper(s) were not added to the chat corpus."
        )
    if not new_studies:
        return warning

    texts = []
    for study in new_studies:
        extracted = extracted_by_pmid.get(study.pmid)
        findings = extracted.main_findings if extracted else ""
        texts.append(f"{study.title}\n\n{study.abstract}\n\n{findings}".strip())

    embeddings = await embed_texts(texts)
    for study, text, embedding in zip(new_studies, texts, embeddings, strict=True):
        await repo.add_pubmed_document(project_id, study.pmid, study.title, text, embedding)

    return warning


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in fixed-size chunks, aborting as soon as the cumulative
    size exceeds ``max_bytes`` — caps memory use at roughly
    ``max_bytes + _UPLOAD_READ_CHUNK_BYTES`` regardless of how much a client
    sends, instead of buffering an arbitrarily large body fully before the
    size check (the previous behavior). Still not a true zero-buffering
    streaming guarantee — see the Security notes in README.md.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDF exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/{project_id}/documents",
    response_model=ProjectDocumentResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def upload_document(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> ProjectDocumentResponse:
    """Upload a paper PDF into a project's chat corpus."""
    repo = ProjectRepository(session)
    await _get_project_or_404(repo, project_id)

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only application/pdf uploads are supported.",
        )

    filename = file.filename or ""
    if len(filename) > MAX_TITLE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Filename exceeds {MAX_TITLE_LENGTH} characters.",
        )

    if await repo.count_documents(project_id) >= MAX_DOCUMENTS_PER_PROJECT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"This project already has {MAX_DOCUMENTS_PER_PROJECT} papers, "
            "its maximum. Delete the project or start a new one to add more.",
        )

    # Fast-path reject on a declared, honest Content-Length before reading
    # anything — the chunked bounded read below is the authoritative guard
    # for clients that omit or understate it.
    content_length = request.headers.get("content-length")
    content_length_known_oversized = (
        content_length is not None
        and content_length.isdigit()
        and int(content_length) > MAX_UPLOAD_BYTES
    )
    if content_length_known_oversized:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    data = await _read_upload_bounded(file, MAX_UPLOAD_BYTES)

    try:
        text = extract_text(data)
    except PdfExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    chunks = chunk_text(text)
    try:
        embeddings = await embed_texts(chunks)
    except Exception as exc:
        logger.error(
            "projects.upload_embedding_failed",
            project_id=project_id,
            error=str(exc),
            exc_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not index this PDF right now (embedding service unavailable). "
            "Please try again shortly.",
        ) from exc

    document = await repo.add_uploaded_document(
        project_id, file.filename or "uploaded.pdf", chunks, embeddings
    )
    return ProjectDocumentResponse(
        id=document.id,
        source=document.source,
        pmid=document.pmid,
        title=document.title,
        created_at=document.created_at,
    )


@router.post(
    "/{project_id}/chat",
    response_model=ChatResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def chat_with_project(
    project_id: str,
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    repo = ProjectRepository(session)
    await _get_project_or_404(repo, project_id)

    try:
        answer = await answer_project_question(repo, project_id, request.message)
    except Exception as exc:
        logger.error(
            "projects.chat_failed",
            project_id=project_id,
            error=str(exc),
            exc_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat failed unexpectedly. Please try again.",
        ) from exc

    await repo.save_chat_message(project_id, "user", request.message)
    await repo.save_chat_message(project_id, "assistant", answer.reply)

    return ChatResponse(reply=answer.reply, cited_document_ids=answer.cited_document_ids)


@router.get("/{project_id}/chat", response_model=list[ChatMessageResponse])
async def get_chat_history(
    project_id: str, session: AsyncSession = Depends(get_session)
) -> list[ChatMessageResponse]:
    repo = ProjectRepository(session)
    await _get_project_or_404(repo, project_id)

    history = await repo.get_chat_history(project_id)
    return [
        ChatMessageResponse(role=m.role, content=m.content, created_at=m.created_at)
        for m in history
    ]
