"""NotebookLM-style Q&A grounded in a project's stored papers (PubMed
abstracts/findings + uploaded PDF chunks).

Mirrors the anti-fabrication discipline already established in
``agents/summary.py``: the LLM is restricted to citing ``[DOC: <id>]`` only
for ids explicitly present in its retrieved context, and anything outside
that set is stripped from the reply before it's returned. Errors from the
embedding/chat call are NOT swallowed here — they propagate to the API route,
which applies the same error-leakage-safe logging/500 pattern as
``POST /research`` (one place handling that, not duplicated per-service).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from medical_research_agent.citation_utils import DOC_CITATION, strip_fabricated_doc_citations
from medical_research_agent.database.project_repository import ProjectRepository
from medical_research_agent.llm.factory import get_chat_model
from medical_research_agent.models.project import ChatAnswer
from medical_research_agent.services.embeddings import (
    MIN_RELEVANCE_SCORE,
    ScoredChunk,
    embed_query,
    top_k_chunks,
)

_SYSTEM_PROMPT = (
    "You are answering questions about a specific research project's collected "
    "papers (PubMed abstracts/findings and/or uploaded PDF excerpts). You may "
    "cite findings ONLY using the exact notation '[DOC: <id>]' and ONLY for ids "
    "explicitly listed in the excerpts below — never cite, invent, or imply a "
    "finding, statistic, or source that is not in that list. If the provided "
    "excerpts don't contain enough information to answer the question, say so "
    "explicitly rather than guessing."
)

_NO_PAPERS_REPLY = (
    "This project doesn't have any papers yet — search PubMed or upload a PDF "
    "within this project first, then ask again."
)
_NOT_RELEVANT_REPLY = (
    "I couldn't find anything in this project's papers relevant to that "
    "question. Try rephrasing, or search/upload papers covering this topic first."
)


def _build_context(ranked: list[ScoredChunk]) -> str:
    lines = ["Relevant excerpts from this project's papers (cite using [DOC: <id>]):", ""]
    for chunk in ranked:
        lines.append(f"[DOC: {chunk.document_id}]")
        lines.append(chunk.content)
        lines.append("")
    return "\n".join(lines)


async def answer_project_question(
    repo: ProjectRepository, project_id: str, message: str
) -> ChatAnswer:
    """Answer a chat question grounded only in this project's stored chunks."""
    chunks = await repo.get_project_chunks(project_id)
    if not chunks:
        return ChatAnswer(reply=_NO_PAPERS_REPLY, cited_document_ids=[])

    query_embedding = await embed_query(message)
    ranked = top_k_chunks(query_embedding, chunks)
    relevant = [chunk for chunk in ranked if chunk.score >= MIN_RELEVANCE_SCORE]
    if not relevant:
        # Nothing clears the relevance bar — decline structurally, without ever
        # calling the chat model. Cheaper, and removes reliance on the model
        # choosing to say "I don't know" via prompt instruction alone.
        return ChatAnswer(reply=_NOT_RELEVANT_REPLY, cited_document_ids=[])

    valid_doc_ids = {chunk.document_id for chunk in relevant}
    context = _build_context(relevant)

    model = get_chat_model()
    response = await model.ainvoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"{context}\nQuestion: {message}"),
        ]
    )
    raw_reply = str(response.content)

    cleaned_reply, _fabricated = strip_fabricated_doc_citations(raw_reply, valid_doc_ids)
    cited_document_ids = sorted(set(DOC_CITATION.findall(cleaned_reply)))
    return ChatAnswer(reply=cleaned_reply, cited_document_ids=cited_document_ids)
