"""Chunking, embedding and similarity ranking for the project chat's retrieval step.

No vector database: per-project corpora are small (tens to low hundreds of
chunks), so a plain JSON float array per chunk plus a pure-Python cosine
similarity scan is the right trade-off — it keeps the whole feature portable
to SQLite in tests without a Postgres `vector` extension, and is fast enough
at this scale without needing an ANN index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from medical_research_agent.llm.factory import get_embeddings_model

CHUNK_SIZE_CHARS = 1500
CHUNK_OVERLAP_CHARS = 150
DEFAULT_TOP_K = 8

# Cosine similarity for genuinely related short biomedical text (OpenAI
# text-embedding-3-small / Gemini text-embedding-004) typically clears
# 0.5+; unrelated pairs usually sit below ~0.2-0.3. 0.3 is a deliberately
# conservative floor — it favors not refusing a reasonably-phrased question
# over aggressively gating it, while still catching clearly out-of-corpus
# topics. Tune if real usage shows it's too loose/tight in either direction.
MIN_RELEVANCE_SCORE = 0.3


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """Split text into overlapping fixed-size chunks, breaking on paragraph
    boundaries where possible so a chunk doesn't cut off mid-sentence.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
        else:
            # A single paragraph longer than chunk_size: hard-split it with overlap.
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end])
                start = end - overlap if end < len(paragraph) else end
            current = ""
    if current:
        chunks.append(current)
    return chunks


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured embedding provider."""
    if not texts:
        return []
    model = get_embeddings_model()
    return await model.aembed_documents(texts)


async def embed_query(text: str) -> list[float]:
    """Embed a single query string (e.g. a chat question)."""
    model = get_embeddings_model()
    return await model.aembed_query(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class ScoredChunk:
    """A retrieved chunk paired with its similarity score, for ranking/display."""

    document_id: str
    content: str
    score: float


def top_k_chunks(
    query_embedding: list[float],
    chunks: list[tuple[str, str, list[float]]],
    k: int = DEFAULT_TOP_K,
) -> list[ScoredChunk]:
    """Rank ``(document_id, content, embedding)`` triples by similarity to the
    query and return the top ``k``.
    """
    scored = [
        ScoredChunk(
            document_id=doc_id,
            content=content,
            score=cosine_similarity(query_embedding, emb),
        )
        for doc_id, content, emb in chunks
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]
