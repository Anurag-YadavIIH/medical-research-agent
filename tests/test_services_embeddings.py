"""Tests for chunking and similarity ranking — pure functions, no live API calls."""

from __future__ import annotations

from medical_research_agent.services.embeddings import (
    chunk_text,
    cosine_similarity,
    top_k_chunks,
)


def test_chunk_text_short_text_is_a_single_chunk() -> None:
    assert chunk_text("A short abstract.") == ["A short abstract."]


def test_chunk_text_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_text_on_paragraph_boundaries() -> None:
    paragraphs = ["Paragraph one. " * 20, "Paragraph two. " * 20, "Paragraph three. " * 20]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, chunk_size=400, overlap=20)

    assert len(chunks) > 1
    assert all(len(c) <= 420 for c in chunks)  # small slack for boundary edge cases
    # No content lost — every paragraph's distinctive word appears somewhere.
    joined = " ".join(chunks)
    assert "one" in joined and "two" in joined and "three" in joined


def test_chunk_text_hard_splits_a_single_oversized_paragraph() -> None:
    text = "x" * 5000  # one giant "paragraph" with no double-newlines at all

    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_a_crash() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_top_k_chunks_ranks_by_similarity_descending() -> None:
    query = [1.0, 0.0]
    chunks = [
        ("doc-far", "unrelated", [0.0, 1.0]),
        ("doc-close", "relevant", [0.99, 0.01]),
        ("doc-mid", "somewhat relevant", [0.5, 0.5]),
    ]

    ranked = top_k_chunks(query, chunks, k=2)

    assert [c.document_id for c in ranked] == ["doc-close", "doc-mid"]
    assert ranked[0].score > ranked[1].score
