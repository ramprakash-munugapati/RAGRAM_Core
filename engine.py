"""RagramCoreEngine: the Naive RAG pipeline coordinator.

Pipeline: document -> chunk -> embed -> store (via adapter) and
query -> embed -> retrieve (cosine k-NN) -> synthesize (grounded LLM answer).

The engine depends only on abstractions (VectorStorageInterface,
EmbeddingProvider, LLMClient), so every layer is swappable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from adapters.base import RetrievedChunk, VectorStorageInterface
from chunking import recursive_character_split
from embeddings import EmbeddingProvider
from llm import LLMClient
from prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_context_block


@dataclass(frozen=True)
class QueryResult:
    """Outcome of a single user query."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    context_used: bool = False  # False => refusal path, LLM never invoked


class RagramCoreEngine:
    """Coordinates chunking, embedding, storage, retrieval, and synthesis."""

    def __init__(
        self,
        storage: VectorStorageInterface,
        embedder: EmbeddingProvider,
        synthesizer: LLMClient,
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        top_k: int = 5,
        max_distance: float | None = 0.5,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._synthesizer = synthesizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        # Cosine-distance cutoff: chunks farther than this are considered
        # irrelevant and dropped before synthesis (None disables filtering).
        self._max_distance = max_distance

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def ingest_document(
        self,
        document: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Chunk a document, embed every chunk, and store them via the adapter.

        Chunk ids are deterministic ("<doc_id>:<index>") so re-ingesting the
        same document overwrites stale chunks instead of duplicating them.
        Returns the list of stored chunk ids.
        """
        text = document.strip()
        if not text:
            raise ValueError("Cannot ingest an empty document.")

        chunks = recursive_character_split(text, self._chunk_size, self._chunk_overlap)
        vectors = self._embedder.embed_documents(chunks)

        document_id = uuid.uuid4().hex
        chunk_ids: list[str] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{document_id}:{index}"
            # Structural metadata travels with the vector for provenance
            # and (future) filtered retrieval.
            chunk_metadata: dict[str, Any] = {
                **(metadata or {}),
                "document_id": document_id,
                "chunk_index": index,
                "chunk_count": len(chunks),
            }
            self._storage.insert_chunk(chunk_id, chunk, vector, chunk_metadata)
            chunk_ids.append(chunk_id)
        return chunk_ids

    # ------------------------------------------------------------------ #
    # Retrieval + synthesis
    # ------------------------------------------------------------------ #

    def query(self, user_query: str) -> QueryResult:
        """Answer `user_query` strictly from the stored knowledge base."""
        question = user_query.strip()
        if not question:
            raise ValueError("Query cannot be empty.")

        # 1. Vectorize the user question.
        query_vector = self._embedder.embed_query(question)

        # 2. Pull the k nearest chunks by cosine distance.
        results = self._storage.similarity_search(query_vector, limit=self._top_k)

        # 3. Relevance filter: drop chunks beyond the distance cutoff.
        if self._max_distance is not None:
            results = [chunk for chunk in results if chunk.distance <= self._max_distance]

        # 4. Hard anti-hallucination guard: with no grounded context there is
        #    nothing legitimate to synthesize, so never call the LLM.
        if not results:
            return QueryResult(answer=NO_CONTEXT_ANSWER, sources=[], context_used=False)

        # 5. Synthesize a grounded answer from the retrieved chunks.
        user_prompt = (
            f"Retrieved context chunks:\n\n{build_context_block(results)}\n\n"
            f"User question: {question}"
        )
        answer = self._synthesizer.synthesize(SYSTEM_PROMPT, user_prompt)
        return QueryResult(answer=answer, sources=results, context_used=True)
