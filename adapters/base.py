"""Abstract storage contract so the vector backend can be swapped later.

Any future backend (Pinecone, Weaviate, Qdrant, another Postgres schema, ...)
only needs to implement `VectorStorageInterface`; the engine never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk returned by a similarity search.

    `distance` is a cosine distance: 0.0 = identical direction,
    up to 2.0 = opposite direction. Lower is more similar.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


class VectorStorageInterface(ABC):
    """Repository interface for persistent vector storage."""

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection and ensure the schema/indexes exist."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the connection pool and clean up resources."""

    @abstractmethod
    def insert_chunk(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Persist one chunk (idempotent upsert keyed on `chunk_id`)."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the `limit` nearest chunks to `query_embedding` (best first)."""


# Alias kept for readability: concrete implementations are named "*Adapter".
VectorStorageAdapter = VectorStorageInterface
