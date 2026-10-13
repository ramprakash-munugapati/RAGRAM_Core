"""Storage adapters for RAGRAM Core (Repository/Adapter pattern)."""

from adapters.base import RetrievedChunk, VectorStorageInterface
from adapters.postgres_adapter import PostgresVectorAdapter

__all__ = ["RetrievedChunk", "VectorStorageInterface", "PostgresVectorAdapter"]
