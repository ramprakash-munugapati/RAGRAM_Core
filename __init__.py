"""RAGRAM Core: vector-only Naive RAG baseline (pgvector + PostgreSQL)."""

from engine import QueryResult, RagramCoreEngine
from adapters.base import RetrievedChunk, VectorStorageInterface
from adapters.postgres_adapter import PostgresVectorAdapter

__all__ = [
    "QueryResult",
    "RagramCoreEngine",
    "RetrievedChunk",
    "VectorStorageInterface",
    "PostgresVectorAdapter",
]
